"""One-time model conversion: TFJS GraphModel → TF SavedModel.

Converts all TF.js graph-model format models to TF SavedModel format
so they can be loaded with tf.saved_model.load() in Python.

This converter reads the TFJS model.json + binary weight shards directly,
avoiding the broken `tensorflowjs` pip package (incompatible with TF 2.16+
and numpy 2.x).

Usage:
    python convert_models.py [--models-dir ../models]

Note: movinet-a3 is already in SavedModel format — no conversion needed.
"""

import argparse
import json
import os
import sys

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.core.framework import types_pb2

MODELS_TO_CONVERT = [
    "efficientnetv2",
    "efficientnet_lite4",
    "landmarks_africa",
    "landmarks_asia",
    "landmarks_europe",
    "landmarks_north_america",
    "landmarks_south_america",
    "landmarks_oceania",
    "musicnn",
]

# TFJS dtype string → numpy dtype
DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
    "complex64": np.complex64,
}

# TF DType enum string → protobuf enum value
_TF_DTYPE_ENUM = {
    "DT_FLOAT": types_pb2.DT_FLOAT,
    "DT_HALF": types_pb2.DT_HALF,
    "DT_DOUBLE": types_pb2.DT_DOUBLE,
    "DT_INT32": types_pb2.DT_INT32,
    "DT_INT8": types_pb2.DT_INT8,
    "DT_QINT8": types_pb2.DT_QINT8,
}

# Ops that gained a required TArgs attr in TF 2.20+
# Only _FusedConv2D uses TArgs; _FusedMatMul and _FusedDepthwiseConv2dNative
# use args:num_args*T (types inferred from T attr, not TArgs)
_FUSED_OPS_NEEDING_TARGS = {"_FusedConv2D"}

# Activation op names decoded from fused_ops base64 values
_ACTIVATION_OPS = {"Relu", "Relu6", "Elu", "LeakyRelu"}


def load_tfjs_model(model_dir):
    """Load a TFJS graph-model: parse model.json, read weight shards.

    Returns:
        (graph_def_json, weights_dict, signature) where weights_dict maps
        weight name → numpy array.
    """
    model_json_path = os.path.join(model_dir, "model.json")
    with open(model_json_path) as f:
        manifest = json.load(f)

    assert manifest["format"] == "graph-model", (
        f"Expected graph-model format, got {manifest['format']}"
    )

    graph_def_json = manifest["modelTopology"]

    # Read all binary weight shards into a contiguous buffer
    weights_manifest = manifest["weightsManifest"]
    weights_dict = {}

    for group in weights_manifest:
        # Concatenate all shard files for this group
        shard_data = bytearray()
        for shard_path in group["paths"]:
            shard_file = os.path.join(model_dir, shard_path)
            with open(shard_file, "rb") as f:
                shard_data.extend(f.read())

        # Parse individual weights from the concatenated buffer
        offset = 0
        for weight_spec in group["weights"]:
            name = weight_spec["name"]
            shape = weight_spec["shape"]
            dtype_str = weight_spec["dtype"]

            dtype = DTYPE_MAP.get(dtype_str)
            if dtype is None:
                raise ValueError(f"Unsupported dtype: {dtype_str} for weight {name}")

            num_elements = 1
            for dim in shape:
                num_elements *= dim

            num_bytes = num_elements * np.dtype(dtype).itemsize
            raw = bytes(shard_data[offset : offset + num_bytes])
            offset += num_bytes

            arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
            weights_dict[name] = arr

    return graph_def_json, weights_dict, manifest.get("signature", {})


def graph_json_to_graphdef(graph_def_json):
    """Convert JSON graph topology to TensorFlow GraphDef protobuf."""
    from google.protobuf import json_format

    graph_def = tf.compat.v1.GraphDef()
    json_format.ParseDict(graph_def_json, graph_def)
    return graph_def


def _decompose_fused_depthwise(graph_def):
    """Decompose FusedDepthwiseConv2dNative into separate ops.

    The _FusedDepthwiseConv2dNative op in TF 2.20 has runtime issues with
    certain fused_ops (e.g. Relu6). Decompose into:
      DepthwiseConv2dNative + BiasAdd + activation

    Input structure of FusedDepthwiseConv2dNative:
      input[0] = data tensor, input[1] = filter, input[2..] = args (bias, etc.)
    """
    nodes_to_remove = set()
    nodes_to_add = []

    for node in graph_def.node:
        if node.op != "FusedDepthwiseConv2dNative":
            continue

        orig_name = node.name
        inputs = list(node.input)
        data_input = inputs[0]
        filter_input = inputs[1]
        bias_input = inputs[2] if len(inputs) > 2 else None

        # fused_ops are already decoded bytes in the protobuf
        fused_ops_raw = (
            list(node.attr["fused_ops"].list.s) if "fused_ops" in node.attr else []
        )
        fused_ops = [
            s.decode("utf-8") if isinstance(s, bytes) else s for s in fused_ops_raw
        ]

        # Get conv attributes
        strides = (
            list(node.attr["strides"].list.i)
            if "strides" in node.attr
            else [1, 1, 1, 1]
        )
        padding = node.attr["padding"].s if "padding" in node.attr else b"SAME"
        dilations = (
            list(node.attr["dilations"].list.i)
            if "dilations" in node.attr
            else [1, 1, 1, 1]
        )
        data_format = (
            node.attr["data_format"].s if "data_format" in node.attr else b"NHWC"
        )
        dtype_val = node.attr["T"].type if "T" in node.attr else types_pb2.DT_FLOAT

        nodes_to_remove.add(orig_name)

        # 1. DepthwiseConv2dNative node
        conv_name = orig_name + "/_conv"
        conv_node = tf.compat.v1.NodeDef()
        conv_node.name = conv_name
        conv_node.op = "DepthwiseConv2dNative"
        conv_node.input.append(data_input)
        conv_node.input.append(filter_input)
        conv_node.attr["T"].type = dtype_val
        for s in strides:
            conv_node.attr["strides"].list.i.append(s)
        conv_node.attr["padding"].s = padding
        for d in dilations:
            conv_node.attr["dilations"].list.i.append(d)
        conv_node.attr["data_format"].s = data_format
        nodes_to_add.append(conv_node)

        # Track current output name for chaining
        current_output = conv_name + ":0"

        # 2. BiasAdd if present
        if "BiasAdd" in fused_ops and bias_input:
            bias_name = orig_name + "/_bias_add"
            bias_node = tf.compat.v1.NodeDef()
            bias_node.name = bias_name
            bias_node.op = "BiasAdd"
            bias_node.input.append(current_output.rsplit(":", 1)[0])
            bias_node.input.append(bias_input)
            bias_node.attr["T"].type = dtype_val
            bias_node.attr["data_format"].s = data_format
            nodes_to_add.append(bias_node)
            current_output = bias_name + ":0"

        # 3. Activation op
        activation = None
        for op_name in fused_ops:
            if op_name in _ACTIVATION_OPS:
                activation = op_name
                break

        if activation:
            # Use the original name for the final node so downstream refs work
            act_node = tf.compat.v1.NodeDef()
            act_node.name = orig_name
            act_node.op = activation
            act_node.input.append(current_output.rsplit(":", 1)[0])
            act_node.attr["T"].type = dtype_val
            nodes_to_add.append(act_node)
        elif current_output.rsplit(":", 1)[0] != orig_name:
            # No activation — add Identity to preserve the original name
            id_node = tf.compat.v1.NodeDef()
            id_node.name = orig_name
            id_node.op = "Identity"
            id_node.input.append(current_output.rsplit(":", 1)[0])
            id_node.attr["T"].type = dtype_val
            nodes_to_add.append(id_node)

    if not nodes_to_remove:
        return 0

    # Rebuild node list: keep non-removed nodes, add new nodes at the end
    remaining = [n for n in graph_def.node if n.name not in nodes_to_remove]
    del graph_def.node[:]
    graph_def.node.extend(remaining)
    graph_def.node.extend(nodes_to_add)

    return len(nodes_to_remove)


def _patch_fused_op_attrs(graph_def):
    """Add missing TArgs attribute to _FusedConv2D for TF 2.20+ compatibility.

    TFJS models were saved with older TF versions where _FusedConv2D didn't
    require TArgs. TF 2.20 made it required. We infer TArgs from num_args and T.
    """
    patched = 0
    for node in graph_def.node:
        if node.op not in _FUSED_OPS_NEEDING_TARGS:
            continue
        if "TArgs" in node.attr:
            continue  # already has it

        # Get the data type from T attr
        t_dtype = node.attr.get("T")
        if t_dtype is None:
            continue
        dtype_val = t_dtype.type  # protobuf enum int

        # Get num_args
        num_args_attr = node.attr.get("num_args")
        num_args = int(num_args_attr.i) if num_args_attr else 0
        if num_args == 0:
            continue

        # Add TArgs as a list of num_args copies of T
        targs = node.attr["TArgs"]
        for _ in range(num_args):
            targs.list.type.append(dtype_val)
        patched += 1

    return patched


def _patch_graph_weights(graph_def, weights_dict):
    """Replace Const node tensor values in the GraphDef with loaded weights."""
    patched = 0
    for node in graph_def.node:
        if node.op != "Const":
            continue
        if node.name in weights_dict:
            arr = weights_dict[node.name]
            tensor_proto = tf.make_tensor_proto(arr, dtype=arr.dtype, shape=arr.shape)
            node.attr["value"].tensor.CopyFrom(tensor_proto)
            patched += 1
    return patched


def convert_model(models_dir, model_name):
    """Convert a single TFJS GraphModel to TF SavedModel."""
    input_dir = os.path.join(models_dir, model_name)
    output_dir = os.path.join(models_dir, f"{model_name}_saved")

    if not os.path.isdir(input_dir):
        print(f"SKIP: {input_dir} does not exist", file=sys.stderr)
        return False

    if os.path.isdir(output_dir) and os.path.exists(
        os.path.join(output_dir, "saved_model.pb")
    ):
        print(f"SKIP: {output_dir} already exists", file=sys.stderr)
        return True

    print(f"Converting {model_name}...", file=sys.stderr)

    # Step 1: Load TFJS model
    graph_def_json, weights_dict, signature = load_tfjs_model(input_dir)
    graph_def = graph_json_to_graphdef(graph_def_json)

    # Step 2: Patch the GraphDef for TF 2.20 compatibility
    n_weights = _patch_graph_weights(graph_def, weights_dict)
    n_decomposed = _decompose_fused_depthwise(graph_def)
    n_attrs = _patch_fused_op_attrs(graph_def)
    print(
        f"  Patched {n_weights} weights, decomposed {n_decomposed} fused DW ops, {n_attrs} TArgs attrs",
        file=sys.stderr,
    )

    # Step 3: Import graph and discover input/output tensors
    with tf.compat.v1.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")

        input_nodes = {}
        output_nodes = {}

        # Try to get inputs/outputs from TFJS signature
        if signature:
            for key, info in signature.get("inputs", {}).items():
                tensor_name = info.get("name", "")
                if tensor_name:
                    try:
                        input_nodes[key] = graph.get_tensor_by_name(tensor_name)
                    except (KeyError, ValueError):
                        pass

            for key, info in signature.get("outputs", {}).items():
                tensor_name = info.get("name", "")
                if tensor_name:
                    try:
                        output_nodes[key] = graph.get_tensor_by_name(tensor_name)
                    except (KeyError, ValueError):
                        pass

        # Fallback: find Placeholder ops as inputs
        if not input_nodes:
            for op in graph.get_operations():
                if op.type == "Placeholder":
                    input_nodes[op.name] = op.outputs[0]

        # Fallback: find leaf ops (no consumers) as outputs
        if not output_nodes:
            all_ops = graph.get_operations()
            consumed = set()
            for op in all_ops:
                for inp in op.inputs:
                    consumed.add(inp.op.name)
            for op in all_ops:
                if op.name not in consumed and op.type not in (
                    "Const",
                    "NoOp",
                    "Placeholder",
                    "Identity",
                ):
                    output_nodes[op.name] = op.outputs[0]

        if not input_nodes or not output_nodes:
            raise RuntimeError(
                f"Could not determine input/output nodes for {model_name}"
            )

        print(f"  Inputs:  {list(input_nodes.keys())}", file=sys.stderr)
        print(f"  Outputs: {list(output_nodes.keys())}", file=sys.stderr)

        # Step 4: Save as SavedModel with serving_default signature
        with tf.compat.v1.Session(graph=graph) as sess:
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(output_dir)

            sig_inputs = {}
            for key, tensor in input_nodes.items():
                sig_inputs[key] = tf.compat.v1.saved_model.utils.build_tensor_info(
                    tensor
                )

            sig_outputs = {}
            for key, tensor in output_nodes.items():
                sig_outputs[key] = tf.compat.v1.saved_model.utils.build_tensor_info(
                    tensor
                )

            prediction_signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_inputs,
                outputs=sig_outputs,
                method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME,
            )

            builder.add_meta_graph_and_variables(
                sess,
                [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={"serving_default": prediction_signature},
            )
            builder.save()

    print(f"  → {output_dir}", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert TFJS models to TF SavedModel")
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Path to the models directory",
    )
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    if not os.path.isdir(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    success = 0
    failed = 0

    for model_name in MODELS_TO_CONVERT:
        try:
            if convert_model(models_dir, model_name):
                success += 1
            else:
                failed += 1
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            print(f"ERROR converting {model_name}: {e}", file=sys.stderr)
            failed += 1

    print(
        f"\nConversion complete: {success} succeeded, {failed} failed", file=sys.stderr
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
