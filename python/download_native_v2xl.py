"""Build native EfficientNetV2-XL from Google's checkpoint and export as SavedModel.

Uses Google's official automl/efficientnetv2 code to build the model architecture
and load the ImageNet-21k-ft1k pretrained checkpoint weights.
"""
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOML_DIR = os.path.join(SCRIPT_DIR, '..', 'automl_effnetv2', 'efficientnetv2')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
CKPT_DIR = os.path.join(MODELS_DIR, 'efficientnetv2-xl-21k-ft1k')
OUTPUT_DIR = os.path.join(MODELS_DIR, 'efficientnetv2_native_saved')

sys.path.insert(0, AUTOML_DIR)

import tensorflow as tf
import numpy as np

# Enable memory growth before any model ops
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def build_and_export():
    print('Building EfficientNetV2-XL model...', file=sys.stderr)

    import effnetv2_model

    t0 = time.perf_counter()
    model = effnetv2_model.get_model(
        'efficientnetv2-xl',
        include_top=True,
        weights=CKPT_DIR,
        training=False,
    )
    t_load = time.perf_counter() - t0
    print(f'Model built and weights loaded in {t_load:.1f}s', file=sys.stderr)

    # Quick test
    dummy = np.random.uniform(-1, 1, (1, 512, 512, 3)).astype(np.float32)
    t0 = time.perf_counter()
    out = model(dummy, training=False)
    t_warmup = time.perf_counter() - t0
    print(f'Warmup inference: {t_warmup:.3f}s, output shape: {out.shape}', file=sys.stderr)

    # GPU memory after warmup
    try:
        import subprocess
        smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True, timeout=5
        )
        print(f'GPU memory after warmup: {smi.strip()} MB', file=sys.stderr)
    except Exception:
        pass

    # Steady-state benchmark
    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model(dummy, training=False)
    t_avg = (time.perf_counter() - t0) / n_iters * 1000
    print(f'Steady-state: {t_avg:.0f} ms/image (avg of {n_iters})', file=sys.stderr)

    # Export as SavedModel
    print(f'\nExporting to {OUTPUT_DIR}...', file=sys.stderr)
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    # Create a frozen SavedModel (variables → constants) for v1 Session compat
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512, 512, 3], dtype=tf.float32)])
    def serve(images):
        return model(images, training=False)

    # Get the concrete function and freeze variables to constants
    concrete = serve.get_concrete_function()
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen = convert_variables_to_constants_v2(concrete)
    frozen_graph = frozen.graph.as_graph_def()

    print(f'Frozen graph: {len(frozen_graph.node)} nodes', file=sys.stderr)

    # Count op types to compare with TFJS model
    from collections import Counter
    op_counts = Counter(n.op for n in frozen_graph.node)
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'  {op}: {count}', file=sys.stderr)
    n_fused_dw = sum(1 for n in frozen_graph.node if n.op == '_FusedDepthwiseConv2dNative')
    n_decomposed_dw = sum(1 for n in frozen_graph.node if n.op == 'DepthwiseConv2dNative')
    print(f'  _FusedDepthwiseConv2dNative: {n_fused_dw}', file=sys.stderr)
    print(f'  DepthwiseConv2dNative (decomposed): {n_decomposed_dw}', file=sys.stderr)

    # Save the frozen graph as a v1 SavedModel
    t0 = time.perf_counter()

    # Import frozen graph into v1 session
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Graph().as_default() as g:
        tf.import_graph_def(frozen_graph, name='')

    input_tensor = g.get_tensor_by_name(frozen.inputs[0].name)
    output_tensor = g.get_tensor_by_name(frozen.outputs[0].name)
    print(f'Input: {input_tensor.name}, Output: {output_tensor.name}', file=sys.stderr)

    with tf.compat.v1.Session(graph=g, config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(OUTPUT_DIR)
        input_info = tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor)
        output_info = tf.compat.v1.saved_model.utils.build_tensor_info(output_tensor)
        sig = tf.compat.v1.saved_model.build_signature_def(
            inputs={'images': input_info},
            outputs={'output': output_info},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={'serving_default': sig},
        )
        builder.save()

    t_export = time.perf_counter() - t0
    print(f'Exported frozen SavedModel in {t_export:.1f}s', file=sys.stderr)

    # Disk size
    total_size = 0
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f'SavedModel size: {total_size / 1024 / 1024:.1f} MB', file=sys.stderr)

    # Verify the exported SavedModel
    print('\nVerifying exported model...', file=sys.stderr)
    loaded = tf.saved_model.load(OUTPUT_DIR)
    sig = loaded.signatures['serving_default']
    print(f'Input: {list(sig.structured_input_signature[1].keys())}', file=sys.stderr)
    print(f'Output: {list(sig.structured_outputs.keys())}', file=sys.stderr)

    out2 = sig(tf.constant(dummy))
    out2_val = list(out2.values())[0].numpy()
    diff = np.max(np.abs(out.numpy() - out2_val))
    print(f'Max output diff (model vs loaded): {diff:.6e}', file=sys.stderr)
    print('\nDone!', file=sys.stderr)


if __name__ == '__main__':
    build_and_export()
