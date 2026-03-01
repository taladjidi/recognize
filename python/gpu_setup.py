"""GPU and threading configuration for TensorFlow.

Must be imported BEFORE tensorflow in all classifier scripts.
Reads RECOGNIZE_GPU and RECOGNIZE_CORES environment variables.
"""
import os
import sys


def configure():
    """Configure GPU visibility and TF threading based on env vars."""
    gpu_requested = os.environ.get('RECOGNIZE_GPU', 'false').lower() == 'true'
    cores = os.environ.get('RECOGNIZE_CORES', '0')

    if not gpu_requested:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Suppress TF info/warning logs (keep errors)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Now safe to import tensorflow
    import tensorflow as tf

    if cores and cores != '0':
        num_cores = int(cores)
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)

    if gpu_requested:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'TensorFlow using GPU: {[g.name for g in gpus]}', file=sys.stderr)
        else:
            print('WARNING: GPU requested but no GPU found by TensorFlow', file=sys.stderr)
    else:
        print('TensorFlow running on CPU', file=sys.stderr)

    return tf
