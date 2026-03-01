"""One-time model conversion: TFJS GraphModel → TF SavedModel.

Converts all TF.js graph-model format models to TF SavedModel format
so they can be loaded with tf.saved_model.load() in Python.

Usage:
    python convert_models.py [--models-dir ../models]

Models to convert:
    - efficientnetv2 → efficientnetv2_saved
    - efficientnet_lite4 → efficientnet_lite4_saved
    - landmarks_africa → landmarks_africa_saved
    - landmarks_asia → landmarks_asia_saved
    - landmarks_europe → landmarks_europe_saved
    - landmarks_north_america → landmarks_north_america_saved
    - landmarks_south_america → landmarks_south_america_saved
    - landmarks_oceania → landmarks_oceania_saved
    - musicnn → musicnn_saved

Note: movinet-a3 is already in SavedModel format — no conversion needed.
"""
import argparse
import os
import sys


MODELS_TO_CONVERT = [
    'efficientnetv2',
    'efficientnet_lite4',
    'landmarks_africa',
    'landmarks_asia',
    'landmarks_europe',
    'landmarks_north_america',
    'landmarks_south_america',
    'landmarks_oceania',
    'musicnn',
]


def convert_model(models_dir, model_name):
    """Convert a single TFJS GraphModel to TF SavedModel."""
    import subprocess

    input_dir = os.path.join(models_dir, model_name)
    output_dir = os.path.join(models_dir, f'{model_name}_saved')

    if not os.path.isdir(input_dir):
        print(f'SKIP: {input_dir} does not exist', file=sys.stderr)
        return False

    if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, 'saved_model.pb')):
        print(f'SKIP: {output_dir} already exists', file=sys.stderr)
        return True

    print(f'Converting {model_name}...', file=sys.stderr)

    # Use tensorflowjs_converter CLI: tfjs_graph_model → tf_saved_model
    result = subprocess.run([
        sys.executable, '-m', 'tensorflowjs.converters.converter',
        '--input_format=tfjs_graph_model',
        '--output_format=tf_saved_model',
        input_dir,
        output_dir,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f'  stderr: {result.stderr}', file=sys.stderr)
        raise RuntimeError(f'Conversion failed for {model_name}: {result.stderr}')

    print(f'  → {output_dir}', file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert TFJS models to TF SavedModel')
    parser.add_argument('--models-dir', default=os.path.join(os.path.dirname(__file__), '..', 'models'),
                        help='Path to the models directory')
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    if not os.path.isdir(models_dir):
        print(f'ERROR: Models directory not found: {models_dir}', file=sys.stderr)
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
            print(f'ERROR converting {model_name}: {e}', file=sys.stderr)
            failed += 1

    print(f'\nConversion complete: {success} succeeded, {failed} failed', file=sys.stderr)
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
