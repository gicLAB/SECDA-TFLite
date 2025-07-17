# Model Generation
This folder provides scripts and notebooks as examples to generate custom TFLite models for benchmarking SECDA-TFLite delegates and accelerators.

## Example 1: model_simple.ipynb
This notebook demonstrates how to generate a simple TFLite model using Keras and TensorFlow. You can modify the model architecture and parameters as needed.

## Example 2: conv_gen.py
This script generates a set of single layer convolutional models with various configurations. It creates TFLite models and saves them in the specified directory. The script also generates a JSON file containing the model configurations. The files are saved into the models sub folder.


## Example 3: tconv_exp.py
This script generates a set of transposed convolutional models with various configurations based on existing layers of known models. It creates TFLite models and saves them in the specified directory. The script also generates a JSON file containing the model configurations. The files are saved into the models sub folder.