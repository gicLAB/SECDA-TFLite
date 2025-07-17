import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging

logging.getLogger("tensorflow").disabled = True
import tensorflow as tf
import numpy as np
import json


def build_tconv_model(params, mdir):
    """params = [stride_x, stride_y, filters, kernel_size, in1, in2, in3, padding_val]"""

    stride_x = params[0]
    stride_y = params[1]
    filters = params[2]
    kernel_size = params[3]
    in1 = params[4]
    in2 = params[5]
    in3 = params[6]
    padding_val = params[7]
    name = params[8]
    out3 = filters
    if padding_val == "same":
        out1 = in1 * stride_x
        out2 = in2 * stride_y
    else:
        out1 = ((in1 - 1) * stride_x) + kernel_size
        out2 = ((in2 - 1) * stride_y) + kernel_size

    print(f"Input: {in1}x{in2}x{in3}")
    print(f"Output: {out1}x{out2}x{out3}")
    print(f"Stride: {stride_x}x{stride_y}")
    inputs = tf.keras.Input((in1, in2, in3))
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=(stride_x, stride_y),
        padding=padding_val,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        use_bias=True,
    )(inputs)
    tconv = tf.keras.Model(inputs, x)
    tconv.compile(optimizer="sgd", loss="mse")
    tconv.fit(
        np.random.rand(1, in1, in2, in3),
        np.random.rand(1, out1, out2, out3),
        batch_size=1,
        epochs=2,
    )
    os.makedirs(os.path.join(mdir), exist_ok=True)

    lamyield = lambda: [
        [np.random.rand(1, in1, in2, in3).astype(np.float32)] for x in range(15)
    ]
    converter = tf.lite.TFLiteConverter.from_keras_model(tconv)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lamyield
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    file_s = f"{name}.tflite"
    with open(mdir + file_s, "wb") as f:
        f.write(tflite_model)
    os.system(f"rm -rf {mdir}tf")
    return file_s


def generate_json_file(layers, filename, name, layer_type):
    """
    Generates a JSON file with the given layers, filename, name, and layer type.

    Args:
        layers (list): List of layer parameters.
        filename (str): Output JSON file path.
        name (str): Name of the model.
        layer_type (str): Type of the layer (e.g., "TRANSPOSE_CONV").
    """

    data = {
        name: {
            "layers": [layer[8] for layer in layers],
            "type": layer_type,
        }
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"JSON file generated: {filename}")


def main():
    params = []

    # Define each layer individually
    #  [s, s, oc, ks, iw, ih, ic, "same"]
    dcgan_layer1 = [2, 2, 512, 5, 4, 4, 1024, "same", "dcgan_layer1"]
    dcgan_layer2 = [2, 2, 256, 5, 8, 8, 512, "same", "dcgan_layer2"]
    dcgan_layer3 = [2, 2, 128, 5, 16, 16, 256, "same", "dcgan_layer3"]
    dcgan_layer4 = [2, 2, 3, 5, 32, 32, 128, "same", "dcgan_layer4"]
    dcgan_layers = [dcgan_layer1, dcgan_layer2, dcgan_layer3, dcgan_layer4]

    # combine all the models into one list
    layers = dcgan_layers

    mdir = "models/tconv_exp/"
    name = "tconv_exp"

    for layer in layers:
        build_tconv_model(layer, mdir)

    # generate_python_list(layers, f"configs/{name}.json", name)
    generate_json_file(layers, f"configs/{name}.json", name, "TRANSPOSE_CONV")

    print(f"Generated {len(params)} models")
    print(f"Saved to ./configs/{name}.json")


if __name__ == "__main__":
    main()
