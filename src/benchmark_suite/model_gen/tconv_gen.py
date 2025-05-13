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
    os.makedirs(mdir + "tf", exist_ok=True)
    tconv.save(mdir + "tf")

    lamyield = lambda: [
        [np.random.rand(1, in1, in2, in3).astype(np.float32)] for x in range(15)
    ]
    converter = tf.lite.TFLiteConverter.from_saved_model(mdir + "tf")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lamyield
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    # file_s = f"tconv_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_{params[5]}_{params[6]}.tflite"
    file_s = f"{name}.tflite"
    with open(mdir + file_s, "wb") as f:
        f.write(tflite_model)
    os.system(f"rm -rf {mdir}tf")
    return file_s

def generate_python_list(params, filename, name):
    # create a json with list of all the models to be used in the benchmarking script
    f = open(filename, "w+")
    f.write("{\n")
    f.write("{}\n".format(f'"{name}" : ['))
    for param in params:
        c = "" if param == params[-1] else ","
        f.write(
            '    "tconv_{}_{}_{}_{}_{}_{}_{}"{}\n'.format(
                param[0], param[1], param[2], param[3], param[4], param[5], param[6], c
            )
        )
    f.write("]}\n")
    f.close()


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

params = []
fs = [16, 32, 64]
ks = [3, 5, 7]
inh = [7, 9, 11]
ic = [32, 64, 128, 256]
strides = [1, 2]

for s in strides:
    for f in fs:
        for k in ks:
            for i in inh:
                for o in ic:
                    params.append([s, s, f, k, i, i, o, "same", f"tconv_{s}_{s}_{f}_{k}_{i}_{i}_{o}"])

# params = [sx, sy, oc, k, ih, iw, ic, "same"]
# dcgan_layer1 = [2, 2, 512, 5, 4, 4, 1024, "same"]
# dcgan_layer2 = [2, 2, 256, 5, 8, 8, 512, "same"]
# dcgan_layer3 = [2, 2, 128, 5, 16, 16, 256, "same"]
# dcgan_layer4 = [2, 2, 3, 5, 32, 32, 128, "same"]
# dcgan = [dcgan_layer1, dcgan_layer2, dcgan_layer3, dcgan_layer4]

# tf_dcgan_layer1 = [1, 1, 128, 5, 7, 7, 256, "same"]
# tf_dcgan_layer2 = [2, 2, 64, 5, 7, 7, 128, "same"]
# tf_dcgan_layer3 = [2, 2, 1, 5, 14, 14, 64, "same"]
# tf_dcgan = [tf_dcgan_layer1, tf_dcgan_layer2, tf_dcgan_layer3]

# fcn_layer1 = [2, 2, 21, 4, 1, 1, 21, "same"]
# fcn_layer2 = [2, 2, 21, 4, 4, 4, 21, "same"]
# fcn_layer3 = [2, 2, 128, 5, 16, 16, 256, "same"]
# fcn = [fcn_layer1, fcn_layer2, fcn_layer3]

# test_layer = [2,2,3,5,4,4,64,"same"]
# test_layer = [2, 2, 1, 8, 14, 14, 64, "same"]
# test_layer = [2, 2, 8, 4, 14, 14, 64, "same"]
# test_layer = [2, 2, 7, 4, 14, 14, 64, "same"]




# pix2pix_layer1 = [2, 2, 512, 4, 1, 1, 512, "same"]
# pix2pix_layer2 = [2, 2, 512, 4, 2, 2, 1024, "same"]
# pix2pix_layer3 = [2, 2, 512, 4, 4, 4, 1024, "same"]
# pix2pix_layer4 = [2, 2, 512, 4, 8, 8, 1024, "same"]
# pix2pix_layer5 = [2, 2, 256, 4, 16, 16, 1024, "same"]
# pix2pix_layer6 = [2, 2, 128, 4, 32, 32, 512, "same"]
# pix2pix_layer7 = [2, 2, 64, 4, 64, 64, 256, "same"]
# pix2pix_layer8 = [2, 2, 3, 4, 128, 128, 128, "same"]
# pix2pix = [
#     pix2pix_layer1,
#     pix2pix_layer2,
#     pix2pix_layer3,
#     pix2pix_layer4,
#     pix2pix_layer5,
#     pix2pix_layer6,
#     pix2pix_layer7,
#     pix2pix_layer8,
# ]

#  [s, s, oc, ks, iw, ih, ic, "same"]
# params = tf_dcgan
# params = pix2pix
# params = fcn
# params = dcgan
# params = [test_layer]


mdir = "models/tconv_gen/"
name = "tconv_gen"

# for param in params:
#     build_tconv_model(param, mdir)
# generate_python_list(params, f"configs/{name}.json", name)


generate_json_file(params, f"configs/{name}.json", name, "TRANSPOSE_CONV")


print(f"Generated {len(params)} models")
print(f"Saved to ./configs/{name}.json")
