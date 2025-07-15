import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging

logging.getLogger("tensorflow").disabled = True
import tensorflow as tf
import numpy as np

# rows = filters
# cols = out1 * out2
# depth = kernel_size * kernel_size * in3


def build_conv_model(params, mdir):
    """params = [stride_x, stride_y, filters, kernel_size, in1, in2, in3, padding_val]"""

    stride_x = params[0]
    stride_y = params[1]
    filters = params[2]
    kernel_size = params[3]
    in1 = params[4]
    in2 = params[5]
    in3 = params[6]
    padding_val = params[7]
    out3 = filters

    if padding_val == "same":
        out1 = in1 // stride_x + (1 if stride_x > 1 else 0)
        out2 = in2 // stride_y + (1 if stride_y > 1 else 0)
    else:
        out1 = in1 + 1 - kernel_size
        out2 = in2 + 1 - kernel_size

    # if padding_val == "same":
    #     out1 = in1 * stride_x
    #     out2 = in2 * stride_y
    # else:
    #     out1 = ((in1 - 1) * stride_x) + kernel_size
    #     out2 = ((in2 - 1) * stride_y) + kernel_size

    print(f"Input: {in1}x{in2}x{in3}")
    print(f"weights: {in3}x{kernel_size}x{kernel_size}x{filters}")
    print(f"Output: {out1}x{out2}x{out3}")
    print(f"Stride: {stride_x}x{stride_y}")

    # matrix size after Im2Col
    # rows = filters
    # cols = out1 * out2
    # depth = kernel_size * kernel_size * in3
    print(f"Im2Col: {filters}x{out1*out2}x{kernel_size*kernel_size*in3}")

    inputs = tf.keras.Input((in1, in2, in3))
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(stride_x, stride_y),
        padding=padding_val,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        use_bias=True,
    )(inputs)
    conv = tf.keras.Model(inputs, x)
    conv.compile(optimizer="sgd", loss="categorical_crossentropy")

    # conv.fit(
    #     np.random.rand(1, in1, in2, in3),
    #     np.random.rand(1, out1, out2, out3),
    #     batch_size=1,
    #     epochs=2,
    # )

    os.makedirs(mdir + "tf", exist_ok=True)
    # conv.save(mdir + "tf")
    conv.save(mdir + "tf/model.h5")
    # sys.exit()
    lamyield = lambda: [
        [np.random.rand(1, in1, in2, in3).astype(np.float32)] for x in range(15)
    ]
    # Load the model
    model = tf.keras.models.load_model(mdir + "tf/model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # converter = tf.lite.TFLiteConverter.from_saved_model(mdir + "tf") ##/model.h5
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lamyield
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    # file_s = f"conv_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_{params[5]}_{params[6]}.tflite"
    file_s = f"mnk_{filters}_{out1*out2}_{kernel_size*kernel_size*in3}.tflite"
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
        stride_x = param[0]
        stride_y = param[1]
        filters = param[2]
        kernel_size = param[3]
        in1 = param[4]
        in2 = param[5]
        in3 = param[6]
        padding_val = param[7]
        out3 = filters
        if padding_val == "same":
            out1 = in1 // stride_x + (1 if stride_x > 1 else 0)
            out2 = in2 // stride_y + (1 if stride_y > 1 else 0)
        else:
            out1 = in1 + 1 - kernel_size
            out2 = in2 + 1 - kernel_size

        c = "" if param == params[-1] else ","
        f.write(
            '    "mnk_{}_{}_{}"{}\n'.format(
                str(filters), str(out1 * out2), str(kernel_size * kernel_size * in3), c
            )
        )
    f.write("]}\n")
    f.close()


# params = []
# fs = [16, 32, 64]
# ks = [3, 5, 7]
# inh = [7, 9, 11]
# ic = [16, 32, 64, 128]
# strides = [1]

params = []
fs = [32, 64, 128, 256, 512, 1024, 2048]
ks = [2]
inh = [8, 16, 32, 64]
ic = [64, 128, 256, 512, 1024]
strides = [1]


# open file


for s in strides:
    for f in fs:
        for k in ks:
            for i in inh:
                for o in ic:
                    params.append([s, s, f, k, i, i, o, "same"])

print(f"Generating {len(params)} models")
# print("params = [stride_x, stride_y, filters, kernel_size, in1, in2, in3, padding_val]", end="\n\n")

## print params in a text file
with open("conv_params.txt", "w") as f:
    f.write(
        "params = [stride_x, stride_y, filters, kernel_size, in1, in2, in3, padding_val]\n"
    )
    for param in params:
        f.write(str(param) + "\n")


# sys.exit()


mdir = "models/mnk/"
name = "mnk_large"

for param in params:
    build_conv_model(param, mdir)
    # sys.exit()
generate_python_list(params, f"configs/{name}.json", name)


print(f"Generated {len(params)} models")
print(f"Saved to ./configs/{name}.json")
