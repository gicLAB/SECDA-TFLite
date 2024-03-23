import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
logging.getLogger("tensorflow").disabled = True
import tensorflow as tf
import numpy as np

def build_tconv_model(params, mdir):
    '''params = [stride_x, stride_y, filters, kernel_size, in1, in2, in3, padding_val]'''

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
    file_s = f"tconv_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_{params[5]}_{params[6]}.tflite"
    with open(mdir + file_s, "wb") as f:
        f.write(tflite_model)
    os.system(f"rm -rf {mdir}tf")
    return file_s

def generate_python_list(params, filename):
    # create a json with list of all the models to be used in the benchmarking script
    f = open(filename, "w+")
    f.write("{\n")
    f.write("{}\n".format('"tconv_models" : ['))
    for param in params:
        c = "" if param == params[-1] else ","
        f.write(
            '    "tconv_{}_{}_{}_{}_{}_{}_{}"{}\n'.format(
                param[0], param[1], param[2], param[3], param[4], param[5], param[6], c
            )
        )
    f.write("]}\n")
    f.close()


params = []
mdir = "models/tconv/"
fs = [16,32,64]
ks = [3, 5, 7]
inh = [7, 9, 11]
ic = [32, 64, 128, 256]
strides = [1, 2]

for s in strides:
    for f in fs:
        for k in ks:
            for i in inh:
                for o in ic:
                    params.append([s, s, f, k, i, i, o, "same"])

# params = []
# dcgan_layer1 = [2,2,512,5,4,4,1024,"same"]
# dcgan_layer2 = [2,2,256,5,8,8,512,"same"]
# dcgan_layer3 = [2,2,128,5,16,16,256,"same"]
# dcgan_layer4 = [2,2,3,5,32,32,128,"same"]
# params.append(dcgan_layer1)
# params.append(dcgan_layer2)
# params.append(dcgan_layer3)
# params.append(dcgan_layer4)
                    
#  [s, s, f, k, i, i, o, "same"]
params = []
fcn_layer1 = [2,2,21,4,1,1,21,"same"]
fcn_layer2 = [2,2,21,4,4,4,21,"same"]
# fcn_layer3 = [2,2,128,5,16,16,256,"same"]

params.append(fcn_layer1)
params.append(fcn_layer2)
# params.append(dcgan_layer3)
# params.append(dcgan_layer4)


for param in params:
    build_tconv_model(param, mdir)

generate_python_list(params, "configs/tconv_models.json")
