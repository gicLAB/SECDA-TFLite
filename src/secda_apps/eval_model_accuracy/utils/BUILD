# Description:
# SECDA TFLite Eval Model

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bmp",
]))



cc_library(
    name = "utils",
    srcs = ["bitmap_helpers.cc",
            "npy.hpp",],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "log.h",
        "get_top_n.h",
        "get_top_n_impl.h",
    ],
    deps = [
        "//tensorflow/lite/examples/secda_apps/eval_model_accuracy:eval_model_lib_hdr",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//tensorflow:android_arm64": [
            "//tensorflow/lite/delegates/gpu:delegate",
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)
