load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = ["-Wall"] + tflite_copts()


cc_library(
    name = "driver_name",
    srcs = [
        "systemc_binding.h",
        "acc_container.h",
        "driver_name.h",
    ],
    copts = common_copts,
    deps = [
        "//secda_delegates_path/tempdel_delegate/v1/accelerator:accelerator",
    ],
)
