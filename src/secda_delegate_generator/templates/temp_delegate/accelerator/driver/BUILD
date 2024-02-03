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
    name = "driver_name_sysc",
    srcs = [
        "systemc_binding.h",
        "acc_container.h",
        "driver_name.h",
        ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_delegates/temp_delegate/accelerator:temp_accelerator",
        "//secda_tflite_path:secda_tflite_sim",
    ],
)

cc_library(
    name = "driver_name",
    srcs = [
        "acc_container.h",
        "driver_name.h",
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/secda_delegates/temp_delegate/accelerator:temp_accelerator_config",
        "//secda_tflite_path:secda_tflite",
    ],
)

