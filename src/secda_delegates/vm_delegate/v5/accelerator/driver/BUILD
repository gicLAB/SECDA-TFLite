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
    name = "driver_sysc",
    srcs = [
        "acc_container.h",
        "gemm_driver.h",
        "gemm_mt.h",
        "systemc_binding.h"
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v5/accelerator:accelerator",
    ],
)


cc_library(
    name = "driver",
    srcs = [
        "acc_container.h",
        "gemm_driver.h",
        "gemm_mt.h",
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v5/accelerator:accelerator_config"
    ],
)


