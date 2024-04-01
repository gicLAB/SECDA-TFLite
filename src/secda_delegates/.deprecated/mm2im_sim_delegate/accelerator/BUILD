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
    name = "accelerator",
    srcs = [
        "acc.sc.cc",
        ],
    hdrs = [
        "acc_config.sc.h",
        "acc.sc.h",
        "data_in.sc.h",
        "in.sc.h",
        "out.sc.h",
        "pe_control.sc.h",
        "pe_module.sc.h",
        "scheduler.sc.h",
        "counter.sc.h",
    ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_tflite:secda_tflite_sim",
    ],
)

