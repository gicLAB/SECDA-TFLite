load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
    visibility = ["//visibility:public"],
)

common_copts = ["-Wall"] + tflite_copts()


cc_library(
    name = "accelerator",
    srcs = [
        "acc_config.sc.h",
    ] + select({
        "@secda_tools//:sysc": ["acc.sc.cc"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@secda_tools//:sysc": [
            "acc.sc.h",
            "compute.sc.h",
            "omni_pe.sc.h"
        ],
        "//conditions:default": [],
    }),
    copts = common_copts,
    # COPILOT NOTE: Added comment. v1_tensor_manager moves accelerator builds
    # to secdav6 APIs so the delegate-side tensor-manager DMA path can use the
    # newer AXI support interfaces consistently.
    deps = ["//third_party:secdav6"],
)