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
            "data_in.sc.h",
            "mapper.sc.h",
            "input_loader.sc.h",
            "in.sc.h",
            "out.sc.h",
            "pe_control.sc.h",
            "pe_module.sc.h",
            "pe_compute.sc.h",
            "pe_output.sc.h",
            "scheduler.sc.h",
            "counter.sc.h",
        ],
        "//conditions:default": [],
    }),
    copts = common_copts,
    deps = ["//third_party:secdav5"],
)
