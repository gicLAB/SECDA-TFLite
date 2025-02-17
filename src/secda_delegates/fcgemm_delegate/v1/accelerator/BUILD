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
    name = "fcgemm_accelerator_config",
    srcs = [
        "acc_config.sc.h",
        ],
    copts = common_copts,
    deps = select({
        ":linux_armhf": [],
        "//conditions:default": [],
    }) + [
        "//third_party:secdav5"
    ],
)

cc_library(
    name = "fcgemm_accelerator",
    srcs = [
        "acc.sc.cc",
        ],
    hdrs = [
        "acc.sc.h",
        "fetch.sc.h",
        "load.sc.h",
        "compute.sc.h",
        "scheduler.sc.h",
        "store.sc.h",
        "counter.sc.h"
    ],
    copts = common_copts,
    deps = [
        ":fcgemm_accelerator_config",
    ],
)

