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

# cc_library(
#     name = "omni_accelerator_config",
#     srcs = [
#         "acc_config.sc.h",
#         ],
#     copts = common_copts,
#     deps = select({
#         ":linux_armhf": [],
#         "//conditions:default": ["@systemc//:systemc"],
#     }),
# )

# cc_library(
#     name = "omni_accelerator",
#     srcs = [
#         "acc.sc.cc",
#     ],
#     hdrs = [
#         "acc.sc.h",
#         "compute.sc.h",
#         "omni_pe.sc.h"
#     ],
#     copts = common_copts,
#     deps = [
#         ":omni_accelerator_config",
#         "//tensorflow/lite/delegates/utils/secda_tflite:secda_tflite_sim",
#     ],
# )


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
    deps = ["//third_party:secdav5"],
)