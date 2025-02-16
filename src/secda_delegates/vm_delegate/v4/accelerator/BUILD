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

# config_setting(
#     name = "linux_aarch64",
#     values = {"cpu": "aarch64"},
#     visibility = ["//visibility:public"],
# )

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "accelerator_config",
    srcs = [
        "acc_config.sc.h",
    ],
    copts = common_copts,
    deps = select({
        ":linux_armhf": [],
        # ":linux_aarch64": [],
        "//conditions:default": ["@systemc"],
    }),
)

cc_library(
    name = "accelerator",
    srcs = [
        "acc.sc.cc",
    ],
    hdrs = [
        # "acc_config.sc.h",
        "acc.sc.h",
        "in.sc.h",
        "data_in.sc.h",
        "scheduler.sc.h",
        "vm_gemm.sc.h",
        "vmm_unit.sc.h",
        "vmm_control.sc.h",
        "vmm_modules.sc.h",
        "write_sync.sc.h",
        "out.sc.h",
        "counter.sc.h",
    ],
    copts = common_copts,
    deps = [
        ":accelerator_config",
        "//tensorflow/lite/delegates/utils/secda_tflite:secda_tflite_sim",
    ],
)
