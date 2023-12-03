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
    name = "eval_model_lib_hdr",
    srcs = [
        "eval_model.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,-rpath=/data/local/tmp",  # for hexagon delegate
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:tflite_execution_providers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)


cc_library(
    name = "eval_model_lib",
    srcs = [
        "eval_model.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,-rpath=/data/local/tmp",  # for hexagon delegate
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/examples/secda_apps/eval_model/utils:utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:tflite_execution_providers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "eval_model",
    srcs = [
        "eval_model.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,-rpath=/data/local/tmp",  # for hexagon delegate
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/examples/secda_apps/eval_model/utils:utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:tflite_execution_providers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)


cc_test(
    name = "eval_model_test",
    srcs = [
        "eval_model_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        "//tensorflow/lite/examples/secda_apps/eval_model/utils:utils",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)
