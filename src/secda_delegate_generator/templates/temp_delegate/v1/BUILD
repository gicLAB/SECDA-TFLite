load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)


config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "temp_delegate",
    srcs = [
        "temp_delegate.cc",
    ],
    hdrs = [
        "temp_delegate.h",
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/delegates/utils/secda_delegates/temp_delegate/v1/accelerator/driver:driver_name"
    ]
)

cc_binary(
    name = "temp_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":temp_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

#### The following are for using the temp test delegate in TFLite tooling ####
cc_library(
    name = "temp_delegate_provider",
    srcs = ["temp_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":temp_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)



cc_binary(
    name = "eval_model_plus_temp_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//secda_apps_path/eval_model:eval_model_lib",
        ":temp_delegate_provider", 
    ],
)


cc_binary(
    name = "label_image_plus_temp_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/examples/label_image:label_image_lib",
        ":temp_delegate_provider",
    ],
)

cc_binary(
    name = "benchmark_model_plus_temp_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
        ":temp_delegate_provider",
    ],
)

cc_binary(
    name = "inference_diff_plus_temp_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
        ":temp_delegate_provider",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_temp_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
        ":temp_delegate_provider",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_temp_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
        ":temp_delegate_provider",
    ],
)