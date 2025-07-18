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
    name = "tempdel_delegate",
    srcs = [
        "tempdel_delegate.cc",
    ],
    hdrs = [
        "tempdel_delegate.h",
        "util.h",
        "util_prep.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:optimized_4bit",
        "//secda_delegates_path/tempdel_delegate/v1/accelerator/driver:driver_name"
    ]
)

cc_binary(
    name = "tempdel_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":tempdel_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

#### The following are for using the tempdel test delegate in TFLite tooling ####
cc_library(
    name = "tempdel_delegate_provider",
    srcs = ["tempdel_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":tempdel_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)



cc_binary(
    name = "eval_model_plus_tempdel_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//secda_apps_path/eval_model:eval_model_lib",
        ":tempdel_delegate_provider",
    ],
)


cc_binary(
    name = "label_image_plus_tempdel_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/examples/label_image:label_image_lib",
        ":tempdel_delegate_provider",
    ],
)

cc_binary(
    name = "benchmark_model_plus_tempdel_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
        ":tempdel_delegate_provider",
    ],
)

cc_binary(
    name = "inference_diff_plus_tempdel_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
        ":tempdel_delegate_provider",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_tempdel_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
        ":tempdel_delegate_provider",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_tempdel_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
        ":tempdel_delegate_provider",
    ],
)