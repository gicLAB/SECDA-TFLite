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

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mm2im_delegate",
    srcs = [
        "mm2im_delegate.cc",
    ],
    hdrs = [
        "mm2im_delegate.h",
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:optimized_base"
    ] + select({
        ":linux_armhf": ["//tensorflow/lite/delegates/utils/secda_delegates/mm2im_delegate/v8/accelerator/driver:driver"],
        ":linux_aarch64": ["//tensorflow/lite/delegates/utils/secda_delegates/mm2im_delegate/v8/accelerator/driver:driver"],
        "//conditions:default": ["//tensorflow/lite/delegates/utils/secda_delegates/mm2im_delegate/v8/accelerator/driver:driver_sysc"],
    }),
)

cc_binary(
    name = "mm2im_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":mm2im_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

#### The following are for using the mm2im test delegate in TFLite tooling ####
cc_library(
    name = "mm2im_delegate_provider",
    srcs = ["mm2im_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":mm2im_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)



cc_binary(
    name = "eval_model_plus_mm2im_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/examples/secda_apps/eval_model:eval_model_lib",
        ":mm2im_delegate_provider", 
    ],
)

cc_binary(
    name = "label_image_plus_mm2im_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/examples/label_image:label_image_lib",
        ":mm2im_delegate_provider", 
    ],
)

cc_binary(
    name = "benchmark_model_plus_mm2im_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
        ":mm2im_delegate_provider",
    ],
)

cc_binary(
    name = "inference_diff_plus_mm2im_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
        ":mm2im_delegate_provider",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_mm2im_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
        ":mm2im_delegate_provider",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_mm2im_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
        ":mm2im_delegate_provider",
    ],
)
