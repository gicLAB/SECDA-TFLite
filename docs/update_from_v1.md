# Manual Update from V1 to V2

This is a guide for transitioning from using the secda_tflite local API to using SECDA TOOLs (SECDA-CORE).

## Overview

SECDA-CORE provides a more robust and flexible way to update and maintain the core SECDA-APIs. This guide will help you understand the changes and how to migrate your existing delegates from the old secda_tflite API to the new SECDA-CORE API.

Additionally, this guide will help you with how to update your local repository to v2 with Git.


## Updating Your Local Repository

1. First off, all I would recommend you save your current work in your own branch and keep your current local repository untouched. 
  
2. Then I would recommend you clone SECDA-TFLite repo in completely new folder in your system and do all the initalising setup using the main readme
   
3. Then create a new branch from the latest commit in the `main` branch. Name the branch something like `v2_username`.

4. Work on this new branch, copy and over all your custom delegates from your old repository to the new one.

5. Make sure to follow the steps below to update your delegates to the new SECDA-CORE API. Please follow the delegate naming conventions and directory structure as shown in Omni v1 delegate example.

6. You should be able to push your changes to the new branch and create a pull request to the main repository whenever you want to merge your delegates to the main branch.



##  Changes to Bazel "BUILD" files

Within each delegate directory, you have three BUILD files that need to be updated.

### Delegate BUILD file
This is the build file for the delegate itself. For example, it should be in `src/secda_delegates/<delegate_name>/version/BUILD`.

Old file contains the following config settings and cc_library:
```python
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
    name = "vm_delegate",
    srcs = [
        "vm_delegate.cc",
    ],
    hdrs = [
        "vm_delegate.h",
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:types"
    ] + select({
        ":linux_armhf": ["//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v3/accelerator/driver:driver"],
        "//conditions:default": ["//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v3/accelerator/driver:driver_sysc"],
    }),
)
```

New file should look like this:
```python
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
    name = "vm_delegate",
    srcs = [
        "vm_delegate.cc",
    ],
    hdrs = [
        "vm_delegate.h",
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
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:optimized_4bit",
        "//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v3/accelerator/driver:vm_driver",
    ]
)
```
The main change is that we have removed the `select` function and replaced it with a direct dependency on the new driver target `vm_driver`. This simplifies the build process and ensures that the correct driver is always used. The select is done in the accelerator BUILD file instead, which allows for more flexibility in how the driver is built based on the target architecture.


### Accelerator BUILD file
This file is responsible for building the accelerator source code. It should be located in `src/secda_delegates/<delegate_name>/version/accelerator/BUILD`.
Old file example:
```python

cc_library(
    name = "accelerator_config",
    srcs = [
        "acc_config.sc.h",
        ],
    copts = common_copts,
    deps = select({
        ":linux_armhf": [],
        "//conditions:default": ["@systemc//:systemc"],
    }),
)


cc_library(
    name = "accelerator",
    srcs = [
        "acc.sc.cc",
        ],
    hdrs = [
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
```

New file example:
```python
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
            "counter.sc.h",
            "data_in.sc.h",
            "in.sc.h",
            "out.sc.h",
            "scheduler.sc.h",
            "vm_gemm.sc.h",
            "vmm_control.sc.h",
            "vmm_modules.sc.h",
            "vmm_unit.sc.h",
            "write_sync.sc.h",
        ],
        "//conditions:default": [],
    }),
    copts = common_copts,
    deps = ["//third_party:secdav5"],
)

```
The main thing we do is combine the two cc_library targets into one, and we use the `select` function to conditionally include the source files based on simulation or not.


### Driver BUILD file
This file is responsible for building the driver source code. It should be located in `src/secda_delegates/<delegate_name>/version/accelerator/driver/BUILD`.
Old file example:
```python

cc_library(
    name = "driver_sysc",
    srcs = [
        "acc_container.h",
        "gemm_driver.h",
        "systemc_binding.h"
        ],
    copts = common_copts,
    deps = [
        "@systemc//:systemc",
        "//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v3/accelerator:accelerator",
        "//tensorflow/lite/delegates/utils/secda_tflite:secda_tflite_sim",
    ],
)


cc_library(
    name = "driver",
    srcs = [
        "acc_container.h",
        "gemm_driver.h",
        ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v3/accelerator:accelerator_config",
        "//tensorflow/lite/delegates/utils/secda_tflite:secda_tflite",
    ],
)
```

New file example:
```python

cc_library(
    name = "vm_driver",
    srcs = [
        "systemc_binding.h",
        "acc_container.h",
        "gemm_driver.h",
        "gemm_mt.h",
    ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/delegates/utils/secda_delegates/vm_delegate/v5/accelerator:accelerator",
    ],
)
```
The main change is that we have removed the `driver_sysc` target and replaced it with a single `vm_driver` target that directly depends on the `accelerator` target. This simplifies the build process and ensures that the correct driver is always used. The `vm_driver` now includes all necessary headers and dependencies for the VM delegate.


NOTE: Making these changes means that when you run ```bazel build```, you will need to add the `--@secda_tools//:config=sysc` or `--@secda_tools//:config=fpga` flag to choose the appropriate configuration.


## Changes to the Delegate Source Code

You will need to update all ```include``` statements in your delegate, drive and accelerator source files to reflect the change to the new SECDA-CORE API. For example, if you previously had:

```cpp
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
```
You should change it to:

```cpp
#include "secda_tools/secda_utils/multi_threading.h"
```

I would highly recommend using the [OMNI v1 delegate](../src/secda_delegates/omni_delegate/v1/) or [VMv5 delegate](../src/secda_delegates/vm_delegate/v5/) includes as a reference for how to structure your includes. 

NOTE: some old secda_tflite includes might be moved to different locations in the new SECDA-CORE API, so you may need to adjust your includes accordingly. Check the [SECDA-CORE GitHub repository](https://github.com/judeharis/secda_tools/tree/main/secda_tools) for the latest structure and available headers.



