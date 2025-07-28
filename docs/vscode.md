# VSCode Documentation for SECDA-TFLite

We highly recommend taking full advantage of the VSCode development environment to streamline your development process with the SECDA-TFLite toolkit.

## Task and Launch Configurations
The VSCode workspace includes pre-configured tasks and launch configurations to help you compile and run simulations efficiently. These configurations are located in the following files:
- `/tensorflow/.vscode/launch.json` - This file contains configurations for launching the simulation.
- `/tensorflow/.vscode/task.json` - This file contains configurations for compiling the code.
When you create a new delegate and add it to [secda_delegates](../src/secda_delegates/), you have to add new tasks and launch configurations to compile and run the new delegate.

### Example Task Configuration
Here is are example task configurations for an "example" delegate, you should atleast add two new tasks per delegate, one for the benchmarking which will use the "benchmark_model" program and one for verification which will use the "inference_diff" program.
Note when you generate a new delegate using [secda_generator](../src/secda_generator/), it will automatically generate the bazel files that contain the "benchmark_model" and "inference_diff" programs attached to your delegate, all you need to do is add the task configurations below to your `/tensorflow/.vscode/task.json` file.

#### Benchmark Model Task Configuration
```json
{
  "label": "benchmark_model_plus_example_delegate_v1",
  "type": "shell",
  "command": "bazel6 build tensorflow/lite/delegates/utils/secda_delegates/example_delegate/v1:benchmark_model_plus_example_delegate -c dbg --cxxopt='-DSYSC' --cxxopt='-DTF_LITE_DISABLE_X86_NEON' --cxxopt='-DACC_PROFILE' --define tflite_with_xnnpack=false --cxxopt='-DRUY_OPT_SET=0' --@secda_tools//:config=sysc ",
  "group": {
    "kind": "build",
    "isDefault": true
  },
}
```

#### Example Inference Diff Task Configuration
```json
{
  "label": "inference_diff_plus_example_delegate_v1",
  "type": "shell",
  "command": "bazel6 build tensorflow/lite/delegates/utils/secda_delegates/example_delegate/v1:inference_diff_plus_example_delegate -c dbg --cxxopt='-DSYSC' --cxxopt='-DTF_LITE_DISABLE_X86_NEON' --cxxopt='-DACC_PROFILE' --define tflite_with_xnnpack=false --cxxopt='-DRUY_OPT_SET=0' --@secda_tools//:config=sysc ",
  "group": {
    "kind": "build",
    "isDefault": true
  },
}
```

### Launch Configuration Example
Here is an example launch configuration for the "example" delegate, you should atleast add two new
launch configurations per delegate, one for the benchmarking which will use the "benchmark_model" program and one for verification which will use the "inference_diff" program.

#### Benchmark Model Launch Configuration
```json
{
  "name": "Benchmark Model | Examplev1",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/bazel-bin/tensorflow/lite/delegates/utils/secda_delegates/example_delegate/v1/benchmark_model_plus_example_delegate",
  "args": [
    "--use_gpu=false",
    "--num_threads=1",
    "--enable_op_profiling=true",
    "--graph=${workspaceFolder}/../data/models/model.tflite",
    "--num_runs=1",
    "--warmup_runs=0",
    "--use_example_delegate=true",
  ],
  "stopAtEntry": false,
  "cwd": "${workspaceFolder}",
  "environment": [],
  "externalConsole": false,
  "MIMode": "gdb",
  "setupCommands": [
    {
      "description": "Enable pretty-printing for gdb",
      "text": "-enable-pretty-printing",
      "ignoreFailures": true
    }
  ]
}
```

#### Inference Diff Launch Configuration
```json
{
  "name": "Inference Diff | Examplev1",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/bazel-bin/tensorflow/lite/delegates/utils/secda_delegates/example_delegate/v1/inference_diff_plus_example_delegate",
  "args": [
    "--model_file=${workspaceFolder}/../data/models/model.tflite",
    "--num_runs=1",
    "--use_vm_deuse_example_delegatelegate=true",
  ],
  "stopAtEntry": false,
  "cwd": "${workspaceFolder}",
  "environment": [],
  "externalConsole": false,
  "MIMode": "gdb",
  "setupCommands": [
    {
      "description": "Enable pretty-printing for gdb",
      "text": "-enable-pretty-printing",
      "ignoreFailures": true
    }
  ]
}
```