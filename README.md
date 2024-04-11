# SECDA-TFLite: A Toolkit for Efficient Development of FPGA-based DNN Accelerators for Edge Inference

The SECDA-TFLite toolkit leverages the TFLite delegate system to provide a robust and extensible set of utilities for integrating DNN accelerators for any DNN operation supported by TFLite.
Ultimately, this increases hardware accelerator developers' productivity, as they can begin developing and refining their design more quickly.


# Paper
Our research paper covers the the SECDA-TFLite toolkit in detail including case studies where we design and integrate new DNN accelerator using our toolkit. If you are using the SECDA-TFLite toolkit in research, we kindly request a reference to the following:

```
@article{Haris2023JPDC,
    title = {SECDA-TFLite: A toolkit for efficient development of FPGA-based DNN accelerators for edge inference},
    author = {Jude Haris and Perry Gibson and José Cano and Nicolas {Bohm Agostini} and David Kaeli},
    journal = {Journal of Parallel and Distributed Computing},
    volume = {173},
    pages = {140-151},
    year = {2023},
    issn = {0743-7315},
    doi = {https://doi.org/10.1016/j.jpdc.2022.11.005},
    url = {https://www.sciencedirect.com/science/article/pii/S0743731522002301}
}
```


# Accelerator Designs
* We provide pre-compiled binaries/bitstream for the PYNQ Z1 along with archived Vivado and Vivado HLS project folders ([release](https://github.com/gicLAB/SECDA-TFLite/releases/tag/v1.0)) to enable synthesis from scratch.
* We also provide source code for all accelerators
* For more information please check out the [accelerator source-code](secda_tflite_accelerator)

## Synthesis for PYNQ-Z1
To perform logic synthesis, we provide Vivado project folders ([release](https://github.com/gicLAB/SECDA-TFLite/releases/tag/v1.0)). These contain the necessary block diagram configuration including AXI DMA's and the accelerators to ensure correct connectivity to the processing system.

**Requirements**
* Vivado HLS 2018.3
* Vivado 2018.3

**Instructions**
To perform create Vivado IP from SystemC source code, do the following:
* Unzip [HLS](https://github.com/gicLAB/SECDA-TFLite/releases/download/v1.0/HLS_projects.zip) accelerator project you want to use.
* Load up Vivado HLS and choose the project folder within the unzipped folder to open the pre-configured Vivado project and solution.
* Here, we can ask Vivado HLS to perform HLS and export RTL using the menu bar at the top.
* For logic synthesis, simply open up Vivado and loaded up [HLx](https://github.com/gicLAB/SECDA-TFLite/releases/download/v1.0/Vivado_projects.zip) project's .xpr file within the tool.
* Use the "Generate Bitstream" option to synthesize and export the FPGA mapping.
* The bitmap and confirguration files will be saved in the HLx project folder: ```project_name.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh``` and ```project_name.runs/impl_1/design_1_wrapper.bit```


# Docker Setup for SECDA-TFLite (TF2.7)
We highly recommend using this container to immediately start developing accelerators and accelerator delegates using the SECDA-TFLite toolkit.

**Requirements**
* Docker
* VSCode (soft requirement)

**Instructions**
* First pull the docker image: 
```
docker pull judeharis97/secda-tflite-toolkit:v1
```
* Simply create a container of the downloaded image using the following command: 
```
docker run -it -d --name secda-tflite judeharis97/secda-tflite-toolkit:v1
```

* Once the container is created and launched, you can access it through [VSCode's attach to container functionality](https://code.visualstudio.com/docs/remote/attach-container)
* Load VSCode workspace at `/root/Workspace/tensorflow/tensorflow.code-workspace` using "open workspace from file" option in the VSCode File menu.
* Once the VSCode workspace is loaded, you are able to run to the launch configurations through the [Run and Debug](https://code.visualstudio.com/docs/editor/debugging) tab to run the end to end simulation.
* These configurations are stored within '/root/Workspace/tensorflow/.vscode/launch.json' (to launch) and /root/Workspace/tensorflow/.vscode/task.json (to compile), you can edit these to change the parameters to compile and launch the the end to end simulation.
* There is three configuration already prepared to run the VM,SA and FC-GEMM accelerator with the simulation delegates

# PYNQ-Z1 TFLite Inference with Accelerators
In order to run the FPGA mapped accelerators we need to cross-compiler our TFLite program for our PYNQ-Z1 board with bundled with the delegate for your target accelerator.
For our TFLite program, we have adapted an example "benchmark_model" provided by Tensorflow to enable our accelerator pipeline.



## Downlaod pynq_scr_folder from release [Method 1]
Download [pynq_src folder](https://github.com/gicLAB/SECDA-TFLite/releases/download/v1.0/pynq_src.zip)
Copy this into the pynq device. This folder contains all the bitmaps, models and binary executable required to run the accelerators. 
___

## Cross Compile [Method 2]
Below are the instructions to cross-compile "benchmark_model" for the target board.**Note**: this assumes you have complete [Docker Setup for SECDA-TFLite (TF2.7)](https://github.com/gicLAB/SECDA-TFLite?tab=readme-ov-file#docker-setup-for-secda-tflite-tf27) and you have are running the following commands from the docker containers's Tensorflow root ```/root/Workspace/tensorflow```

**Make an output arm_bin folder:**
```shell 
  mkdir -p /root/Workspace/tensorflow/arm_bin
```

**To cross-compiler the BERT accelerator:**
``` shell
$ bazel build --config=elinux_armhf -c opt //tensorflow/lite/delegates/utils/secda_bert_delegate:benchmark_model_plus_secda_bert_delegate --cxxopt='-mfpu=neon' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' 
$ cp /root/Workspace/tensorflow/bazel-out/armhf-opt/bin/tensorflow/lite/delegates/utils/secda_bert_delegate/benchmark_model_plus_secda_bert_delegate /root/Workspace/tensorflow/arm_bin/benchmark_model_secda_bert
```

**To cross-compiler the VM accelerator:**
``` shell
$ bazel build --config=elinux_armhf -c opt //tensorflow/lite/delegates/utils/secda_vm_delegate:benchmark_model_plus_secda_vm_delegate --cxxopt='-mfpu=neon' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' 
$ cp /root/Workspace/tensorflow/bazel-out/armhf-opt/bin/tensorflow/lite/delegates/utils/secda_vm_delegate/benchmark_model_plus_secda_vm_delegate /root/Workspace/tensorflow/arm_bin/benchmark_model_secda_vm
```

**To cross-compiler the SA accelerator:**
``` shell
$ bazel build --config=elinux_armhf -c opt //tensorflow/lite/delegates/utils/secda_sa_delegate:benchmark_model_plus_secda_sa_delegate --cxxopt='-mfpu=neon' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' 
$ cp /root/Workspace/tensorflow/bazel-out/armhf-opt/bin/tensorflow/lite/delegates/utils/secda_sa_delegate/benchmark_model_plus_secda_sa_delegate /root/Workspace/tensorflow/arm_bin/benchmark_model_secda_sa
```

Then copy the executables within ```arm_bin```, the models from ```models```, the accelerator ```.bitmaps``` and ```.hwh``` files to the board.
``` shell
$ cp ./arm_bin/benchmark_model_secda_[vm|sa|bert] TO_PYNQ_DEVICE
$ cp ./models/* TO_PYNQ_DEVICE
$ cp ./bitmaps/* TO_PYNQ_DEVICE 
```
Note bitmaps needs to be downloaded from [release](https://github.com/gicLAB/SECDA-TFLite/releases/tag/v1.0)
___

## Running accelerators on the board

**Now we assume we have a pynq_src folder containing everything we need on the PYNQ board.**
Map the bitstream of the accelerator on to the pynq board. PYNQ python library provides simple API call to do this.
``` shell
# opens the python interpreter (sudo access in needed for bitstream mapping)
$ cd path/to/pynq_src
$ sudo python3
from pynq import Overlay
overlay = Overlay('/path/to/bitmaps/bitmap.bit')
# to map the BERT accelerator when using the pynq_src_folder from the release
overlay = Overlay('./bitmaps/fc_acc_v1.bit') 
```


**Usage:** 
 ``` shell
 # Make binary executable
 $ cd path/to/pynq_src
 $ sudo chmod +x ./benchmark_model_secda_[vm|sa|bert]
 # Run benchmark_model tool
 $ sudo ./benchmark_model_secda_[vm|sa|bert] --graph=[path/to/tflite/model.tflite] --num_threads=[1|2] --num_runs=1 --warmup_runs=0 --enable_op_profiling=[true|false] --profiling_output_csv_file="prof.txt" --use_secda_[vm|sa|bert]_delegate=[true|false]
 ```
* "--graph" path to model
* "--num_threads" threads used [1|2]
* "--num_runs" number of inference runs to perform [1-100]
* "--warmup_runs" number of warmup runs to perform [1-100]
* "--enable_op_profiling" enables profiling
* "--profiling_output_csv_file=" save the profiling to file
* "--use_secda_[vm|sa|bert]_sim_delegate" enables the acceelerator [true|false]
  
**Example:** 
``` shell
# Runs albert_int8 model with BERT accelerator
$ sudo ./benchmark_model_secda_bert --graph=tmp/albert_int8.tflite --num_threads=1 --num_runs=1 --warmup_runs=0 --enable_op_profiling=true --profiling_output_csv_file="bert.txt"  --use_secda_bert_delegate=true 
# Runs albert_int8 model with CPU only
$ sudo ./benchmark_model_secda_bert --graph=tmp/albert_int8.tflite --num_threads=1 --num_runs=1 --warmup_runs=0 --enable_op_profiling=true --profiling_output_csv_file="cpu.txt" --use_secda_vm_delegate=false 
```
