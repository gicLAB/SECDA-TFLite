# SECDA-TFLite: A Toolkit for Efficient Development of FPGA-based DNN Accelerators for Edge Inference

The SECDA-TFLite toolkit leverages the TFLite delegate system to provide a robust and extensible set of utilities for integrating DNN accelerators for any DNN operation supported by TFLite.
Ultimately, this increases hardware accelerator developers' productivity, as they can begin developing and refining their design more quickly.


# Internal Developer Info
* Accelerator Design source code can be found inside the respective simulation delegate
  * Note: we use the same source files for HLS, we manually define __SYNTHESIS__ before HLS

### Setup repo
```
git clone https://github.com/judeharis/SECDA-TFLite.git
cd SECDA-TFLite
git submodule init
git submodule update
cd tensorflow 
git checkout secda-tflite-v1
./configure
```
* The default configuration is good as along as you have atleast python-3.7+

### Setup Bazel
```
sudo apt install apt-transport-https curl gnupg -y && \
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
mv bazel-archive-keyring.gpg /usr/share/keyrings && \
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
sudo apt update && sudo apt install bazel-3.7.2 -y && \
sudo ln -s /usr/bin/bazel-3.7.2 /usr/bin/bazel
```

### Test Bazel
```
cd tensorflow
bazel build --jobs 1 //tensorflow/lite/examples/systemc:hello_systemc
bazel run //tensorflow/lite/examples/systemc:hello_systemc
```


## Optional
```
sudo apt-get -y install gdb
```



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
* For more information please check out the [accelerator source-code](secda_tflite_accel)

## Synthesis for PYNQ-Z1
To perform logic synthesis, we provide Vivado project folders ([release](https://github.com/gicLAB/SECDA-TFLite/releases/tag/v1.0)). These contain the necessary block diagram configuration including AXI DMA's and the accelerators to ensure correct connectivity to the processing system.

**Requirements**
* Vivado HLS 2018.3
* Vivado 2018.3


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