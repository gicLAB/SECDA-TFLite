# SECDA-TFLite: A Toolkit for Efficient Development of FPGA-based DNN Accelerators for Edge Inference

The SECDA-TFLite toolkit leverages the TFLite delegate system to provide a robust and extensible set of utilities for integrating DNN accelerators for any DNN operation supported by TFLite.
Ultimately, this increases hardware accelerator developers' productivity, as they can begin developing and refining their design more quickly.


# Internal Developer Info
* Accelerator Design source code can be found inside the respective simulation delegate
  * Note: we use the same source files for HLS, we manually define __SYNTHESIS__ before HLS
 

## 0. Requirements and Recommendations
- Use a Debian-based linux distro
  - Install [docker for linux](https://docs.docker.com/engine/install/ubuntu/)
- Otherwise, if using Windows:
   - Install WSL for windows and use Ubuntu 22.04
   - Install [docker for windows](https://docs.docker.com/desktop/setup/install/windows-install/)

## 1. Setup repo
Make sure you are in linux-based workspace environment with git installed.
```bash
git clone git@github.com:judeharis/SECDA-TFLite.git
cd SECDA-TFLite
git checkout v1.2
git submodule init
git submodule update
cd scripts
./create_symlink.sh # you might have to make this script executetable "chmod +x ./create_symlink.sh
cd ../tensorflow 
git checkout secda-tflite-v1_tf_r2.15
sudo apt install -y jq
```

You can now setup the dev environment natively (2.A) or via docker (2.B).
 
## 2.A Create dev environment on natively
### Setup Bazel
```bash
sudo apt install apt-transport-https curl gnupg -y && \
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings && \
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list && \
sudo apt update && sudo apt install bazel-6.1.0 -y && \
sudo ln -sf /usr/bin/bazel-6.1.0 /usr/bin/bazel6
```

### Setup miniconda environment needed for Tensorflow build process (Working with Ubuntu 22.04) 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda config --set auto_activate_base false
conda create -n secda-tflite python -y
conda activate secda-tflite
pip3 install numpy
```

###  Configure Tensorflow & Test Bazel build (make sure to activate secda-tflite environment)
```bash
conda activate secda-tflite
cd tensorflow
# make sure to set python path to /home/your_user/miniconda3/bin/python3
./configure # make sure say no to clang as the default compiler
bazel build --jobs 1 //tensorflow/lite/examples/systemc:hello_systemc
bazel run //tensorflow/lite/examples/systemc:hello_systemc
```


### Optional
```bash
sudo apt-get -y install gdb
```

Once the environment is created, we recommend using VSCode to immediately start developing. Checkout the VSCode instructions below.

## 2.B Use VSCode Dev Container for containerized version
- Ensure docker is up and running and current user is part of docker group
  - ``` sudo usermod -aG docker $USER ```

- Install "Dev Containers" extension https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
- Reload the workspace
- These following popup should appear
  
![alt text](docs/image-1.png)
- Press Reopen in Container
- It should take a while to download and install the container.
- Once the container is created it should reopen you into the VSCode with the container active.
- You can access the container through "Dev Containers: Open Folder in Container" VSCode command.
  
## VSCode Instructions
* Load VSCode `SECDA-TFLite.code-workspace` using "open workspace from file" option in the VSCode File menu. Note: within the container this workspace will be located at `/working_dir/SECDA-TFLite.code-workspace`.

* Once the VSCode workspace is loaded, you are able to run to the launch configurations through the [Run and Debug](https://code.visualstudio.com/docs/editor/debugging) tab to run the end to end simulation.
* These configurations are stored within '/tensorflow/.vscode/launch.json' (to launch) and /tensorflow/.vscode/task.json (to compile), you can edit these to change the parameters to compile and launch the the end to end simulation.
* There are some configurations already prepared to run the VM,SA and FC-GEMM accelerator with the simulation delegates




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


# Docker Setup for SECDA-TFLite-dev (TF2.7) (Deprecated)
We highly recommend using this container to immediately start developing accelerators and accelerator delegates using the SECDA-TFLite toolkit.

**Requirements**
* Docker
* VSCode
* jq
* gdb

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
