# SECDA-TFLite: A Toolkit for Efficient Development of FPGA-based DNN Accelerators for Edge Inference (Needs Update)

The SECDA-TFLite toolkit leverages the TFLite delegate system to provide a robust and extensible set of utilities for integrating DNN accelerators for any DNN operation supported by TFLite.
Ultimately, this increases hardware accelerator developers' productivity, as they can begin developing and refining their design more quickly.

## Repo Structure
Overview of the repo structure with the important directories presented below:
```
SECDA-TFLite_v1.2/
├── LICENSE
├── README.md
├── WORKSPACE
├── config.json
├── data/
│   ├── inputs/
│   └── models/
├── docs/
├── hardware_automation/
├── scripts/
├── src/
│   ├── benchmark_suite/
│   ├── experimental/
│   ├── secda_apps/
│   ├── secda_generator/
│   ├── secda_delegates/
│   ├── secda_profilier/
│   ├── secda_tflite/
├── tensorflow/
```

# Installation

## 0. Requirements and Recommendations
---
- Debian-based linux distro (highlighy recommended)
  - Install [docker for linux](https://docs.docker.com/engine/install/ubuntu/)
  - Install git: ```sudo apt install git```
- Otherwise, if using Windows:
   - Install WSL for windows and use Ubuntu 22.04
   - Install [docker for windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- Install [VSCode](https://code.visualstudio.com/download) (highlighy recommended)
   - Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
 - Hardware Synthesis (Not required for simulation but recommended for updating simulation timing using HLS):
   - Vivado 2019.2 (required for SystemC HLS)
   - Vitis 2024.2 (for logic synthesis for KV260)


## 1. Repo Download and Configuration (#TODO: Update paths)
### Download the SECDA-TFLite repo and install basic dependencies
Make sure you are in linux-based workspace environment with git installed. Run the following commands to download the SECDA-TFLite repo and install the basic dependencies:
```bash
git clone git@github.com:judeharis/SECDA-TFLite.git
cd SECDA-TFLite
git checkout v1.3
git submodule init
git submodule update
sudo apt install -y jq
```

### Configure the SECDA-TFLite paths
To enable all the tools of SECDA-TFLite has access to correct paths, you need to configure the [config.json](./config.json) file. This file contains the paths to the various directories and files used by the SECDA-TFLite toolkit. You **MUST** update the paths in the `config.json` file to point to the correct directories in your system. Please refer to the [config.md](./docs/config.md) file for more information on how to configure the `config.json` file.


### Run the setup script
Once you have configured the `config.json` file, you can run the setup script and checkout to the correct submodule branch of the tensorflow repo:
```bash
cd scripts
./setup.sh # you might have to make this script executable "chmod +x ./setup.sh
cd ../tensorflow
git checkout secda-tflite-v2_prerelease
```

Now you have the SECDA-TFLite repo downloaded and the basic dependencies installed. You can now proceed to set up the development environment using VSCode dev containers (2A - highly recommended) or natively on your system (2B).

## 2.A: Using VSCode Dev Container
- Ensure docker is up and running and current user is part of docker group
  - ``` sudo usermod -aG docker $USER ```
- Install "Dev Containers" extension https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
- Reload the workspace
- These following pop-up should appear otherwise you can open the command palette (Ctrl+Shift+P) and search for "Dev Containers: Reopen in Container"

![alt text](docs/image-1.png)
- Press `Reopen in Container`
- It should take a while to download and install the container.
- Once the container is created it should reopen you into the VSCode with the container active.
- You can access the container through "Dev Containers: Open Folder in Container" VSCode command.
  

## 2.B: Create development environment on natively
### Setup Bazel & GDB
```bash
sudo apt install apt-transport-https curl gnupg -y && \
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings && \
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list && \
sudo apt update && sudo apt install bazel-6.1.0 -y && \
sudo ln -sf /usr/bin/bazel-6.1.0 /usr/bin/bazel6
sudo apt-get -y install gdb
```

### Setup miniconda environment needed for Tensorflow build process (Working with Ubuntu 22.04) 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda config --set auto_activate_base false
conda create -n secda-tflitev2 python -y
conda activate secda-tflitev2
pip install -r ./devcontainer/requirements.txt
```

###  Configure Tensorflow & Test Bazel build (make sure to activate secda-tflitev2 environment)
```bash
conda activate secda-tflitev2
cd tensorflow
# make sure to set python path to /home/your_user/miniconda3/bin/python3
./configure # make sure say no to clang as the default compiler
bazel build --jobs 1 //tensorflow/lite/examples/systemc:hello_systemc
bazel run //tensorflow/lite/examples/systemc:hello_systemc
```

Once the environment is created, we recommend using VSCode to immediately start developing. Checkout the VSCode instructions below.



  

## 4. VSCode Instructions
* Load VSCode `SECDA-TFLite.code-workspace` using "open workspace from file" option in the VSCode File menu. Note: within the container this workspace will be located at `/working_dir/SECDA-TFLite.code-workspace`.

* Once the VSCode workspace is loaded, you are able to run to the launch configurations through the [Run and Debug](https://code.visualstudio.com/docs/editor/debugging) tab to run the end to end simulation.
* These configurations are stored within '/tensorflow/.vscode/launch.json' (to launch) and /tensorflow/.vscode/task.json (to compile), you can edit these to change the parameters to compile and launch the the end to end simulation.
* There are some configurations already prepared to run the VM,SA and FC-GEMM accelerator with the simulation delegates






# Paper
Our research paper covers the SECDA-TFLite toolkit in detail including case studies where we design and integrate new DNN accelerator using our toolkit. If you are using the SECDA-TFLite toolkit in research, we kindly request a reference to the following:

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
