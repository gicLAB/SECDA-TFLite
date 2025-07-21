# Hardware Automation

## Purpose
The hardware automation toolkit automates the process of running High Level Synthesis (HLS) and logic synthesis for hardware designs created within SECDA-TFLite. It provides utilities for hardware generation using json-based configuration files and TCL scripts.

**NOTE**: The HLS and logic synthesis scripts needs to run outside the VSCode dev container to work properly with Vivado/Vitis.


## Table of Contents
- [Hardware Automation](#hardware-automation)
  - [Purpose](#purpose)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Hardware Configuration](#hardware-configuration)
    - [Example Configuration File](#example-configuration-file)
  - [Project Generation](#project-generation)
  - [Hardware Generation](#hardware-generation)
  - [Notes](#notes)

## Overview

The hardware automation toolkit consists of the following components located in the `hardware_automation/` folder:

1. **Hardware Configurations (`configs/`)** - Configuration files for different accelerators
   - Each accelerator should have its own directory under `configs/`
   - Within each directory, you can define new accelerators by creating a new JSON file (new json per version of the accelerator)
2. **Hardware Generation (`hw_gen.py`)** - Main Python script for automating HLS and logic synthesis
3. **Template Script (`hw_gen.tpl.sh`)** - Shell script template for hardware generation workflows
4. **HLX TCL Scripts (`hlx_scripts/`)** - Template TCL scripts used for logic synthesis
5. **Accelerator Sources (`acc_srcs/`)** - Temporary directory for accelerator source files (symlinked to delegates's accelerator source files)
6. **Generated Output (`generated/`)** - Generated HW projects and synthesis results

## Prerequisites

Before using the hardware automation tool, ensure that you have the following prerequisites installed:
- **Python 3.8+** 
- **Vivado Design Suite 2019.1** *(Required for SystemC HLS)*
- **Vitis Design Suite 2024.2** (for KV260 Board)


## Hardware Configuration
The hardware configuration files are located in the `configs/` directory. 
Using these configuration files, you can define the hardware parameters for different accelerators.
The options available in the configuration files include:
- "acc_name": Name of the accelerator
- "acc_version": Version of the accelerator
- "acc_sub_version": Sub-version of the accelerator
- "acc_src": Path to the accelerator source files
- "acc_link_folder": Name of folder where the accelerator source files symlinked inside the `acc_srcs/` directory
- "hlx_tcl_script": Name of the TCL script used for HLX synthesis
- "top": Top-level module name for the accelerator (use the name that is defined as ACCNAME in the acc_config.sc.h file of your accelerator)
- "del": Name of the delegate used for the accelerator
- "del_version": Version of the delegate (the delegate folder should have different versions, "1" for the "v1" delegate sub-folder)
- "hls_clock": HLS clock frequency in nanoseconds (e.g., "4" for 250 MHz)
- "hlx_Mhz": HLX clock frequency in MHz
- "axi_bitW": Data bit-width for AXI interface (only supports 32 currently)
- "axi_burstS": AXI burst size (only supports 32 currently)
- "board": Target board for synthesis (supported values: "Z1", "KV260"), use "Z1" for Zynq-7000 boards and "KV260" for Kria SOM boards


### Example Configuration File
```json
{
  "acc_name": "ADD_ACC",
  "acc_version": 2,
  "acc_sub_version": 0,
  "acc_src": "add_delegate/v2/accelerator",
  "acc_link_folder": "add_src_v2",
  "hlx_tcl_script": "mdma_axilite_hlx_reconfig_1_xfull.tcl",
  "top": "ADD_ACC",
  "del" : "add_delegate",
  "del_version": "2",
  "hls_clock" : "5",
  "hlx_Mhz": "200",
  "axi_bitW": "32",
  "axi_burstS": "32",
  "board": "Z1"
}
```


## Project Generation
The [`hw_gen.py`](./hw_gen.py) script is used to create the base hardware automation project from a given json configuration file. To generate a hardware project, run the following command in the terminal:
```bash
cd hardware_automation
python3 hw_gen.py [path_to_config_file] # Example: python3 hw_gen.py configs/ADD/ADDv1_0.json
```
The script will create a new hardware project in `generated/` which will contain all the necessary files for synthesis.



## Hardware Generation
Once the hardware project is generated, you can automate the process of running High Level Synthesis (HLS) and logic synthesis and copying files to the target board using the `run.sh` script generated in the `generated/[acc_name]` directory.
To run the hardware generation workflow, first exit the VSCode dev container if you are using it, then navigate to the generated hardware project directory and execute the following command:
```bash
cd generated/[acc_name]
./run.sh [run_hls] [run_hlx] [run_remote] [transfer_bitstream]
```
Where:
- `run_hls`: Set to `1` to run HLS, `0` to skip
- `run_hlx`: Set to `1` to run HLX synthesis, `0` to skip
- `run_remote`: Set to `1` to run HLS/HLX synthesis on a remote server, `0` to run locally
- `transfer_bitstream`: Set to `1` to transfer the generated bitstream to the target board, `0` to skip


Example:
```bash
cd generated/ADD_ACC
./run.sh 1 1 0 1 # This will run HLS and HLX synthesis locally and transfer the bitstream to the target board
```

## Notes

- A successful HLX run will automatically try to transfer the generated bitstream to the target board using rsync. Ensure that the target board is reachable via SSH and the necessary credentials are set up.
- The `run.sh` script is generated from the `hw_gen.tpl.sh` template file, which can be customized to include additional commands or configurations as needed.
- If the hardware configuration file is updated, then you need to regenerate the hardware project by running the `hw_gen.py` script again with the updated configuration file.
- Running `run.sh`  will overwrite any existing files generated by previous runs.
- The `run.sh` with HLS will automatically copy in the latest source files from the `acc_srcs/` directory and since those files are symlinked to the accelerator source files within the `secda_delegates/` directory, you can update the accelerator source files in the `secda_delegates/` directory and then regenerate the hardware project to use the latest source files, without needing to modify the hardware configuration file.
- Note that HLX is based on the `hlx_tcl_script`, this needs to be updated if your accelerator IP has different ports or requires a custom block design. You can create a new template TCL script using the [HLX Script Templater](hlx_templater/readme.md) tool.
