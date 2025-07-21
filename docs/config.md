# SECDA-TFLite Configuration

The `config.json` file is essential for the SECDA-TFLite toolkit to function correctly. It contains paths to various directories and files used by the toolkit.

**NOTE**: SECDA-TFLite when run from the VSCode Dev Container will use the [config.json](../.devcontainer/config.json) file located in [.devcontainer](../.devcontainer).
This config file will be updated automatically depending on the root [config.json](./config.json) file when you open the Dev Container in VSCode.

## Config Fields

The following explains the key attributes in the `config.json` file:

**secda_tflite_path**:
- This is the root directory of the SECDA-TFLite toolkit.
- It should point to the directory where the SECDA-TFLite toolkit is located.

**vivado_2019_path**: Absolute path 
- This is the path to the Vivado 2019 installation.
- This is a critical path for the hardware automation scripts to function correctly.
- Not used if you are using remote server for hardware automation.


**vivado_2024_path**: Absolute path
- This is the path to the Vivado 2024 installation.
- This is a critical path for the hardware automation scripts to function correctly for KV260 board.
- Not used if you are using remote server for hardware automation.

**models_dirs**: Absolute path
- The benchmark suite will look for models in these directories when running benchmarks, and will send the folders with the models to the board (when the `Initialize boards` or `Send models` boxes are checked in the GUI).
- Keep the model folders inside the SECDA-TFLite as shown in default config. (Update absolute path according to where your SECDA-TFLite is located)

**path_to_dels**: relative path to the `secda_tflite_path`
- This is the path to the secda delegates folder within the tensorflow submodule.
- The default value should be kept as is unless you have a custom setup.

**hw_configs**: relative path to the `secda_tflite_path`
- It should point to the directory where the hardware configuration files are located.
- The default value should be kept as is unless you have a custom setup.

**hlx_scripts**: relative path to the `secda_tflite_path`
- It should point to the directory where the hardware tcl template scripts are located.
- The default value should be kept as is unless you have a custom setup.




**out_dir**:
- This is the output folder name that is used generally across the toolkit.
- Do not change from the default value unless you have a specific reason to do so.



  
**boards**:
- This is a dictionary that contains the configuration for each board.
- Each board configuration should follow this example format:
```json
    {"Z1": {
      "board_user": "username",
      "board_hostname": "ip_address",
      "board_port": "board_port",
      "board_dir": "/home/username/path/to/workspace/secda_tflite/",
      "fpga_part": "xc7z020clg400-1",
      "hlx_version": "2019"
    }}
```

**remote_server**:
- This is a dictionary that contains the configuration for the remote server.
- Each remote server configuration should follow this example format:
```json
    {    
    # Optional, if you need to connect through a gateway, leave empty if not needed 
    "gateway": "-J gateway_user@gateway_ip", 
    "server_user": "remote_user",
    "server_hostname": "remote_ip",
    "server_port": remote_port,
    "server_dir": "/home/jharis/Workspace/remote/SECDA-TFLite/hardware_automation_remote/",
    "server_2019_path": "/path/to/Vivado/2019.2/bin/",
    "server_2024_path": "/path/to/Vivado/2024.1/bin/"
    }
```

**push_bullet_token**:
- This is the Pushbullet API token used for sending notifications to you via Pushbullet so that you can be notified when the hardware automation scripts are done running.
- If you do not want to use Pushbullet, you can leave this empty.
- If you want to use Pushbullet, set up an account and get your API token from the [Pushbullet website](https://www.pushbullet.com/).


## SSH Configurations

- If you are using a remote server for hardware automation, you make sure your remote server is reachable via SSH key authentication.
- Recommend setting up your key to the remote server using [ssh-copy-id](https://www.ssh.com/academy/ssh/copy-id) command:
```bash
sudo apt install ssh-copy-id
ssh-copy-id remote_user@remote_ip
```

- Similarly, if you are using a target board for benchmarking, you need to make sure the board is reachable via SSH key authentication.
- You can use the same `ssh-copy-id` command to copy your SSH key to the target board:
```bash
ssh-copy-id board_user@board_ip
```

- It is ideal if you can remove the password authentication for SSH to avoid any interruptions during the hardware automation scripts execution.