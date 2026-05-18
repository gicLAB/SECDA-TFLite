# FPGA Setup and Support Documentation

SECDA-TFLite supports FPGA development for board with [PYNQ](https://www.pynq.io/boards.html) support.

## Tested Devices
- Xilinx Pynq-Z1
- Xilinx Pynq-Z2
- Xilinx Kria KV260 Vision AI

## Board Setup
- We assume you have done default setup of the board, and you have access to the board via SSH.
- Copy [load_bitstream.py](../scripts/load_bitstream.py) to the board at your home directory.
   ``` rsync -avz ./scripts/load_bitstream.py <username>@<board_ip>:~/ ```
- Ensure you have the necessary permissions to run the script. You can set the permissions using:
  ```bash
  chmod +x load_bitstream.py
  ```

- Run the script to load the bitstream:
  ```bash
  sudo python3 load_bitstream.py
  ```
- You should get the following output:
  ```bash
  usage: load_bitstream.py [-h] [-q] bitstream
  load_bitstream.py: error: the following arguments are required: bitstream
  ```

- Otherwise, you might have to use the following command to load the bitstream:
  ```bash
  sudo -i python3 load_bitstream.py
  ```


- This means the script is ready to load the bitstream, you can now run the script with the path to your bitstream file:
  ```bash
  python3 load_bitstream.py /path/to/your/bitstream.bit
  ```

If you are able to load the bitstream successfully with this script then SECDA-TFLite should also be able to load the bitstream when run the [Benchmark Suite](../src/benchmark_suite/readme.md).

## Board Setup for SECDA Apps Evaluation Suite

Before running the [SECDA Apps Evaluation Suite](../src/secda_apps_evaluation_suite/readMe.md), perform these setup steps on your target board (Z1, Z2, or Kria).

### 1. Configure SSH Key-Based Authentication

Set up passwordless SSH access so the suite can deploy and run experiments without prompts:

```bash
ssh-copy-id -p <board_port> <board_user>@<board_hostname>
```

Then verify login works:
```bash
ssh -p <board_port> <board_user>@<board_hostname> 'exit'
```

### 2. Configure Sudo Without Password Prompt

The evaluation suite uses `sudo` to load bitstreams and manage power collection. Enable passwordless sudo for the board user:

```bash
ssh -p <board_port> <board_user>@<board_hostname> 'sudo nano /etc/sudoers'
```

Add this line at the end (replace `<board_user>` with your actual username):
```
<board_user> ALL=(ALL) NOPASSWD: ALL
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`). **Without this, bitstream loading and power collection will fail.**

### 3. Set Up XRT Environment (Required for Bitstream Loading)

Create `/etc/profile.d/xrt_setup.sh` on the board:

```bash
ssh -p <board_port> <board_user>@<board_hostname> 'sudo tee /etc/profile.d/xrt_setup.sh' << 'EOF'
export XILINX_XRT=/usr
EOF
```

### 4. Set Up PYNQ Virtual Environment (Kria Only)

For Kria boards, create `/etc/profile.d/pynq_venv.sh`:

```bash
ssh -p <board_port> <board_user>@<board_hostname> 'sudo tee /etc/profile.d/pynq_venv.sh' << 'EOF'
source /usr/local/share/pynq-venv/bin/activate
export PYNQ_JUPYTER_NOTEBOOKS=/root/jupyter_notebooks
export BOARD=KV260
export XILINX_XRT=/usr
export PATH=$PATH:/usr/local/share/pynq-venv/bin/microblazeel-xilinx-elf/bin/
python3 /usr/local/share/pynq-venv/pynq-dts/insert_dtbo.py
EOF
```

If setup is complete, proceed to the [SECDA Apps Evaluation Suite](../src/secda_apps_evaluation_suite/readMe.md).


