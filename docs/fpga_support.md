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

