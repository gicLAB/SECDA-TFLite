# Getting Started with Kria KV260

This guide will help you set up the Kria KV260 board for use with SECDA-TFLite.

## Prerequisites

We target the Ubuntu Desktop 22.04 LTS image for Kria because KRIA-PYNQ supports this version.

**Resources:**
- [Kria-PYNQ Repository](https://github.com/Xilinx/Kria-PYNQ)
- [Ubuntu 22.04 Desktop Image Download](https://ubuntu.com/download/amd#kria-k26)
- [Installation Instructions](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/linux_boot/ubuntu_22_04/build/html/docs/sdcard.html)

## Initial Setup

1. Download the Ubuntu Desktop 22.04 LTS image from the link above
2. Follow the official installation instructions to flash the image to an SD card
3. Boot the Kria board and complete the initial setup
4. **Important:** Reboot the board after installation is complete

## Important Tips

### Safe Shutdown
Always shutdown the board properly. Do not unplug the power supply directly.
```bash
sudo shutdown -h now
```

### System Updates
Update and upgrade the system once you login to the board. This step is described in the [official boot guide](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/linux_boot/ubuntu_22_04/build/html/docs/sdcard.html).

Install application-specific repositories, Ubuntu updates, and upgrade the system (this may take 10-20 minutes):
```bash
sudo add-apt-repository ppa:xilinx-apps --yes &&
sudo add-apt-repository ppa:ubuntu-xilinx/sdk --yes &&
sudo add-apt-repository ppa:xilinx-apps/xilinx-drivers --yes &&
sudo add-apt-repository ppa:lely/ppa --yes &&
sudo apt update --yes &&
sudo apt upgrade --yes
```

### System Verification

**Check firmware version:**
```bash
uname -r
```

**Check kernel configuration:**
```bash
zcat /proc/config.gz | grep CONFIG_STRICT_DEVMEM
```
The output should show `CONFIG_STRICT_DEVMEM is not set` indicating the configuration is correct. Otherwise, you need to rebuild the kernel following the guide below.

**Check CMA allocation:**
```bash
cat /proc/meminfo | grep -i cma
# or
sudo dmesg | grep -i cma
```


## Why Kernel Rebuilding is Necessary

### Memory Access Requirements
In SECDA-TFLite, we use `mmap()` to access memory during `dma_init()`. However, the `mmap()` function will not work if `CONFIG_STRICT_DEVMEM` is enabled in the kernel. By default, `CONFIG_STRICT_DEVMEM` is enabled, so we need to disable it to use the `mmap()` function.

### Common Error Scenario
If you encounter failures when running `secda_benchmark_suite.sh` immediately after loading the bitstream, this is likely due to the `mmap()` function in `dma_init()`. In this case, you need to rebuild the kernel with the new configuration.

# Rebuilding the Kernel Image

We need to rebuild the Ubuntu image for the Kria board to change some Linux kernel configurations.

**Reference:** [AMD Adaptive Support Guide](https://adaptivesupport.amd.com/s/feed/0D54U00007wzkKPSAY)

> **Note:** The following steps are also applicable to Ubuntu 20.04 - use `focal` where `jammy` is referenced below.

## Step 1: Configure the Build Environment

Before fetching the Linux kernel sources, configure the build environment with tools such as `git`, Aarch64 `gcc`, and `fakeroot`.

```bash
echo "deb-src http://archive.ubuntu.com/ubuntu jammy main" | sudo tee -a /etc/apt/sources.list.d/jammy.list

sudo apt-get update

sudo apt-get build-dep linux

sudo apt-get install git fakeroot libncurses-dev gcc-aarch64-linux-gnu linux-tools-common
```

## Step 2: Clone the Linux Kernel Source Code

```bash
git clone https://git.launchpad.net/~canonical-kernel/ubuntu/+source/linux-xilinx-zynqmp/+git/jammy
```

After cloning the source code, switch to the latest tag. To check for the latest tag:

```bash
cd jammy
git tag
```

Then checkout the latest tag:

```bash
git checkout tags/debPkgs_5.15.0-1046.50
```

## Step 3: Configure the Build Environment

**Set the target architecture:**
```bash
export ARCH=arm64
```

**Export dpkg variables for packaging Debian packages for the target (arm64) architecture:**
```bash
export $(dpkg-architecture -aarm64)
```

**Handle cross-compilation warnings:**
If you see the following warning:
```
dpkg-architecture: warning: specified GNU system type aarch64-linux-gnu does not match CC system type x86_64-linux-gnu, try setting a correct CC environment variable
```

Then run:
```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++

export $(dpkg-architecture -aarm64)
```

**Set cross-compilation variables:**
Since we are compiling in an x86_64 environment for an arm64 target:
```bash
export CROSS_COMPILE=aarch64-linux-gnu-
```

## Step 4: Edit the Linux Kernel Configuration

**Distinguish the kernel build:**
First, make a quick edit to distinguish our kernel build from the binaries installed from Canonical. Add `+local` to the version in the changelog:

```bash
fakeroot debian/rules clean
vi debian.zynqmp/changelog
```

**Modify kernel configuration:**
If the Linux kernel configuration requires modification, the standard Linux kernel menuconfig can be accessed with the standard Debian editconfigs rule. This is where new drivers, modules, or features can be enabled.

For our case, we need to disable the `CONFIG_STRICT_DEVMEM` option:

1. Go to the **Kernel hacking** section
2. Find **Filter access to /dev/mem** option
3. Press **N** to disable it
4. Save the configuration and exit

```bash
fakeroot debian/rules clean
fakeroot debian/rules editconfigs
```

## Step 5: Build the Linux Kernel

**Prepare for build:**
To build the Linux kernel, use the standard Debian rules system with the `binary` target. The output of this process will be a series of standard `.deb` Debian packages.

```bash
fakeroot debian/rules clean
```

**Choose your build method:**
There are two ways to build the kernel:

**Method 1 - Standard build:**
```bash
do_tools=false fakeroot debian/rules binary
```

**Method 2 - Skip checks (recommended for configuration changes):**
If you are making changes to the configuration that may impact modules and/or the ABI, then you may need to skip those checks as shown below:

```bash
do_tools=false skipmodule=true skipconfig=true skipabi=true fakeroot debian/rules binary
```

**Build output:**
The generated `.deb` packages are located one directory higher than the Linux kernel source directory.

## Step 6: Install the Linux Kernel Packages

**If compiled on target (Kria board):**
If the kernel was compiled directly on the target device, installing the new kernel is straightforward. Navigate to the directory where the build process placed the Debian packages and install them with the `dpkg` tool:

```bash
cd ..  # or to wherever the .deb packages are located
sudo dpkg -i *.deb
```

**If cross-compiled (x86_64 to ARM64):**
If you compiled the kernel on an x86_64 system for the ARM64 Kria board, you need to copy the generated `.deb` files to the Kria board and install them there:

1. **Copy files to Kria board:**
   Transfer the generated `.deb` files to the Kria board's `/boot/firmware` directory by enabling root access:

2. **Install on Kria board:**
   ```bash
   cd /boot/firmware
   sudo dpkg -i *.deb
   ```

**Reboot the system:**
After installing the new kernel, reboot the Kria board to boot with the new kernel:

```bash
sudo reboot
```

## Step 7: Verify the New Kernel

**Check kernel version:**
After the Kria board has rebooted, verify the new kernel is running with the `uname` command:

```bash
uname -r
```

**Verify configuration:**
You can also verify that the `CONFIG_STRICT_DEVMEM` option is properly disabled by checking the kernel configuration:

```bash
zcat /proc/config.gz | grep STRICT_DEVMEM
```

The output should show `CONFIG_STRICT_DEVMEM is not set`, confirming the configuration change was successful.

**Test functionality:**
Once verified, you can proceed to test the SECDA benchmark suite to ensure the kernel changes resolve any `mmap()` related issues.

---

## Troubleshooting

If you encounter any issues during the kernel rebuild process:

1. **Build failures:** Ensure you have sufficient disk space and all required dependencies
2. **Configuration issues:** Double-check that `CONFIG_STRICT_DEVMEM` is properly disabled
3. **Boot problems:** Keep the original kernel packages as backup for recovery
4. **Permission errors:** Ensure proper sudo access for installation steps

For additional support, refer to the main SECDA-TFLite documentation or contact the development team.

Check the kernel configuration with the ```zcat``` command.

```
zcat /proc/config.gz | grep CONFIG_STRICT_DEVMEM
```

The output should show the ```CONFIG_STRICT_DEVMEM is not set``` indicating the configuration change was successful.

## If Kernel gets updated automatically

These instructions are for the case when the kernel gets updated automatically.  In this case, the kernel will be updated to the latest version and the configuration changes will be lost.  In this case, the kernel will need to be rebuilt with the desired configuration changes for the new kernel version.

- If you download the kernel source code from the website, jammy, remove the old source code and start from the beginning step [**Configure the Build Environment**](#configure-the-build-environment).

## Error handling: During the build process, if you see the following error

```
find: 'dwarfdump': No such file or directory
```

This is due to the fact that the ```dwarfdump``` tool is not installed.  To install it, run the following command:

```
sudo apt install dwarfdump
```
