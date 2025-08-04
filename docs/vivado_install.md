# Quick Guide on Installing Vivado 2019.2

This guide provides a quick overview of how to install Vivado 2019.2, which is required for SystemC HLS in the SECDA-TFLite project.
We recommend following the official Xilinx installation guide for detailed instructions but here are a brief guide to install Vivado 2019.2 via terminal in a Linux environment.

# Installation Steps

1. **Download Vivado 2019.2**:
   - The "All OS Installer" for Vivado 2019.2 is available for download from the Xilinx website [Vivado 2019.2](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html).
   - This [link](https://www.xilinx.com/member/forms/download/xef.html?filename=Xilinx_Vivado_2019.2_1106_2127.tar.gz) will ask you to log in to your Xilinx account before downloading the tar.gz file.
   - If you have no GUI access on the machine you are installing Vivado on then:
     - Use a machine with GUI access and start the download on that machine with browser, once the download is started, you cancel the download and get the download link by right-clicking on the download button and copying the link address.
     - Then on your target machine, you can use `wget` to download the file using the copied link:
       ```bash
       wget link_to_tar.gz
       ```
2. **Extract the Installer**:
   - Once the download is complete, extract the tar.gz file:
     ```bash
     tar -zxvf Xilinx_Vivado_2019.2_1106_2127.tar.gz
     ```

3. **Run the Setup**:
   - Navigate to the extracted directory:
     ```bash
     cd Xilinx_Vivado_2019.2_1106_2127
     ```
   - Run the setup script:
     ```bash
     ./xsetup -b ConfigGen
     ```
    - You press ‘1’, then enter, and the setup tool creates a config file in your home directory.
    - You edit the config file located at `~/.Xilinx/install_config.txt` (replace `<user>` with your actual username) to customize the installation options. You can use any text editor, for example:
      ```bash
      nano /home/<user>/.Xilinx/install_config.txt
      ```
4. **Run the Installer**:
   - After the configuration is done, run the installer:
     ```bash
     ./xsetup --a XilinxEULA,3rdPartyEULA,WebTalkTerms -b Install -c ~/.Xilinx/install_config.txt
     ```
   - Follow the on-screen instructions to complete the installation. You can choose the default options for most prompts, but ensure you select the "Design Suite" option when prompted.
5. **Post-Installation**:
   - After installation, you may need to source the Vivado settings script to set up the environment variables:
     ```bash
     source /opt/Xilinx/Vivado/2019.2/settings64.sh
     ```  
  - You can add this line to your `~/.bashrc` or `~/.bash_profile` to make it persistent across terminal sessions:
    ```bash
    echo "source /opt/Xilinx/Vivado/2019.2/settings64.sh" >> ~/.bashrc
    ```
    
6. **Patch Vivado (MUST)**:
  - Vivado has bug which Xilinx has created a patch for, you can download the patch from [here](https://adaptivesupport.amd.com/s/article/76960?language=en_US) and go to the bottom of the page and click on "y2k22_patch-1.2.zip" to download the patch.
  - Extract the patch and move the `vivado` directory to the Vivado installation directory:
    ```bash
    unzip y2k22_patch-1.2.zip
    sudo mv vivado /opt/Xilinx/Vivado/2019.2/ # Adjust the path if your Vivado is installed in a different location
    ```
  - Follow the instructions in the patch README file to apply the patch.
  - You might find that you need to change the "python" command to "python2.7" depending on what python the Vivado installation comes  with (check inside /opt/Xilinx/Vivado/2019.2/tps/lnx64/python-2.7.5/bin/)
  



