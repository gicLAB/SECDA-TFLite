# this script is used to run the files in the FPGA board
# first upload the bitstream
# then run the binary


copyTo="/home/xilinx/binariesAndDesign"

runFilesInFPGABoard_bitstream_flag
## upload the bitstream
if [ $runFilesInFPGABoard_bitstream_flag -eq 1 ]; then
    echo "xilinx" | ssh -t xilinx@$ip_addr_FPGA_board "sudo -S bash -c 'source /etc/profile.d/xrt_setup.sh; source /etc/profile.d/pynq_venv.sh; pwd; cd \"$copyTo\"; pwd; python $trial_name/load.py'"

    echo "!!!!!!!!!bitstream loaded!!!!!!!!!"
fi

## run the binary
if [ $runFilesInFPGABoard_binary_flag -eq 1 ]; then
    echo "xilinx" | ssh -t xilinx@$ip_addr_FPGA_board "cd \"$copyTo\"; pwd; chmod +x $trial_name/bin_*; sudo -S ./$trial_name/bin_* -mmodels/$modle_name -imodels/inputs/$image_name -lmodels/inputs/$label_name -t $num_threads --use_secda_vm_delegate=true -c $num_runs"

    echo "!!!!!!!!!binary run completed!!!!!!!!!"
fi
