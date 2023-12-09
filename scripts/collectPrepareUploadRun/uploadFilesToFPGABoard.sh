# first it will check the connection with the board
# this files is used to copy the files to the fpga board



copyFrom="$filesForFPGA_path/$trial_name"
copyTo="/home/xilinx/binariesAndDesign"

## create copyTo folder if not exists
ssh xilinx@$ip_addr_FPGA_board "[ ! -d $copyTo ] && mkdir -p $copyTo"

## remove the test folder from the pynq board
ssh xilinx@$ip_addr_FPGA_board "rm -rf $copyTo/$trial_name"

## copy files over
scp -r $copyFrom xilinx@$ip_addr_FPGA_board:$copyTo

echo "files copied to the FPGA board!!!!"