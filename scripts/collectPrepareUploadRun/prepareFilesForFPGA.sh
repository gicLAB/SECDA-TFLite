# this file is used to prepare the files for FPGA
# copy the binary to filesForFPGA folder
# copy the .bit and .hwh files to filesForFPGA folder
# create a load.py file in filesForFPGA folder


copyFilesFrom="$result_path/$trial_name"
copyFilesTo="$filesForFPGA_path/$trial_name"

if [ ! -d "$copyFilesTo" ]; then
    mkdir -p "$copyFilesTo"
fi

#create a binary name from three variables
binaryName="bin_"$app_Name"_"$del_name"_"$trial_name""

## copy the binary to the pynq share folder
rm -f $copyFilesTo/$binaryName
cp $copyFilesFrom/$binaryName $copyFilesTo/

## copy .bit and .hwh files to the pynq share folder
rm -f $copyFilesTo/*.bit
rm -f $copyFilesTo/*.hwh
cp $copyFilesFrom/hlx_output/*.bit $copyFilesTo/
cp $copyFilesFrom/hlx_output/*.hwh $copyFilesTo/


## preapre load.py file
# create a new load.py file
rm -f $copyFilesTo/load.py
touch $copyFilesTo/load.py
# add the following lines to the load.py file
echo "from pynq import Overlay" >> $copyFilesTo/load.py
echo "overlay = Overlay(\"/home/xilinx/binariesAndDesign/$trial_name/$trial_name.bit\")" >> $copyFilesTo/load.py
