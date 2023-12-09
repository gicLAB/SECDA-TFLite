####### USER INPUTS ########

# give the delegate name
del_name="secda_vm_delegate"
# give the app name
app_Name="eval_model"
# give the name of the experiment
trial_name="G4_6PE_ORI_DSP_MULT_200M"


ip_addr_FPGA_board="192.168.2.99"
modle_name="mobilenetv1.tflite"
image_name="grace_hopper.bmp"
label_name="labels2.txt"
num_threads="2"
num_runs="100"

#### flags #####
#create a flag array like [1,1,1,1] to run all the scripts
flags=(1 1 1 0 1)

collectBinaries_flag=${flags[0]}
prepareFilesForFPGA_flag=${flags[1]}
uploadFilesToFPGABoard_flag=${flags[2]}
runFilesInFPGABoard_bitstream_flag=${flags[3]}
runFilesInFPGABoard_binary_flag=${flags[4]}

####### USER INPUTS END ########

## run other sh
path="$(pwd)"

## find the path for result folder
cd ..
result_path="$(pwd)/results"

## find the path for filesForFPGA folder
filesForFPGA_path="$(pwd)/filesForFPGA"

cd $path

## for windows base WSL
# SecdaTFLitePath="/mnt/d/workspace/SecdaTFLite/SecdaTfliteUpdated/SECDA-TFLite/tensorflow"

## for linux base
SecdaTFLitePath="/home/rppv15/workspace/SECDA-TFLite/tensorflow"

bazel_build_path="//tensorflow/lite/delegates/utils/secda_delegates/$del_name:$app_Name""_plus_$del_name"

binaries_path="$SecdaTFLitePath/bazel-out/armhf-opt/bin/tensorflow/lite/delegates/utils/secda_delegates/$del_name/$app_Name""_plus_$del_name"

# check $result_path trial_name folder exists or not
if [ ! -d "$result_path/$trial_name" ]; then
    mkdir -p "$result_path/$trial_name"
    echo "$result_path/$trial_name folder created .hwh and .bit files may be missing!!"
fi

# check $filesForFPGA_path trial_name folder exists or not
if [ ! -d "$filesForFPGA_path/$trial_name" ]; then
    mkdir -p "$filesForFPGA_path/$trial_name"
fi

if [ $collectBinaries_flag -eq 1 ]; then
    source collectBinaries.sh
fi

if [ $prepareFilesForFPGA_flag -eq 1 ]; then
    cd $path
    source prepareFilesForFPGA.sh
fi

if [ $uploadFilesToFPGABoard_flag -eq 1 ]; then
    cd $path
    source uploadFilesToFPGABoard.sh
fi

if [ $runFilesInFPGABoard_bitstream_flag -eq 1 ] || [ $runFilesInFPGABoard_binary_flag -eq 1 ]; then
    cd $path
    source runFilesInFPGABoard.sh
fi
