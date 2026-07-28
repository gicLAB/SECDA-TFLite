#!/bin/bash

helpFunction() {
  echo ""
  echo "Usage: $0 -j aec_path -i -b -c -l -p -n name"
  echo -e "\t-j apps evaluation config.json file"
  echo -e "\t-i Initialize the board"
  echo -e "\t-b Generate binaries"
  echo -e "\t-c Copy BitStreams to Boards"
  echo -e "\t-l Do not Load BitStreams to FPGA"
  echo -e "\t-p Power collection"
  echo -e "\t-n Name of the experiment"
  exit 1 # Exit script after printing help
}

# Optional arguments
# bin_gen: Generate binaries
aec_path="" ## apps evaluation config.json file , give the default path
init=0
name="exp_0"
bin_gen=0
cpy_bst=0
ld_bst=1
collect_power=0
now=$(date +"%Y_%m_%d_%H_%M")

while getopts "hj:in:bclp" flag; do
  case $flag in
    h)
      helpFunction
      exit
      ;;
    j) aec_path=$OPTARG ;;
    i) init=1 ;;
    n) name=$OPTARG ;;
    b) bin_gen=1 ;;
    c) cpy_bst=1 ;;
    l) ld_bst=0 ;;
    p) collect_power=1 ;;
    :)
      echo "Missing argument for option -$OPTARG"
      exit 1
      ;;
    \?)
      helpFunction
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

sc_path="../../config.json"
if [ "$aec_path" == "" ]; then
  aec_path="configs/default_exp.json"
fi


trap ctrl_c INT
echo "-----------------------------------------------------------"
echo "-- SECDA-TFLite Evaluation Suite --"
echo "-----------------------------------------------------------"
echo "  Name: ${name}"
echo "  Time: ${now}"
echo "-----------------------------------------------------------"

# create a directory in the host in the path of the script
mkdir -p ./results
# create subdirectory in the name of ${name}
if [ -d "./results/${name}" ]; then
  rm -rf "./results/${name}"
fi
mkdir -p "./results/${name}"
#copy the aec_path file to the results directory
cp "$aec_path" "./results/${name}/"


out_dir=$(jq -r '.out_dir' ${sc_path})

# process the multiple configurations file to generate the 
# files in sc "out_dir" location
python3 scripts/process_flags_n_config.py $sc_path $aec_path $init $bin_gen $cpy_bst $ld_bst $collect_power

#copy generated config files to ./results
cp -r ${out_dir} "./results/${name}/"

# python3 scripts/run_exps_n_process_result.py $sc_path $collect_power

# #copy generated config files to ./results
# cp -r ${out_dir} "./results/${name}/"

echo "-----------------------------------------------------------"
echo "Exiting SECDA-TFLite Evaluation Suite"
echo "-----------------------------------------------------------"