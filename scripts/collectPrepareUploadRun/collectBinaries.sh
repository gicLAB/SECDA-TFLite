# this file will create the binaries and copy them to the result folder

copyBinaryTo="$result_path/$trial_name"

#create a binary name from three variables
binaryName="bin_"$app_Name"_"$del_name"_"$trial_name""

cd $SecdaTFLitePath

bazel build --config=elinux_armhf -c opt $bazel_build_path --copt="-DACC_PROFILE" --copt="-DTFLITE_ENABLE_XNNPACK=OFF" --copt="-DTFLITE_WITHOUT_XNNPACK" --copt="-DACC_NEON"

echo "removing binary to $copyBinaryTo/$binaryName !!!"
rm -f "$copyBinaryTo/$binaryName"

echo "copying binary to $copyBinaryTo/$binaryName !!!"
cp $binaries_path "$copyBinaryTo/$binaryName"
