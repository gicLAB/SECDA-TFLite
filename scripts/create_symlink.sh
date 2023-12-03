pushd ../tensorflow/tensorflow/lite/examples/
ln -s -f ../../../../src/secda_apps/ ./ 
popd
pushd ../tensorflow/tensorflow/lite/delegates/utils/
ln -s -f ../../../../../src/secda_delegates/ ./
ln -s -f ../../../../../src/secda_tflite/ ./
ln -s -f ../../../../../src/utils/.gitignore ./
ln -s -f ../../../../../src/utils/.clang-format ./
popd