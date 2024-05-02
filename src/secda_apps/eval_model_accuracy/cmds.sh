#!/bin/bash
set -e
pushd /home/jude/Workspace/SECDA-TFLite_v1.2/tensorflow

bazel6 build --config=elinux_armhf -c opt //tensorflow/lite/examples/secda_apps/eval_model_accuracy:eval_model_accuracy --cxxopt='-mfpu=neon' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' 
rsync -r -avz -e 'ssh -p 2202' /home/jude/Workspace/SECDA-TFLite_v1.2/tensorflow/bazel-out/armhf-opt/bin/tensorflow/lite/examples/secda_apps/eval_model_accuracy/eval_model_accuracy xilinx@jharis.ddns.net:/home/xilinx/Workspace/test/
ssh -t -p 2202 xilinx@jharis.ddns.net 'cd /home/xilinx/Workspace/test/ && chmod 775 ./eval* && sudo ./eval_model_accuracy -mmodel.tflite -ltmp/labels_cifar10.txt -itmp/testX_0_cifar10.bmp -dtmp/cifar10_test -htmp/cifar10_test_labels_names.txt -oaccuracy -u 10000 -t 2'

popd
