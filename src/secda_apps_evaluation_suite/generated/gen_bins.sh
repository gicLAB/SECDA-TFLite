#!/bin/bash
set -e
pushd /home/rappy/workspace/secda-tflite-v1.2-rppv2/SECDA-TFLite/tensorflow
bazel6 build --config=elinux_aarch64 -c opt //tensorflow/lite/delegates/utils/secda_delegates/vm_shift_delegate_rpp/v12:eval_model_accuracy_plus_vm_shift_delegate_rpp --copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --copt='-DKRIA' 
rsync -r -avz -e 'ssh -p 2222' /home/rappy/workspace/secda-tflite-v1.2-rppv2/SECDA-TFLite/tensorflow/bazel-out/aarch64-opt/bin/tensorflow/lite/delegates/utils/secda_delegates/vm_shift_delegate_rpp/v12/eval_model_accuracy_plus_vm_shift_delegate_rpp ubuntu@kriarpp:/home/ubuntu/Workspace/secda_apps_evaluation_suite/bins/ema_vm_shift_delegate_rpp_12
ssh -t -p 2222 ubuntu@kriarpp 'cd /home/ubuntu/Workspace/secda_apps_evaluation_suite/bins/ && chmod 775 ./*'
popd
