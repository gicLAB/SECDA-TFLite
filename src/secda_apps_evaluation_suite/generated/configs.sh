declare -a hw_array=(
  "VMRPP_KRIA_SH_APOT_OPT_GEMM8_250M" 
)
declare -a tag_array=(
  "eval_model_accuracy_VMRPP_KRIA_SH_APOT_OPT_GEMM8_250M_vm_shift_delegate_rpp_12_model_best_apot_dq_int8_int8_1000.tflite_4_model_output_labels_10.txt_testX_000010.bmp__groundTruth_testData_labels.txt__10000" 
)
declare -a app_array=(
  "eval_model_accuracy" 
)
declare -a model_array=(
  "model_best_apot_dq_int8_int8_1000.tflite" 
)
declare -a cmd_array=(
  "/home/ubuntu/Workspace/secda_apps_evaluation_suite/bins/ema_vm_shift_delegate_rpp_12  --tflite_model=/home/ubuntu/Workspace/data/cifar10/models/model_best_apot_dq_int8_int8_1000.tflite --threads=4 --labels=/home/ubuntu/Workspace/data/cifar10/labels/model_output_labels_10.txt --image=/home/ubuntu/Workspace/data/cifar10/testData/testX_000010.bmp --test_dataset_location=/home/ubuntu/Workspace/data/cifar10/testData/ --ground_truth_labels_file_name=/home/ubuntu/Workspace/data/cifar10/labels/groundTruth_testData_labels.txt --output_file_name= --no_of_images=10000 --use_vm_shift_delegate_rpp=true" 
)
declare -a del_array=(
  "vm_shift_delegate_rpp" 
)
declare -a del_version_array=(
  "12" 
)
declare -a version_array=(
  "12_4" 
)
