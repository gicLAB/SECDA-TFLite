#ifndef TENSORFLOW_LITE_EXAMPLES_eval_model_eval_model_H_
#define TENSORFLOW_LITE_EXAMPLES_eval_model_eval_model_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace eval_model {

struct Settings {
  bool verbose = false;
  bool accel = false;
  TfLiteType input_type = kTfLiteFloat32;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool xnnpack_delegate = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  string model_name = "./mobilenet_quant_v1_224.tflite";
  tflite::FlatBufferModel* model;
  string input_bmp_name = "./grace_hopper.bmp";
  string input_npy_name = "./tmp/grace_hopper.npy";

  string labels_file_name = "./labels.txt";
  int number_of_threads = 4;
  int number_of_results = 5;
  int max_profiling_buffer_entries = 1024;
  int number_of_warmup_runs = 2;
};

}  // namespace eval_model
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_eval_model_eval_model_H_
