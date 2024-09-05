

#include "tensorflow/lite/examples/secda_apps/eval_model_accuracy/eval_model.h"

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "./utils/npy.hpp"
#include "absl/memory/memory.h"
#include "tensorflow/lite/examples/secda_apps/eval_model_accuracy/utils/bitmap_helpers.h"
#include "tensorflow/lite/examples/secda_apps/eval_model_accuracy/utils/get_top_n.h"
#include "tensorflow/lite/examples/secda_apps/eval_model_accuracy/utils/log.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include <fstream> // Include the necessary header file

namespace tflite {
namespace eval_model {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

// Copied from label_image.cc
class DelegateProviders {
public:
  DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int *argc, const char **argv) {
    std::vector<tflite::Flag> flags;
    delegate_list_util_.AppendCmdlineFlags(flags);

    const bool parse_result = Flags::Parse(argc, argv, flags);
    if (!parse_result) {
      std::string usage = Flags::Usage(argv[0], flags);
      LOG(ERROR) << usage;
    }
    return parse_result;
  }

  // According to passed-in settings `s`, this function sets corresponding
  // parameters that are defined by various delegate execution providers. See
  // lite/tools/delegates/README.md for the full list of parameters defined.
  void MergeSettingsIntoParams(const Settings &s) {
    // Parse settings related to GPU delegate.
    // Note that GPU delegate does support OpenCL. 'gl_backend' was introduced
    // when the GPU delegate only supports OpenGL. Therefore, we consider
    // setting 'gl_backend' to true means using the GPU delegate.
    if (s.gl_backend) {
      if (!params_.HasParam("use_gpu")) {
        LOG(WARN) << "GPU deleate execution provider isn't linked or GPU "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_gpu", true);
        // The parameter "gpu_inference_for_sustained_speed" isn't available for
        // iOS devices.
        if (params_.HasParam("gpu_inference_for_sustained_speed")) {
          params_.Set<bool>("gpu_inference_for_sustained_speed", true);
        }
        params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
      }
    }

    // Parse settings related to NNAPI delegate.
    if (s.accel) {
      if (!params_.HasParam("use_nnapi")) {
        LOG(WARN) << "NNAPI deleate execution provider isn't linked or NNAPI "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
      }
    }

    // Parse settings related to Hexagon delegate.
    if (s.hexagon_delegate) {
      if (!params_.HasParam("use_hexagon")) {
        LOG(WARN) << "Hexagon deleate execution provider isn't linked or "
                     "Hexagon delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_hexagon", true);
        params_.Set<bool>("hexagon_profiling", s.profiling);
      }
    }

    // Parse settings related to XNNPACK delegate.
    if (s.xnnpack_delegate) {
      if (!params_.HasParam("use_xnnpack")) {
        LOG(WARN) << "XNNPACK deleate execution provider isn't linked or "
                     "XNNPACK delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<bool>("num_threads", s.number_of_threads);
      }
    }
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate>
  CreateAllDelegates() const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;
};

// Copied from label_image.cc
// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string &file_name,
                            std::vector<string> *result,
                            size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(ERROR) << "Labels file " << file_name << " not found";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

// Copied from label_image.cc
void PrintProfilingInfo(const profiling::ProfileEvent *e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symbolic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->elapsed_time) / 1000.0 << ", Subgraph " << std::setw(3)
            << std::setprecision(3) << subgraph_index << ", Node "
            << std::setw(3) << std::setprecision(3) << op_index << ", OpCode "
            << std::setw(3) << std::setprecision(3) << registration.builtin_code
            << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code));
}

void ResizeBMPInput(Settings *settings, tflite::Interpreter *interpreter,
                    int input, int image_width, int image_height,
                    int image_channels) {
  int wanted_height = interpreter->tensor(input)->dims->data[1];
  int wanted_width = interpreter->tensor(input)->dims->data[2];
  int wanted_channels = interpreter->tensor(input)->dims->data[3];
  
  std::vector<uint8_t> in = read_bmp(settings->input_bmp_name, &image_width,
                                     &image_height, &image_channels, settings);

  // change the pixel rgb to bgr
  for (int i = 0; i < image_width * image_height; i++) {
    uint8_t temp = in[i * 3];
    in[i * 3] = in[i * 3 + 2];
    in[i * 3 + 2] = temp;
  }

  //save in a csv file
  // std::ofstream input_tensor_stg1;
  // input_tensor_stg1.open("/home/rppv15/workspace/SECDA-TFLite/tensorflow/aData/csvfiles/target_image_raw_secda.csv");
  // for (const auto& value : in) {
  //     input_tensor_stg1 << static_cast<int>(value) << std::endl;
  // }
  // input_tensor_stg1.close();

  switch (settings->input_type) {
  case kTfLiteFloat32:
    resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                  image_height, image_width, image_channels, wanted_height,
                  wanted_width, wanted_channels, settings);
    break;
  case kTfLiteInt8:
    resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                   image_height, image_width, image_channels, wanted_height,
                   wanted_width, wanted_channels, settings);
    break;
  case kTfLiteUInt8:
    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, settings);
    break;
  default:
    LOG(ERROR) << "cannot handle input type "
               << interpreter->tensor(input)->type << " yet";
    exit(-1);
  }
}

void processOutput(tflite::Interpreter *interpreter, int output, float threshold,
                   int number_of_results, string ground_truth_label, Settings *settings) {
  std::vector<std::pair<float, int>> top_results;
  TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
  auto output_size = output_dims->data[output_dims->size - 1];

  switch (interpreter->tensor(output)->type) {
  case kTfLiteFloat32:
    get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                     number_of_results, threshold, &top_results,
                     settings->input_type);
    break;
  case kTfLiteInt8:
    get_top_n<int8_t>(interpreter->typed_output_tensor<int8_t>(0), output_size,
                      number_of_results, threshold, &top_results,
                      settings->input_type);
    break;
  case kTfLiteUInt8:
    get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                       output_size, number_of_results, threshold, &top_results,
                       settings->input_type);
    break;
  default:
    LOG(ERROR) << "cannot handle output type "
               << interpreter->tensor(output)->type << " yet";
    exit(-1);
  }
  std::vector<string> labels;
  size_t label_count;
  if (ReadLabelsFile(settings->labels_file_name, &labels, &label_count) !=
      kTfLiteOk)
    exit(-1);
  
  // outputFile << "Predicted lables" << std::endl;
  static int top_1_count = 0;
  static int top_5_count = 0;
  static int no_of_images = 0;
  int loop_count = 0;
  for (const auto &result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    if (ground_truth_label == labels[index]) {
        top_5_count++;
    }
    
    if (loop_count == 0 && ground_truth_label == labels[index]) {
      top_1_count++;
    }
    if (loop_count == 0)
    {
      static int ccd = 0;
      LOG(INFO) << "No: " << ccd << " ground_truth_label: " << ground_truth_label //
      << " Pred "<< confidence << ": " << index << " " << labels[index];
      ccd++;
    }
    // if (loop_count==0)
    //   LOG(INFO) << confidence << ": " << index << " " << labels[index];
    // outputFile << labels[index] << " ";
    loop_count++;
  }
  if (no_of_images == settings->no_of_images - 1) {
    LOG(INFO) << "Top 1 Accuracy: " << (float)top_1_count *100 / settings->no_of_images;
    LOG(INFO) << "Top 5 Accuracy: " << (float)top_5_count *100 / settings->no_of_images;
    std::ofstream outputFile(settings->output_file_name,std::ios::app);
    outputFile << "Top 1 Accuracy: " << (float)top_1_count *100 / settings->no_of_images << std::endl;
    outputFile << "Top 5 Accuracy: " << (float)top_5_count *100 / settings->no_of_images << std::endl;
    outputFile.close(); // Close the output file
  }
  else no_of_images++;
}

void Load_Data_NPY(std::unique_ptr<tflite::Interpreter> &interpreter,
                   std::string npy_file_path) {
  const std::vector<int> &t_inputs = interpreter->inputs();
  TfLiteTensor *tensor = interpreter->tensor(t_inputs[0]);
  int input_size = tensor->dims->size;
  int input_len = 1;
  for (int i = 0; i < input_size; i++)
    input_len *= tensor->dims->data[i];

  std::vector<unsigned long> shape;
  bool fortran_order;
  shape.clear();
  if (tensor->type == 9) {
    std::vector<char> indata;
    indata.clear();
    npy::LoadArrayFromNumpy(npy_file_path, shape, fortran_order, indata);
    auto in_data = tensor->data.int8;
    for (int i = 0; i < input_len; i++)
      in_data[i] = (int8_t)indata[i];
  } else if (tensor->type == 3) {
    std::vector<float> indata;
    indata.clear();
    std::cout << "UINT8  Loaded" << std::endl;
    npy::LoadArrayFromNumpy(npy_file_path, shape, fortran_order, indata);
    auto in_data = tensor->data.uint8;
    int data_len = shape[0] * shape[1] * shape[2] * shape[3];
    input_len = data_len > input_len ? input_len : data_len;
    for (int i = 0; i < input_len; i++)
      in_data[i] = (uint8_t)indata[i];
  } else {
    std::vector<float> indata;
    indata.clear();
    std::cout << "FLOAT  Loaded" << std::endl;
    npy::LoadArrayFromNumpy(npy_file_path, shape, fortran_order, indata);
    auto in_data = tensor->data.f;
    int data_len = shape[0] * shape[1] * shape[2] * shape[3];
    input_len = data_len > input_len ? input_len : data_len;
    for (int i = 0; i < input_len; i++)
      in_data[i] = indata[i];
  }
  std::cout << "Input  Loaded" << std::endl;
}

void LoadModel(Settings *settings,
               std::unique_ptr<tflite::FlatBufferModel> &model,
               std::unique_ptr<tflite::Interpreter> &interpreter) {
  model = tflite::FlatBufferModel::BuildFromFile(settings->model_name.c_str());
  if (!model) {
    LOG(ERROR) << "Failed to mmap model " << settings->model_name;
    exit(-1);
  }
  settings->model = model.get();
  LOG(INFO) << "Loaded model " << settings->model_name;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(ERROR) << "Failed to construct interpreter";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(settings->allow_fp16);

  if (settings->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size();
    LOG(INFO) << "nodes size: " << interpreter->nodes_size();
    LOG(INFO) << "inputs: " << interpreter->inputs().size();
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0);

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point;
    }
  }

  if (settings->number_of_threads != -1) {
    interpreter->SetNumThreads(settings->number_of_threads);
  }
}


std::string removeCarriageReturn(const std::string& str) {
    std::string result;
    for (char c : str) {
        if (c != '\r') {
            result += c;
        }
    }
    return result;
}

void RunInference(Settings *settings,
                  const DelegateProviders &delegate_providers) {
  if (!settings->model_name.c_str()) {
    LOG(ERROR) << "no model file name";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  LoadModel(settings, model, interpreter);

  // ======================================================================

  int input = interpreter->inputs()[0];
  if (settings->verbose)
    LOG(INFO) << "input: " << input;

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (settings->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size();
    LOG(INFO) << "number of outputs: " << outputs.size();
  }

  auto delegates = delegate_providers.CreateAllDelegates();
  for (auto &delegate : delegates) {
    const auto delegate_name = delegate.provider->GetName();
    if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) !=
        kTfLiteOk) {
      LOG(ERROR) << "Failed to apply " << delegate_name << " delegate.";
      exit(-1);
    } else {
      LOG(INFO) << "Applied " << delegate_name << " delegate.";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!";
    exit(-1);
  }

  if (settings->verbose)
    PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray *dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  settings->input_type = interpreter->tensor(input)->type;
  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  // int image_width = 28;
  // int image_height = 28;
  // int image_channels = 1;

  // check the test dataset location and take first settings->no_of_images images location in an array
  // read the image from the location and resize it to 224x224x3
  std::vector<std::string> image_files, ground_truth_image_labels;
  if (GetSortedFileNames(StripTrailingSlashes(settings->test_dataset_location),
                        &image_files) != kTfLiteOk) {
    LOG(ERROR) << "setting->test_dataset_location: " << settings->test_dataset_location;  
    LOG(ERROR) << "Could not read ground truth image folder location";
    exit(-1);
  }

  size_t ground_truth_image_label_count;
  if (ReadLabelsFile(settings->ground_truth_labels_file_name, &ground_truth_image_labels, &ground_truth_image_label_count) !=
      kTfLiteOk)
  {
    LOG(ERROR) << "Could not read ground truth image labels file location";
    exit(-1);
  }

  // LOG(INFO) << "ground_truth_image_label_count: " << ground_truth_image_label_count;

  // clear settings->output_file_name file before writing to it
  std::ofstream outputFile(settings->output_file_name);
  outputFile.close();

  const int step = settings->no_of_images / 100;
  
  struct timeval start_time, stop_time;
  LOG(INFO) << "Starting Evaluation: ";
  gettimeofday(&start_time, nullptr);
  
  for (int i=0; i<settings->no_of_images; i++) {
    if (step > 1 && i % step == 0) {
      LOG(INFO) << "Evaluated: " << i / step << "%";
    }

    settings->input_bmp_name = image_files[i];
    // LOG(INFO) << "settings->input_bmp_name: " << settings->input_bmp_name;
    ResizeBMPInput(settings, interpreter.get(), input, image_width, image_height,
                 image_channels);
    //  Manual Inputs v2
    // Load_Data_NPY(interpreter, settings->input_npy_name);

    // Manual Start
    // std::cout << "Press Enter to Go";
    // std::cin.ignore();

    auto profiler = absl::make_unique<profiling::Profiler>(
        settings->max_profiling_buffer_entries);
    interpreter->SetProfiler(profiler.get());
    if (settings->profiling)
      profiler->StartProfiling();

    // const float* preproc_out = interpreter->typed_tensor<float>(0);
    // // save output tensor in a csv file
    // std::ofstream input_tensor_stg4;
    // input_tensor_stg4.open("/home/rppv15/workspace/SECDA-TFLite/tensorflow/aData/csvfiles/target_image_normalized_preProcOut_secda.csv");
    // for (int i = 0; i < (32*32*3); i++) {
    //   input_tensor_stg4 << std::fixed << std::setprecision(6) << preproc_out[i] << std::endl;
    // }
    // input_tensor_stg4.close();

    for (int i = 0; i < settings->loop_count; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(ERROR) << "Failed to invoke tflite!";
        exit(-1);
      }
    }

    if (settings->profiling) {
      profiler->StopProfiling();
      auto profile_events = profiler->GetProfileEvents();
      for (int i = 0; i < profile_events.size(); i++) {
        auto subgraph_index = profile_events[i]->extra_event_metadata;
        auto op_index = profile_events[i]->event_metadata;
        const auto subgraph = interpreter->subgraph(subgraph_index);
        const auto node_and_registration =
            subgraph->node_and_registration(op_index);
        const TfLiteRegistration registration = node_and_registration->second;
        PrintProfilingInfo(profile_events[i], subgraph_index, op_index,
                          registration);
      }
    }

    const float threshold = 0.000001f;
    std::vector<std::pair<float, int>> top_results;
    int output = interpreter->outputs()[0];
    string ground_truth_label = removeCarriageReturn(ground_truth_image_labels[i]);

    processOutput(interpreter.get(), output, threshold,
                  settings->number_of_results,ground_truth_label, settings);
  }
  gettimeofday(&stop_time, nullptr);
  std::ofstream outputFile1(settings->output_file_name,std::ios::app);
  LOG(INFO) << "average time per image: "
            << (get_us(stop_time) - get_us(start_time)) /
                  (settings->no_of_images * 1000)
            << " ms";
  LOG(INFO) << "Total time: "
            << (get_us(stop_time) - get_us(start_time)) / 1000000
            << " s";
  outputFile1 << "average time per image: "
            << (get_us(stop_time) - get_us(start_time)) /
                  (settings->no_of_images * 1000)
            << " ms" << std::endl;
  outputFile1 << "Total time: "
            << (get_us(stop_time) - get_us(start_time)) / 1000000
            << " s" << std::endl; 
  outputFile1.close(); // Close the output file

}

void display_usage() {
  LOG(INFO)
      << "eval_model_accuracy\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--gl_backend, -g: [0|1]: use GL GPU Delegate on Android\n"
      << "--hexagon_delegate, -j: [0|1]: use Hexagon Delegate on Android\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--npy, -n: image_name.npy\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--xnnpack_delegate, -x [0:1]: xnnpack delegate\n"
      << "--test_dataset_location, -d: location of the test dataset\n"
      << "--ground_truth_labels_file_name, -h: ground truth labels file name\n"
      << "--output_file_name, -o: output file name\n";

}

// Copied from label_image.cc
int Main(int argc, char **argv) {
  DelegateProviders delegate_providers;
  bool parse_result = delegate_providers.InitFromCmdlineArgs(
      &argc, const_cast<const char **>(argv));
  if (!parse_result) {
    return EXIT_FAILURE;
  }

  Settings s;

  int c;
  // Parse command line arguments and set corresponding parameters.
  while (true) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"npy", required_argument, nullptr, 'n'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {"hexagon_delegate", required_argument, nullptr, 'j'},
        {"xnnpack_delegate", required_argument, nullptr, 'x'},
        {"test_dataset_location", required_argument, nullptr, 'd'},
        {"ground_truth_labels_file_name", required_argument, nullptr, 'h'},
        {"output_file_name", required_argument, nullptr, 'o'},
        {"no_of_images", required_argument, nullptr, 'u'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:d:e:f:g:i:j:l:m:n:p:r:s:t:v:w:x:d:h:o:u:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 'a':
      s.accel = strtol(optarg, nullptr, 10);
      break;
    case 'b':
      s.input_mean = strtod(optarg, nullptr);
      break;
    case 'c':
      s.loop_count = strtol(optarg, nullptr, 10);
      break;
    case 'e':
      s.max_profiling_buffer_entries = strtol(optarg, nullptr, 10);
      break;
    case 'f':
      s.allow_fp16 = strtol(optarg, nullptr, 10);
      break;
    case 'g':
      s.gl_backend = strtol(optarg, nullptr, 10);
      break;
    case 'i':
      s.input_bmp_name = optarg;
      break;
    case 'j':
      s.hexagon_delegate = optarg;
      break;
    case 'l':
      s.labels_file_name = optarg;
      break;
    case 'm':
      s.model_name = optarg;
      break;
    case 'n':
      s.input_npy_name = optarg;
      break;
    case 'p':
      s.profiling = strtol(optarg, nullptr, 10);
      break;
    case 'r':
      s.number_of_results = strtol(optarg, nullptr, 10);
      break;
    case 's':
      s.input_std = strtod(optarg, nullptr);
      break;
    case 't':
      s.number_of_threads = strtol(optarg, nullptr, 10);
      break;
    case 'v':
      s.verbose = strtol(optarg, nullptr, 10);
      break;
    case 'w':
      s.number_of_warmup_runs = strtol(optarg, nullptr, 10);
      break;
    case 'x':
      s.xnnpack_delegate = strtol(optarg, nullptr, 10);
      break;
    case 'd':
      s.test_dataset_location = optarg; 
      break;
    case 'h':
      s.ground_truth_labels_file_name = optarg;
      break;
    case 'o':
      s.output_file_name = optarg;
      break;
    case 'u':
      s.no_of_images = strtol(optarg, nullptr, 10);
      break;
    case '?':
      /* getopt_long already printed an error message. */
      display_usage();
      exit(-1);
    default:
      exit(-1);
    }
  }
  // LOG(INFO) << "Settings: ";
  // LOG(INFO) << "Labels file name: " << s.labels_file_name;
  // LOG(INFO) << "test_dataset_location: " << s.test_dataset_location;
  // LOG(INFO) << "ground_truth_labels_file_name: " << s.ground_truth_labels_file_name;
  // LOG(INFO) << "output_file_name: " << s.output_file_name;
  // LOG(INFO) << "no_of_images: " << s.no_of_images;

  // exit(-1);
  delegate_providers.MergeSettingsIntoParams(s);
  RunInference(&s, delegate_providers);
  return 0;
}

} // namespace eval_model
} // namespace tflite

int main(int argc, char **argv) { return tflite::eval_model::Main(argc, argv); }