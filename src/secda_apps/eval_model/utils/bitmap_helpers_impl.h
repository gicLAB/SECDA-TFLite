/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXAMPLES_eval_model_BITMAP_HELPERS_IMPL_H_
#define TENSORFLOW_LITE_EXAMPLES_eval_model_BITMAP_HELPERS_IMPL_H_

#include "tensorflow/lite/examples/secda_apps/eval_model/eval_model.h"

#include<fstream>
#include<iomanip>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace eval_model {

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings* s) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

  ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  // original
  // float inp_mean[3] = {0.4914f, 0.4822f, 0.4465f};
  // float inp_std[3] = {0.2023f, 0.1994f, 0.2010f};
  // reversed
  // float inp_mean[3] = {0.4465f, 0.4822f, 0.4914f};
  // float inp_std[3] = {0.2010f, 0.1994f, 0.2023f};

  // float inp_mean[3] = {0.4465f, 0.4822f, 0.4914f};
  // float inp_std[3] = {0.2616f, 0.2435f, 0.2470f};
  
  float inp_mean[3] = {0.44653124f, 0.48215827f, 0.49139968f};
  float inp_std[3] = {0.26158768f, 0.24348505f, 0.24703233f};


  // save output tensor in a csv file
  // std::ofstream input_tensor_stg2;
  // input_tensor_stg2.open("/home/rppv15/workspace/Quantization/data/cifar10/input_tensor_stg2_SECDA.csv");
  // for (int i = 0; i < output_number_of_pixels; i++) {
  //   input_tensor_stg2 << output[i] << std::endl;
  // }
  // input_tensor_stg2.close();

  for (int i = 0; i < output_number_of_pixels; i+=3) {
    switch (s->input_type) {
      case kTfLiteFloat32:
        //MNIST Norm//
        out[i] = (output[i]) / 255.0f;
        out[i+1] = (output[i+1]) / 255.0f;
        out[i+2] = (output[i+2]) / 255.0f;
        
        //CIFAR 10 Norm//
        // out[i] = (output[i]) / 255.0f;
        // out[i+1] = (output[i+1]) / 255.0f;
        // out[i+2] = (output[i+2]) / 255.0f;

        // out[i] = (out[i] - inp_mean[0]) / inp_std[0];
        // out[i+1] = (out[i+1] - inp_mean[1]) / inp_std[1];
        // out[i+2] = (out[i+2] - inp_mean[2]) / inp_std[2];
        
        // out[i] = (output[i] - s->input_mean) / s->input_std;
        break;
      case kTfLiteInt8:
        out[i] = static_cast<int8_t>(output[i] - 128);
        break;
      case kTfLiteUInt8:
        out[i] = static_cast<uint8_t>(output[i]);
        break;
      default:
        break;
    }
  }

  // save output tensor in a csv file
  // std::ofstream input_tensor_stg3;
  // input_tensor_stg3.open("/home/rppv15/workspace/Quantization/data/cifar10/input_tensor_stg3_SECDA.csv");
  // for (int i = 0; i < output_number_of_pixels; i++) {
  //   input_tensor_stg3 << std::fixed << std::setprecision(6) << out[i] << std::endl;
  // }
  // input_tensor_stg3.close();
}

}  // namespace eval_model
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_eval_model_BITMAP_HELPERS_IMPL_H_