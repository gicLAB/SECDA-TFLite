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

#ifndef TENSORFLOW_LITE_EXAMPLES_eval_model_BITMAP_HELPERS_H_
#define TENSORFLOW_LITE_EXAMPLES_eval_model_BITMAP_HELPERS_H_

#include "tensorflow/lite/examples/secda_apps/eval_model_accuracy/utils/bitmap_helpers_impl.h"
#include "tensorflow/lite/examples/secda_apps/eval_model_accuracy/eval_model.h"

#include <unordered_set>

namespace tflite {
namespace eval_model {

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels, Settings* s);


std::string StripTrailingSlashes(const std::string& path);

// If extension set is empty, all files will be listed. The strings in
// extension set are expected to be in lowercase and include the dot.
TfLiteStatus GetSortedFileNames(
    const std::string& directory, std::vector<std::string>* result,
    const std::unordered_set<std::string>& extensions);

inline TfLiteStatus GetSortedFileNames(const std::string& directory,
                                       std::vector<std::string>* result) {
  return GetSortedFileNames(directory, result,
                            std::unordered_set<std::string>());
}

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings* s);

// explicit instantiation
template void resize<float>(float*, unsigned char*, int, int, int, int, int,
                            int, Settings*);
template void resize<int8_t>(int8_t*, unsigned char*, int, int, int, int, int,
                             int, Settings*);
template void resize<uint8_t>(uint8_t*, unsigned char*, int, int, int, int, int,
                              int, Settings*);

}  // namespace eval_model
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_eval_model_BITMAP_HELPERS_H_
