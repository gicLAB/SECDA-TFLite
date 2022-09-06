/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_TOY_DELEGATE_TOY_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_TOY_DELEGATE_TOY_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} ToyDelegateOptions;

// Returns a structure with the default delegate options.
ToyDelegateOptions TfLiteToyDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteToyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteToyDelegateCreate(const ToyDelegateOptions* options);

// Destroys a delegate created with `TfLiteToyDelegateCreate` call.
void TfLiteToyDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteToyDelegateCreateUnique(const ToyDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteToyDelegateCreate(options), TfLiteToyDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_TOY_DELEGATE_TOY_DELEGATE_H_
