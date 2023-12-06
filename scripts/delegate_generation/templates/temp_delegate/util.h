#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_TEMP_DELEGATE_TEMP_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_TEMP_DELEGATE_TEMP_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"
#include <cassert>

using namespace std;

inline TfLiteTensor *GetTensorAtIndex(const TfLiteContext *context,
                                      int tensor_index) {
  return &context->tensors[tensor_index];
}

inline TfLiteStatus GetMutableInputSafe(const TfLiteContext *context,
                                        int tensor_index,
                                        const TfLiteTensor **tensor) {
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

TfLiteStatus GetInputSafe(const TfLiteContext *context, int tensor_index,
                          const TfLiteTensor **tensor) {
  return GetMutableInputSafe(context, tensor_index, tensor);
}

TfLiteStatus GetOutputSafe(const TfLiteContext *context, int tensor_index,
                           TfLiteTensor **tensor) {
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

#endif