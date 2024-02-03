#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_ADD_DELEGATE_ADD_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_ADD_DELEGATE_ADD_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"
#include <cassert>

using namespace std;

// Works only for ADD int8 // edit for other ops
struct OpData {
  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32 output_activation_min;
  int32 output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32 input1_multiplier;
  int32 input2_multiplier;
  int32 output_multiplier;
  int output_shift;
  int left_shift;
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;

  // This parameter is used to indicate whether
  // parameter scale is power of two.
  // It is used in 16-bit -> 16-bit quantization.
  bool pot_scale_int16;
};


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