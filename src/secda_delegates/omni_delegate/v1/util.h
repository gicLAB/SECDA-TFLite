#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_OMNI_DELEGATE_OMNI_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_OMNI_DELEGATE_OMNI_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/fully_connected_4bit.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"
#include <cassert>
#include <iostream>

using namespace std;

// =========================================================
// OpDatas for all supported Ops
// =========================================================
const int kTensorNotAllocated = -1;
static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024; // 1GB

template <typename T>
constexpr int LUTSize() {
  static_assert(std::is_same<T, uint8_t>::value ||
                    std::is_same<T, int8_t>::value ||
                    std::is_same<T, int16_t>::value,
                "Only LUTs with uint8, int8 or int16 inputs are supported.");
  // As per c++11: constexpr methods cannot have more than one return statement.
  return (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value)
             ? 256
             : 513;
}

// ADD INT8
struct ADD_Data {
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

// CONV2D INT8
struct Conv2D_Data {
  int im2col_id = kTensorNotAllocated;
  int hwcn_weights_id = kTensorNotAllocated;
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;
  int accum_scratch_id = kTensorNotAllocated;
  int row_sums_id = kTensorNotAllocated;

  TfLitePaddingValues padding;
  int32_t output_multiplier;
  int output_shift;

  vector<int32_t> per_channel_output_multiplier;
  vector<int> per_channel_output_shift;

  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t im2col_index;
  int32_t hwcn_weights_index;
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t accum_scratch_index;
  int32_t input_offset_index;
  int32_t row_sums_index;

  bool need_hwcn_weights = false;
  bool have_weights_been_transposed = false;
  bool need_im2col = false;
  bool im2col_oversized = false;

  bool supports_multithreaded_kernel = false;
  bool is_hybrid_per_channel = false;
  bool compute_hybrid_row_sums = true;
};

// FC INT8
struct FC_Data {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
  bool compute_row_sums = false;
  // Only used for sparse hybrid fully connected kernels.
  bool ledger_initialized;
  // Used for 4bit hybrid
  std::unique_ptr<tflite::optimized_4bit::OpData4Bit> op_data_4bit = nullptr;
  TfLiteType quantized_bias_type = kTfLiteNoType;
};

// DWCONV2D_INT8
struct DWCONV2D_Data {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;

  // Hybrid per channel temporary tensors.
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t input_offset_index;
};

// TCONV_INT8
struct TCONV_Data {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int col2im_id = kTensorNotAllocated;
  int transposed_weights_id = kTensorNotAllocated;
  int scratch_tensor_id = kTensorNotAllocated;

  // col2im is the temporary tensor allocated and used in optimized path for
  // storing col2im data:gemm result for input_matrix x filter_matrix.
  int32_t col2im_index;

  // TfLiteConverter will transpose weights from HWOI to OHWI order.
  // In optimized path, we will transpose them back to HWOI, this temporary
  // tensor is allocated for storing transposed weights.
  int32_t transposed_weights_index;

  // Scratch tensor is used in the quantized path for storing accumulation
  // results.
  int32_t scratch_tensor_index;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int32_t> per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  bool has_col2im = false;
  bool weights_are_transposed = false;

  TfLiteType quantized_bias_type = kTfLiteNoType;
};

// SOFTMAX_INT8
struct SOFTMAX_Data {
  struct SoftmaxParams params();
  float table[256];
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  uint8_t uint8_table1[256];
  uint8_t uint8_table2[256];
#endif
  static constexpr int kInt16LUTArraySize = LUTSize<int16_t>();
  int16_t exp_lut[kInt16LUTArraySize]; // int16 LUT for exp(x), where x uniform
                                       // distributed between [-10.0 , 0.0]
  int16_t one_over_one_plus_x_lut[kInt16LUTArraySize]; // int16 LUT for 1 /
                                                       // (1 + x), where x
                                                       // uniform distributed
                                                       // between [0.0 , 1.0]
};

// =========================================================
// Helper functions
// =========================================================

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

TfLiteStatus ResizeTensor(TfLiteContext *context, TfLiteTensor *shape_tensor,
                          TfLiteTensor *tensor_to_resize) {
  TfLiteIntArray *shape = TfLiteIntArrayCreate(shape_tensor->dims->size);
  for (int i = 0; i < shape->size; ++i) {
    shape->data[i] = shape_tensor->dims->data[i];
  }
  return context->ResizeTensor(context, tensor_to_resize, shape);
}
TfLiteStatus ResizeTensorOutShapeTensor(TfLiteContext *context,
                                        const TfLiteTensor *shape_tensor,
                                        TfLiteTensor *tensor_to_resize) {
  // Currently only support int32 for output shape.
  if (shape_tensor->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "Output shape is %s, not int32.",
                       TfLiteTypeGetName(shape_tensor->type));
    return kTfLiteError;
  }

  TfLiteIntArray *shape =
      TfLiteIntArrayCreate(tflite::NumElements(shape_tensor));
  for (int i = 0; i < shape->size; ++i) {
    shape->data[i] = shape_tensor->data.i32[i];
  }

  return context->ResizeTensor(context, tensor_to_resize, shape);
}

TfLiteStatus ResizeCol2ImTensor(TfLiteContext *context,
                                const TfLiteTensor *output_shape,
                                const TfLiteTensor *weights,
                                const TfLiteTensor *input,
                                TfLiteTensor *col2im) {
  if (output_shape->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "col2im shape is %s, not int32.",
                       TfLiteTypeGetName(output_shape->type));
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, tflite::NumElements(output_shape), 4);
  TfLiteIntArray *col2im_shape_array = TfLiteIntArrayCreate(2);
  const tflite::RuntimeShape &input_shape = tflite::GetTensorShape(input);
  const tflite::RuntimeShape &weights_shape = tflite::GetTensorShape(weights);
  col2im_shape_array->data[0] = input_shape.Dims(1) * input_shape.Dims(2);
  col2im_shape_array->data[1] =
      weights_shape.Dims(0) * weights_shape.Dims(1) * weights_shape.Dims(2);

  col2im->type = input->type == kTfLiteFloat32 ? kTfLiteFloat32 : kTfLiteInt32;
  col2im->allocation_type = kTfLiteDynamic;
  return context->ResizeTensor(context, col2im, col2im_shape_array);
}

// =========================================================
// IsNodeSupportedByDelegate
// =========================================================

bool IsNode_ADD_INT8(const TfLiteRegistration *registration,
                     const TfLiteNode *node, TfLiteContext *context) {
  // Only supports Add ops
  if (kTfLiteBuiltinAdd != registration->builtin_code) return false;

  if (node->inputs->size != 2) return false;

  // This delegate only supports int8 types.
  for (int i = 0; i < 2; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  TfLiteTensor input1 = context->tensors[node->inputs->data[0]];
  TfLiteTensor input2 = context->tensors[node->inputs->data[1]];

  if (!TfLiteIntArrayEqual(input1.dims, input2.dims)) return false;

  return true;
}

bool IsNode_TCONV_INT8(const TfLiteRegistration *registration,
                       const TfLiteNode *node, TfLiteContext *context) {
  // Only supports TCONV ops
  if (kTfLiteBuiltinTransposeConv != registration->builtin_code) return false;

  // This delegate requires at least 3 inputs.
  // Input, Weight,  Output shape tensor and maybe Bias.
  if (node->inputs->size < 3) return false;

  // This delegate only supports int8 types.
  for (int i = 1; i < 3; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  // Ensures output shape tensor is supports int32 type
  auto &tensor = context->tensors[node->inputs->data[0]];
  if (tensor.type != kTfLiteInt32) return false;

  if (node->inputs->size == 4) {
    // Ensures bias tensor is supports int32 type
    auto &tensor2 = context->tensors[node->inputs->data[3]];
    if (tensor2.type != kTfLiteInt32) return false;
  }

  return true;
}

bool IsNode_FC_INT8(const TfLiteRegistration *registration,
                    const TfLiteNode *node, TfLiteContext *context) {
  // Only supports FC ops
  if (kTfLiteBuiltinFullyConnected != registration->builtin_code) return false;

  if (node->inputs->size != 3 && node->inputs->size != 2) return false;
  // This delegate only supports int8 types.
  for (int i = 0; i < 2; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
    auto &tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32 && tensor.type <= 16) return false;
  }

  return true;
}

bool IsNode_DWCONV2D_INT8(const TfLiteRegistration *registration,
                          const TfLiteNode *node, TfLiteContext *context) {
  // Only supports FC ops
  if (kTfLiteBuiltinDepthwiseConv2d != registration->builtin_code) return false;

  if (node->inputs->size != 3 && node->inputs->size != 2) return false;
  // This delegate only supports int8 types.
  for (int i = 0; i < 2; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
    auto &tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32 && tensor.type <= 16) return false;
  }

  return true;
}

bool IsNode_CONV2D_INT8(const TfLiteRegistration *registration,
                        const TfLiteNode *node, TfLiteContext *context) {
  // Only supports CONV2D ops
  if (kTfLiteBuiltinConv2d != registration->builtin_code) return false;

  // This delegate only supports int8 types.
  if (node->inputs->size != 3 && node->inputs->size != 2) return false;
  for (int i = 0; i < 2; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  if (node->inputs->size == 3) {
    auto &tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32) return false;
  }

  return true;
}

bool IsNode_SHAPE_INT8(const TfLiteRegistration *registration,
                       const TfLiteNode *node, TfLiteContext *context) {
  // Only supports SHAPE ops
  if (kTfLiteBuiltinShape != registration->builtin_code) return false;

  // This delegate only supports int8 types.
  if (node->inputs->size != 1) return false;
  for (int i = 0; i < 1; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  // Verify the output tensor type is int32
  if (node->outputs->size != 1) return false;
  auto &tensor = context->tensors[node->outputs->data[0]];
  if (tensor.type != kTfLiteInt32) return false;

  return true;
}

bool IsNode_SOFTMAX_INT8(const TfLiteRegistration *registration,
                         const TfLiteNode *node, TfLiteContext *context) {
  // Only supports SOFTMAX ops
  if (kTfLiteBuiltinSoftmax != registration->builtin_code) return false;

  // This delegate only supports int8 types.
  if (node->inputs->size != 1) return false;

  auto &itensor = context->tensors[node->inputs->data[0]];
  if (itensor.type != kTfLiteInt8) return false;

  // Verify the output tensor type is int32
  if (node->outputs->size != 1) return false;
  auto &otensor = context->tensors[node->outputs->data[0]];
  if (otensor.type != kTfLiteInt8) return false;

  if (!TfLiteIntArrayEqual(itensor.dims, otensor.dims)) return false;

  return true;
}

#endif

// #include <fstream>
// #include <iostream>
// int counter = 0;
// using namespace std;
// ofstream file;
// file.open("aData/???/" + std::to_string(counter++) + "_out_cpu.csv");
// file.open("aData/???/" + std::to_string(counter2++) +
//                   "_out_acc.csv");