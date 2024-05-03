#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_VM_SIM_DELEGATE_VM_SIM_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_VM_SIM_DELEGATE_VM_SIM_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"

// This file mostly contains generic tflite code to prepare CONV2D tensors /
// IM2COL

constexpr int kOutputShapeTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kDataInputTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kOutputTensor = 0;
const int kTensorNotAllocated = -1;

struct OpData {
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

namespace tflite {

// From im2col_utils.h End
TfLiteStatus ResizeAndTransposeWeights(TfLiteContext *context,
                                       const TfLiteTensor *weights,
                                       TfLiteTensor *transposed_weights) {
  TfLiteIntArray *transposed_weights_shape_array = TfLiteIntArrayCreate(4);
  const RuntimeShape &input_shape = GetTensorShape(weights);
  transposed_weights_shape_array->data[0] = input_shape.Dims(1);
  transposed_weights_shape_array->data[1] = input_shape.Dims(2);
  transposed_weights_shape_array->data[2] = input_shape.Dims(0);
  transposed_weights_shape_array->data[3] = input_shape.Dims(3);

  transposed_weights->type = weights->type;
  transposed_weights->allocation_type = kTfLiteDynamic;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, transposed_weights,
                                              transposed_weights_shape_array));
  // transposed_weights->allocation_type = kTfLiteArenaRw;
  // Transpose the weights from OHWI order to HWOI order.
  TransposeParams transpose_params;
  transpose_params.perm_count = 4;
  transpose_params.perm[0] = 1;
  transpose_params.perm[1] = 2;
  transpose_params.perm[2] = 0;
  transpose_params.perm[3] = 3;

  if (weights->type == kTfLiteFloat32) {
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<float>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<float>(transposed_weights));
  } else if (weights->type == kTfLiteUInt8) {
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<uint8>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<uint8>(transposed_weights));
  } else if (weights->type == kTfLiteInt8) {
    // int16 transpose_conv also with int8 weights
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<int8>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<int8>(transposed_weights));
  } else {
    TF_LITE_KERNEL_LOG(
        context,
        "Only float32, uint8, int8, int16 is supported currently, got %s.",
        TfLiteTypeGetName(weights->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

} // namespace tflite

bool IsIm2ColRequired(const TfLiteTensor *input, TfLiteConvParams *params,
                      const TfLiteTensor *filter, OpData *data,
                      bool is_hybrid) {
  // If HWCN weights are required, Im2Col not required
  // if (data->need_hwcn_weights) return false;

  // segregate based on dilated conv & non-dialated conv
  const bool need_dilated_im2col =
      params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
  const bool need_non_dilated_im2col =
      params->stride_width != 1 || params->stride_height != 1 ||
      filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

  const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

  // Return early as basic requirement is not met
  if (!need_im2col) return false;

  // Special case for Hybrid, as it supports only non-dilated im2col currently
  const bool is_hybrid_non_dilated = is_hybrid && need_non_dilated_im2col;
  const bool is_quantized =
      input->type == kTfLiteUInt8 || input->type == kTfLiteInt8;

  if (is_hybrid && !need_non_dilated_im2col) {
    return false;
  } else {
    return true;
  }
}

TfLiteStatus ResizeTensor(TfLiteContext *context,
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

TfLiteStatus ResizeTensor2(TfLiteContext *context, TfLiteTensor *shape_tensor,
                           TfLiteTensor *tensor_to_resize) {
  TfLiteIntArray *shape = TfLiteIntArrayCreate(shape_tensor->dims->size);
  for (int i = 0; i < shape->size; ++i) {
    shape->data[i] = shape_tensor->dims->data[i];
  }
  return context->ResizeTensor(context, tensor_to_resize, shape);
}

TfLiteStatus ResizeTensor3(TfLiteContext *context, TfLiteIntArray *shape,
                           TfLiteTensor *tensor_to_resize) {
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
  // col2im->allocation_type = kTfLiteArenaRw;
  return context->ResizeTensor(context, col2im, col2im_shape_array);
}

static TfLiteStatus AllocateTemporaryTensorsIfRequired(
    TfLiteContext *context, TfLiteNode *node, TfLiteTransposeConvParams *params,
    OpData *data, bool req_temp_out, int temp_out_tid, int &temp_out_id) {
  int temporaries_count = node->temporaries->size;
  if (data->col2im_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->col2im_id);
  }
  data->col2im_index = temporaries_count;
  data->has_col2im = true;
  ++temporaries_count;

  if (data->transposed_weights_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->transposed_weights_id);
  }
  data->transposed_weights_index = temporaries_count;
  data->weights_are_transposed = true;
  ++temporaries_count;

  if (data->scratch_tensor_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->scratch_tensor_id);
  }
  data->scratch_tensor_index = temporaries_count;
  ++temporaries_count;

  // If these is a need for temporary output tensor, allocate it.
  // This is true when there are multiple nodes within the delegate partition.
  // The temporary output tensor is used to store the output of the current node
  // and
  // is used as input to the next node in the delegate partition.
  if (req_temp_out) {
    temp_out_id = temporaries_count;
    if (temp_out_tid == kTensorNotAllocated) {
      context->AddTensors(context, 1, &temp_out_tid);
    }
    ++temporaries_count;
  }

  auto temp_array = TfLiteIntArrayCreate(temporaries_count);
  for (int i = 0; i < node->temporaries->size; i++)
    temp_array->data[i] = node->temporaries->data[i];

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = temp_array;

  return kTfLiteOk;
}

#endif