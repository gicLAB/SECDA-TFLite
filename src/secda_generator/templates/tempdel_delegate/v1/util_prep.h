#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_PREP_TEMPDEL_DELEGATE_TEMPDEL_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_PREP_TEMPDEL_DELEGATE_TEMPDEL_DELEGATE_UTIL_H_

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "util.h"
using namespace std;

#define PadKernelMaxDimensionCount 5
const int kMaxConstantOutputTensorSize = 8;

// =========================================================
// Layer Specific Structs
// =========================================================
struct PadContext {
  PadContext(TfLiteContext *context, int i, vector<vector<int>> inputs_,
             vector<vector<int>> outputs_) {
    GetInputSafe(context, inputs_[i][0], &input);
    GetInputSafe(context, inputs_[i][1], &paddings);

    if (inputs_[i].size() == 3)
      GetInputSafe(context, inputs_[i][2], &constant_values);
    else constant_values = nullptr;

    GetOutputSafe(context, outputs_[i][0], &output);
    // output = GetOutput(context, node, 0);
    dims = tflite::NumDimensions(input);
    switch (paddings->type) {
    case kTfLiteInt64: {
      SetResizingCategory<int64_t>(context);
      break;
    }
    case kTfLiteInt32: SetResizingCategory<int32_t>(context); break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Padding type %s is currently not supported by Pad.",
                         TfLiteTypeGetName(paddings->type));
    }
  }

  template <typename padding_integer_type>
  void SetResizingCategory(TfLiteContext *context) {
    const padding_integer_type *paddings_data =
        tflite::GetTensorData<padding_integer_type>(paddings);
    resizing_category = tflite::ResizingCategory::kGenericResize;
    const int paddings_total = tflite::GetTensorShape(paddings).FlatSize();
    // Paddings will be a n,2 array, and we need to detect 4D arrays with the
    // pattern { {0,0}, {a, b}, {c, d}, {0,0} }.
    if (tflite::IsConstantTensor(paddings) && paddings_total == 8 &&
        (paddings_data[0] == 0 && paddings_data[1] == 0) &&
        (paddings_data[6] == 0 && paddings_data[7] == 0)) {
      resizing_category = tflite::ResizingCategory::kImageStyle;
    }
  }

  const TfLiteTensor *constant_values;
  const TfLiteTensor *input;
  const TfLiteTensor *paddings;
  TfLiteTensor *output;
  int dims;
  tflite::ResizingCategory resizing_category;
};

struct ReduceOpContext {
  ReduceOpContext(TfLiteContext *context, TfLiteReducerParams *params_, int i,
                  vector<vector<int>> inputs_, vector<vector<int>> outputs_) {
    params = params_;
    GetInputSafe(context, inputs_[i][0], &input);
    GetInputSafe(context, inputs_[i][1], &axis);
    GetOutputSafe(context, outputs_[i][0], &output);
  }
  TfLiteReducerParams *params;
  const TfLiteTensor *input;
  const TfLiteTensor *axis;
  TfLiteTensor *output;
};

struct DequantizeOpContext {
  DequantizeOpContext(TfLiteContext *context, int i,
                      vector<vector<int>> inputs_,
                      vector<vector<int>> outputs_) {
    GetInputSafe(context, inputs_[i][0], &input);
    GetOutputSafe(context, outputs_[i][0], &output);
  }
  const TfLiteTensor *input;
  TfLiteTensor *output;
};

// =========================================================
// Layer Specific Helper functions
// =========================================================

bool IsIm2ColRequired(const TfLiteTensor *input, TfLiteConvParams *params,
                      const TfLiteTensor *filter, Conv2D_Data *data,
                      bool is_hybrid) {
  // If HWCN weights are required, Im2Col not required
  if (data->need_hwcn_weights) return false;

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

// =========================================================
namespace tflite {
// From im2col_utils.h Start
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
// From im2col_utils.h End
// =========================================================

static void AddTempOutTensor(TfLiteContext *context, TfLiteNode *node,
                             bool req_temp_out, int &temporaries_count,
                             int &temp_out_tid, int &temp_out_id) {
  if (req_temp_out) {
    temp_out_id = temporaries_count;
    if (temp_out_tid == kTensorNotAllocated)
      context->AddTensors(context, 1, &temp_out_tid);
    ++temporaries_count;
  }
}

static TfLiteStatus UpdateTempTensors(TfLiteNode *node, int temporaries_count) {
  auto temp_array = TfLiteIntArrayCreate(temporaries_count);
  for (int i = 0; i < node->temporaries->size; i++)
    temp_array->data[i] = node->temporaries->data[i];

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = temp_array;
  return kTfLiteOk;
}

static TfLiteStatus ResizeTempOutTensor(TfLiteContext *context,
                                        TfLiteNode *node, bool req_temp_out,
                                        int temp_out_id,
                                        vector<vector<int>> &outputs_, int i,
                                        TfLiteIntArray *output_size) {
  if (req_temp_out) {
    node->temporaries->data[temp_out_id] = outputs_[i][0];
    TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
    temp_out_tensor->type = kTfLiteInt8;
    temp_out_tensor->allocation_type = kTfLiteArenaRw;
    auto temp_out_tensor_status =
        context->ResizeTensor(context, temp_out_tensor, output_size);
    if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
  }
  return kTfLiteOk;
}

static TfLiteStatus
ResizeTempOutTensorDefault(TfLiteContext *context, TfLiteNode *node,
                           bool req_temp_out, int temp_out_id,
                           vector<vector<int>> &outputs_, int i,
                           TfLiteIntArray *output_size) {
  if (req_temp_out) {
    node->temporaries->data[temp_out_id] = outputs_[i][0];
    TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
    auto temp_out_tensor_status =
        context->ResizeTensor(context, temp_out_tensor, output_size);
    if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
  }
  return kTfLiteOk;
}

// Returns the output shape for reduce operations.
TfLiteStatus GetReduceOutputShape(TfLiteContext *context,
                                  ReduceOpContext *op_context,
                                  TfLiteIntArray **output_shape) {
  size_t num_axis = tflite::NumElements(op_context->axis);
  const TfLiteIntArray *input_dims = op_context->input->dims;
  int input_num_dims = tflite::NumDimensions(op_context->input);
  if (input_num_dims == 0) {
    *output_shape = TfLiteIntArrayCreate(0);
    return kTfLiteOk;
  }
  const int *axis = tflite::GetTensorData<int>(op_context->axis);
  if (op_context->params->keep_dims) {
    TfLiteIntArray *output_dims = TfLiteIntArrayCreate(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          is_axis = true;
          break;
        }
      }
      if (is_axis) {
        output_dims->data[idx] = 1;
      } else {
        output_dims->data[idx] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  } else {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i) {
      int current = axis[i];
      if (current < 0) {
        current += input_num_dims;
      }
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; ++j) {
        int previous = axis[j];
        if (previous < 0) {
          previous += input_num_dims;
        }
        if (current == previous) {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    TfLiteIntArray *output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  }
}

// Resizes the temp tensor that stores resolved axis.
TfLiteStatus ResizeTempAxis(TfLiteContext *context, ReduceOpContext *op_context,
                            TfLiteTensor *resolved_axis) {
  TfLiteIntArray *axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = static_cast<int>(tflite::NumElements(op_context->axis));
  return context->ResizeTensor(context, resolved_axis, axis_size);
}

// Resizes the temp tensor that stores normalized dims.
TfLiteStatus ResizeTempDims(TfLiteContext *context, ReduceOpContext *op_context,
                            TfLiteTensor *normalized_dims) {
  TfLiteIntArray *dims_size = TfLiteIntArrayCreate(1);
  dims_size->data[0] = (op_context->input->dims->size);
  return context->ResizeTensor(context, normalized_dims, dims_size);
}

// Resizes output array based on the input size and resolved axis.
TfLiteStatus ResizeOutputTensor(TfLiteContext *context,
                                ReduceOpContext *op_context) {
  TfLiteIntArray *output_dims;
  TF_LITE_ENSURE_OK(context,
                    GetReduceOutputShape(context, op_context, &output_dims));
  return context->ResizeTensor(context, op_context->output, output_dims);
}

// Resizes the temp tensor that stores temp sum of reduced elements.
TfLiteStatus ResizeTempAccum(TfLiteContext *context,
                             ReduceOpContext *op_context,
                             TfLiteTensor *temp_accum) {
  TfLiteIntArray *size = TfLiteIntArrayCreate(1);
  size->data[0] = static_cast<int>(tflite::NumElements(op_context->output));
  return context->ResizeTensor(context, temp_accum, size);
}

static TfLiteStatus AllocateTemporaryTensorsIfRequiredCONV2D(
    TfLiteContext *context, TfLiteNode *node, bool is_hybrid,
    bool is_per_channel, size_t im2col_bytes, TfLiteConvParams *params,
    Conv2D_Data *data, bool req_temp_out, int temp_out_tid, int &temp_out_id,
    int input_tid, int filter_tid) {
  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  const TfLiteTensor *input;
  const TfLiteTensor *filter;
  GetInputSafe(context, input_tid, &input);
  GetInputSafe(context, filter_tid, &filter);
  data->need_hwcn_weights = false;
  data->need_im2col = IsIm2ColRequired(input, params, filter, data, is_hybrid);

  int temporaries_count = node->temporaries->size;
  if (data->need_im2col) {
    data->im2col_index = temporaries_count;
    if (data->im2col_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->im2col_id);
    }
    ++temporaries_count;
  }
  if (data->need_hwcn_weights) {
    data->hwcn_weights_index = temporaries_count;
    if (data->hwcn_weights_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->hwcn_weights_id);
    }
    ++temporaries_count;
  }

  AddTempOutTensor(context, node, req_temp_out, temporaries_count, temp_out_tid,
                   temp_out_id);
  return UpdateTempTensors(node, temporaries_count);
}

static TfLiteStatus AllocateTemporaryOutTensorsIfRequiredTCONV(
    TfLiteContext *context, TfLiteNode *node, TCONV_Data *data,
    bool req_temp_out, int temp_out_tid, int &temp_out_id) {
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

  AddTempOutTensor(context, node, req_temp_out, temporaries_count, temp_out_tid,
                   temp_out_id);
  return UpdateTempTensors(node, temporaries_count);
}

static TfLiteStatus AllocateTemporaryOutTensorsIfRequiredMEAN(
    TfLiteContext *context, TfLiteNode *node, REDUCE_Data *data,
    bool req_temp_out, int temp_out_tid, int &temp_out_id,
    vector<tuple<int, int>> &temp_tensor_ids) {

  int scratch_tensor_index = data->scratch_tensor_index;
  int resolved_axis_tensor_index = data->scratch_tensor_index + 1;
  int temp_accum_tensor_index = data->scratch_tensor_index + 2;
  int normalized_dims_tensor_index = data->scratch_tensor_index + 3;

  int temporaries_count = node->temporaries->size;
  if (data->scratch_tensor_index == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->scratch_tensor_index);
  }
  temp_tensor_ids.push_back(
      make_tuple(temporaries_count, scratch_tensor_index));
  ++temporaries_count;

  if ((resolved_axis_tensor_index) == kTensorNotAllocated) {
    context->AddTensors(context, 1, &resolved_axis_tensor_index);
  }
  temp_tensor_ids.push_back(
      make_tuple(temporaries_count, resolved_axis_tensor_index));
  ++temporaries_count;

  if ((temp_accum_tensor_index) == kTensorNotAllocated) {
    context->AddTensors(context, 1, &temp_accum_tensor_index);
  }
  temp_tensor_ids.push_back(
      make_tuple(temporaries_count, temp_accum_tensor_index));
  ++temporaries_count;

  if ((normalized_dims_tensor_index) == kTensorNotAllocated) {
    context->AddTensors(context, 1, &normalized_dims_tensor_index);
  }
  temp_tensor_ids.push_back(
      make_tuple(temporaries_count, normalized_dims_tensor_index));
  ++temporaries_count;

  AddTempOutTensor(context, node, req_temp_out, temporaries_count, temp_out_tid,
                   temp_out_id);

  return UpdateTempTensors(node, temporaries_count);
}

static TfLiteStatus
AllocateTemporaryOutTensorsIfRequired(TfLiteContext *context, TfLiteNode *node,
                                      bool req_temp_out, int temp_out_tid,
                                      int &temp_out_id) {
  int temporaries_count = node->temporaries->size;
  AddTempOutTensor(context, node, req_temp_out, temporaries_count, temp_out_tid,
                   temp_out_id);
  return UpdateTempTensors(node, temporaries_count);
}

bool CheckPaddingOverflow(PadContext *op_context) {
  if (op_context->paddings->type == kTfLiteInt64) {
    const int64_t *paddings_data =
        tflite::GetTensorData<int64_t>(op_context->paddings);
    if (paddings_data != nullptr) {
      int64_t int32_min =
          static_cast<int64_t>(std::numeric_limits<int32_t>::min());
      int64_t int32_max =
          static_cast<int64_t>(std::numeric_limits<int32_t>::max());
      for (int idx = 0; idx < op_context->dims; ++idx) {
        int64_t padding = paddings_data[idx];
        if (padding < int32_min || padding > int32_max) {
          return true;
        }
      }
    }
  }
  return false;
}

// =========================================================
// Compute/Preload Functions
// =========================================================

void precal_wsum(const int8_t *weight_data, int *dims, vector<int> &wt_sum) {
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];
  int max = width * depth;

  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));

  for (int i = 0; i < w; i++) {
    int s0 = 0;
    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 = (i * depth + j >= max) ? 0 : weight_data[i * depth + j];
        s0 += w0;
      }
    }
    wt_sum.push_back(s0);
  }
}

void precal_wsum(const TfLiteTensor *filter, vector<int> &wt_sum) {
  // Assumes weight channels is always the first dimension
  int width = filter->dims->data[0];
  int depth = 1;
  for (int i = 1; i < filter->dims->size; i++) depth *= filter->dims->data[i];
  int max = width * depth;
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));

  for (int i = 0; i < width; i++) {
    int s0 = 0;
    for (int j = 0; j < depth; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * depth + j >= max) ? 0 : filter->data.int8[i * depth + j];
        s0 += w0;
      }
    }
    wt_sum.push_back(s0);
  }
}

TfLiteStatus ComputeDepthMultiplier(TfLiteContext *context,
                                    const TfLiteTensor *input,
                                    const TfLiteTensor *filter,
                                    int16 *depth_multiplier) {
  int num_filter_channels = tflite::SizeOfDimension(filter, 3);
  int num_input_channels = tflite::SizeOfDimension(input, 3);
  TF_LITE_ENSURE(context, num_input_channels != 0);
  TF_LITE_ENSURE_EQ(context, num_filter_channels % num_input_channels, 0);
  *depth_multiplier = num_filter_channels / num_input_channels;
  return kTfLiteOk;
}

inline int32_t RoundingDivideByPOT(int32_t x, int exponent) {
  std::int32_t msk = (1 << exponent) - 1;
  std::int32_t sm = msk >> 1;
  std::int32_t val_3 = x >> exponent;

  std::int32_t temp_2 = x & msk;
  std::int32_t temp_3 = (x < 0) & 1;
  std::int32_t temp_4 = sm + temp_3;
  std::int32_t temp_5 = ((temp_2 > temp_4) & 1);
  std::int32_t result_32 = val_3 + temp_5;
  return result_32;
}

inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                      std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(
    int32_t x, int32_t quantized_multiplier, int left_shift) {
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}

int Quantised_Multiplier_V1(int x, int qm, int shift, int out_offset,
                            int out_min, int out_max) {
  int nshift = shift;
  int total_shift = 31 - shift;
  int64_t x_64 = x;
  int64_t quantized_multiplier_64(qm);
  int64_t one = 1;
  int64_t round = one << (total_shift - 1);
  int64_t result = x_64 * quantized_multiplier_64 + round;
  result = result >> total_shift;
  int nresult = result;
  if (result > std::numeric_limits<int32_t>::max())
    result = std::numeric_limits<int32_t>::max();
  if (result < std::numeric_limits<int32_t>::min())
    result = std::numeric_limits<int32_t>::min();
  int32_t result_32 = result;
  result_32 += out_offset;
  // clamp
  result_32 = std::max(result_32, out_min);
  result_32 = std::min(result_32, out_max);
  return result_32;
}

int32_t Quantised_Multiplier_V2(int32_t x, int32_t quantized_multiplier,
                                int shift, int out_offset, int out_min,
                                int out_max) {

  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  int32_t result_32 =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                              x * (1 << left_shift), quantized_multiplier),
                          right_shift);
  result_32 += out_offset;
  // clamp
  result_32 = std::max(result_32, out_min);
  result_32 = std::min(result_32, out_max);
  return result_32;
}

template <typename OutType>
void ExtractShape(const TfLiteTensor *input, OutType *output_data) {
  for (int i = 0; i < tflite::NumDimensions(input); ++i) {
    output_data[i] = tflite::SizeOfDimension(input, i);
  }
}

TfLiteStatus InitializeMeanOutputTyped(TfLiteTensor *output) {
  tflite::RuntimeShape output_shape = tflite::GetTensorShape(output);
  const size_t flat_size = output_shape.FlatSize();
  int8_t *output_data = tflite::GetTensorData<int8_t>(output);
  int8_t nan_value = std::numeric_limits<int8_t>::quiet_NaN();
  for (int idx = 0; idx < flat_size; ++idx) {
    *output_data++ = nan_value;
  }
  return kTfLiteOk;
}

inline bool IsQuantizedPerChannel(const TfLiteTensor *input) {
  if (input->quantization.type == kTfLiteAffineQuantization &&
      input->quantization.params) {
    auto *quant_params = reinterpret_cast<TfLiteAffineQuantization *>(
        input->quantization.params);
    return (quant_params->scale && quant_params->scale->size > 1);
  }
  return false;
}
namespace {

template <QuantizeKernelType kernel_type, typename output_type>
static inline void AffineQuantize(const tflite::QuantizationParams &op_params,
                                  const tflite::RuntimeShape &input_shape,
                                  const float *input_data,
                                  const tflite::RuntimeShape &output_shape,
                                  output_type *output_data) {
  if (kernel_type == kReference) {
    tflite::reference_ops::AffineQuantize(op_params, input_shape, input_data,
                                          output_shape, output_data);
  } else {
    tflite::optimized_ops::AffineQuantize(op_params, input_shape, input_data,
                                          output_shape, output_data);
  }
}

template <QuantizeKernelType kernel_type, typename input_type,
          typename output_type>
static inline void Requantize(const input_type *input_data, int32_t size,
                              int32_t effective_scale_multiplier,
                              int32_t effective_scale_shift,
                              int32_t input_zeropoint, int32_t output_zeropoint,
                              output_type *output_data) {
  if (kernel_type == kReference) {
    tflite::reference_ops::Requantize(
        input_data, size, effective_scale_multiplier, effective_scale_shift,
        input_zeropoint, output_zeropoint, output_data);
  } else {
    tflite::optimized_ops::Requantize(
        input_data, size, effective_scale_multiplier, effective_scale_shift,
        input_zeropoint, output_zeropoint, output_data);
  }
}
} // namespace

// =========================================================
// Prepare function for all supported Ops
// =========================================================

// tensorflow/lite/kernels/add.cc
bool Prepare_ADD_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                      void *layers_params, void *opdatas,
                      vector<vector<int>> inputs_, vector<vector<int>> outputs_,
                      int &out_tid) {

  TfLiteAddParams *params = reinterpret_cast<TfLiteAddParams *>(layers_params);
  ADD_Data *data = reinterpret_cast<ADD_Data *>(opdatas);
  const TfLiteTensor *input1;
  const TfLiteTensor *input2;
  TfLiteTensor *output;

  GetInputSafe(context, inputs_[i][0], &input1);
  GetInputSafe(context, inputs_[i][1], &input2);
  GetOutputSafe(context, outputs_[i][0], &output);

  output->type = input2->type;
  TfLiteIntArray *output_size = TfLiteIntArrayCopy(input1->dims);
  const bool requires_broadcast = false;
  bool general_scale_int16 = false;
  bool input1_scale_is_pot = false;
  bool input2_scale_is_pot = false;
  bool output_scale_is_pot = false;
  int input1_scale_log2_rounded{0};
  int input2_scale_log2_rounded{0};
  int output_scale_log2_rounded{0};
  data->pot_scale_int16 = !general_scale_int16;
  data->input1_offset = -input1->params.zero_point;
  data->input2_offset = -input2->params.zero_point;
  data->output_offset = output->params.zero_point;
  data->left_shift = general_scale_int16 ? 15 : 20;
  const double twice_max_input_scale =
      2 * std::max(input1->params.scale, input2->params.scale);
  const double real_input1_multiplier =
      input1->params.scale / twice_max_input_scale;
  const double real_input2_multiplier =
      input2->params.scale / twice_max_input_scale;
  const double real_output_multiplier =
      twice_max_input_scale / ((1 << data->left_shift) * output->params.scale);
  tflite::QuantizeMultiplierSmallerThanOneExp(
      real_input1_multiplier, &data->input1_multiplier, &data->input1_shift);
  tflite::QuantizeMultiplierSmallerThanOneExp(
      real_input2_multiplier, &data->input2_multiplier, &data->input2_shift);
  tflite::QuantizeMultiplierSmallerThanOneExp(
      real_output_multiplier, &data->output_multiplier, &data->output_shift);
  tflite::CalculateActivationRangeQuantized(context, params->activation, output,
                                            &data->output_activation_min,
                                            &data->output_activation_max);

  // Output tensor management
  context->ResizeTensor(context, output, output_size);

  // Temporary tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));

  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  return kTfLiteOk;
}

// tensorflow/lite/kernels/conv.cc
bool Prepare_CONV2D_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                         void *layers_params, void *opdatas,
                         vector<vector<int>> inputs_,
                         vector<vector<int>> outputs_, int &out_tid,
                         vector<int> &wt_sum, vector<int8_t> &temp_im2col) {

  TfLiteConvParams *params =
      reinterpret_cast<TfLiteConvParams *>(layers_params);
  Conv2D_Data *data = reinterpret_cast<Conv2D_Data *>(opdatas);

  TfLiteTensor *output;
  const TfLiteTensor *input;
  const TfLiteTensor *filter;
  const TfLiteTensor *bias;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &input);
  GetInputSafe(context, inputs_[i][1], &filter);
  GetInputSafe(context, inputs_[i][2], &bias);

  const bool is_hybrid = false;
  int channels_in = filter->dims->data[3];
  int channels_out = filter->dims->data[0];
  int width = input->dims->data[2];
  int height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int batches = input->dims->data[0];
  auto padding = params->padding;
  int out_width, out_height;
  data->padding = tflite::ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  size_t im2col_type_size = sizeof(int8_t);
  const size_t im2col_bytes = static_cast<size_t>(batches) * out_height *
                              out_width * channels_in * filter_height *
                              filter_width * im2col_type_size;

  // Quantization Parameters Calculation
  TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                    kTfLiteAffineQuantization);
  const auto *affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                           affine_quantization->scale->size == channels_out));
  data->per_channel_output_multiplier.resize(channels_out);
  data->per_channel_output_shift.resize(channels_out);

  TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
      context, input, filter, bias, output, params->activation,
      &data->output_multiplier, &data->output_shift,
      &data->output_activation_min, &data->output_activation_max,
      data->per_channel_output_multiplier.data(),
      data->per_channel_output_shift.data(), channels_out));

  // Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequiredCONV2D(
      context, node, is_hybrid, data->is_hybrid_per_channel, im2col_bytes,
      params, data, req_temp_out, outputs_[i][0], temp_out_id, inputs_[i][0],
      inputs_[i][1]));

  TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  auto output_status = context->ResizeTensor(context, output, output_size);
  if (output_status != kTfLiteOk) return output_status;

  // IM2COL tensor management
  if (data->need_im2col) {
    node->temporaries->data[data->im2col_index] = data->im2col_id;
    TfLiteIntArray *im2col_size = TfLiteIntArrayCreate(4);
    int input_depth = input->dims->data[3];
    im2col_size->data[0] = output_size->data[0];
    im2col_size->data[1] = output_size->data[1];
    im2col_size->data[2] = output_size->data[2];
    im2col_size->data[3] = input_depth * filter_height * filter_width;

    TfLiteTensor *im2col =
        &context->tensors[node->temporaries->data[data->im2col_index]];
    im2col->type = input->type;
    if (is_hybrid) {
      im2col->type = filter->type;
    }
    im2col->allocation_type = kTfLiteArenaRw;
    auto im2col_status = context->ResizeTensor(context, im2col, im2col_size);
    if (im2col_status != kTfLiteOk) return im2col_status;
    temp_im2col.resize(im2col_bytes);
  }

  // Weights tensor management
  if (data->need_hwcn_weights) {
    node->temporaries->data[data->hwcn_weights_index] = data->hwcn_weights_id;
    TfLiteIntArray *hwcn_weights_size = TfLiteIntArrayCreate(2);

    int input_depth = input->dims->data[3];
    hwcn_weights_size->data[0] = (filter_height * filter_width * input_depth);
    hwcn_weights_size->data[1] = channels_out;

    TfLiteTensor *hwcn_weights =
        &context->tensors[node->temporaries->data[data->hwcn_weights_index]];
    hwcn_weights->type = input->type;
    hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;
    auto hwcn_weights_status =
        context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
    if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

    data->have_weights_been_transposed = false;
  }
  // Temporary output tensor management
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  // Accelerator specific optimisations
  precal_wsum(filter, wt_sum);
  return kTfLiteOk;
}

// tensorflow/lite/kernels/fully_connected.cc
bool Prepare_FC_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                     void *layers_params, void *opdatas,
                     vector<vector<int>> inputs_, vector<vector<int>> outputs_,
                     int &out_tid, vector<int> &wt_sum) {
  TfLiteFullyConnectedParams *params =
      reinterpret_cast<TfLiteFullyConnectedParams *>(layers_params);
  FC_Data *data = reinterpret_cast<FC_Data *>(opdatas);

  const TfLiteTensor *input;
  const TfLiteTensor *filter;
  const TfLiteTensor *bias;
  TfLiteTensor *output;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &input);
  GetInputSafe(context, inputs_[i][1], &filter);
  bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
  if (isBias) GetInputSafe(context, inputs_[i][2], &bias);
  else bias = nullptr;

  // Get Qaunt Params.
  double real_multiplier = 0.0;
  tflite::GetQuantizedConvolutionMultipler(context, input, filter, bias, output,
                                           &real_multiplier);
  int exponent;
  tflite::QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                             &exponent);
  data->output_shift = exponent;

  // Populate per-channel quantization parameters, if per-channel
  // quantization.
  TF_LITE_ENSURE_EQ(context, input->quantization.type,
                    kTfLiteAffineQuantization);
  TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                    kTfLiteAffineQuantization);
  const auto *affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  const int per_channel_quantization_size = affine_quantization->scale->size;
  const bool is_per_channel = per_channel_quantization_size > 1;
  if (is_per_channel) {
    //  Currently only Int8/Int16 is supported for per channel quantization.
    TF_LITE_ENSURE(context,
                   input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
    TF_LITE_ENSURE(context, (filter->type == kTfLiteInt8));
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      per_channel_quantization_size);
    TF_LITE_ENSURE_EQ(
        context, per_channel_quantization_size,
        filter->dims->data[affine_quantization->quantized_dimension]);
    // Populate multiplier and shift using affine quantization.
    const float input_scale = input->params.scale;
    const float output_scale = output->params.scale;
    const float *filter_scales = affine_quantization->scale->data;
    data->per_channel_output_multiplier.resize(per_channel_quantization_size);
    data->per_channel_output_shift.resize(per_channel_quantization_size);
    int32_t *per_channel_multiplier =
        data->per_channel_output_multiplier.data();
    int32_t *per_channel_shift = data->per_channel_output_shift.data();
    for (int i = 0; i < per_channel_quantization_size; ++i) {
      const float scale = filter_scales[i];
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t significand;
      int channel_shift;
      tflite::QuantizeMultiplier(effective_output_scale, &significand,
                                 &channel_shift);
      per_channel_multiplier[i] = significand;
      per_channel_shift[i] = channel_shift;
    }
  }

  TF_LITE_ENSURE_STATUS(tflite::CalculateActivationRangeQuantized(
      context, params->activation, output, &data->output_activation_min,
      &data->output_activation_max));

  // Resize output.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++)
    input_size *= input->dims->data[i];
  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  const int out_dim1 = batch_size;
  const int out_dim2 = num_units;
  TfLiteIntArray *output_size = nullptr;
  if (params->keep_num_dims) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1],
                      filter->dims->data[1]);
    output_size = TfLiteIntArrayCopy(input->dims);
    output_size->data[output_size->size - 1] = num_units;
  } else {
    // Otherwise, the output is (potentially flattened to) a 2-D matrix.
    output_size = TfLiteIntArrayCreate(2);
    output_size->data[0] = batch_size;
    output_size->data[1] = num_units;
  }
  auto output_status = context->ResizeTensor(context, output, output_size);
  if (output_status != kTfLiteOk) return output_status;

  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));
  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);
  // Accelerator specific optimisations
  precal_wsum(filter, wt_sum);
  return kTfLiteOk;
}

// tensorflow/lite/kernels/depthwise_conv.cc
bool Prepare_DWCONV2D_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                           void *layers_params, void *opdatas,
                           vector<vector<int>> inputs_,
                           vector<vector<int>> outputs_, int &out_tid,
                           vector<int> &wt_sum) {

  TfLiteDepthwiseConvParams *params =
      reinterpret_cast<TfLiteDepthwiseConvParams *>(layers_params);
  DWCONV2D_Data *data = reinterpret_cast<DWCONV2D_Data *>(opdatas);

  const TfLiteTensor *input;
  const TfLiteTensor *filter;
  const TfLiteTensor *bias;
  TfLiteTensor *output;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &input);
  GetInputSafe(context, inputs_[i][1], &filter);
  bool has_bias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
  if (has_bias) GetInputSafe(context, inputs_[i][2], &bias);
  else bias = nullptr;

  const TfLiteType data_type = input->type;
  const TfLiteType filter_type = filter->type;

  if (has_bias) {
    TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(bias), 1);
    TF_LITE_ENSURE_EQ(context, tflite::SizeOfDimension(filter, 3),
                      tflite::SizeOfDimension(bias, 0));
  }
  // Filter in DepthwiseConv is expected to be [1, H, W, O].
  TF_LITE_ENSURE_EQ(context, tflite::SizeOfDimension(filter, 0), 1);

  int channels_out = tflite::SizeOfDimension(filter, 3);
  int width = tflite::SizeOfDimension(input, 2);
  int height = tflite::SizeOfDimension(input, 1);
  int filter_width = tflite::SizeOfDimension(filter, 2);
  int filter_height = tflite::SizeOfDimension(filter, 1);
  int batches = tflite::SizeOfDimension(input, 0);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;

  data->padding = tflite::ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Quantization Parameters Calculation
  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training or
  // calibration.
  if (data_type != kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE(context, filter->quantization.type != kTfLiteNoQuantization);
    const auto *affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization *>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                             affine_quantization->scale->size == channels_out));

    data->per_channel_output_multiplier.resize(channels_out);
    data->per_channel_output_shift.resize(channels_out);
    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), channels_out));
  }

  // Output tensor management
  TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  auto output_status = context->ResizeTensor(context, output, output_size);
  if (output_status != kTfLiteOk) return output_status;

  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));
  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  // Accelerator specific optimisations
  precal_wsum(filter, wt_sum);
  return kTfLiteOk;
}

// tensorflow/lite/kernels/transpose_conv.cc
bool Prepare_TCONV_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                        void *layers_params, void *opdatas,
                        vector<vector<int>> inputs_,
                        vector<vector<int>> outputs_, int &out_tid,
                        vector<int> &wt_sum) {
  TfLiteTransposeConvParams *params =
      reinterpret_cast<TfLiteTransposeConvParams *>(layers_params);
  TCONV_Data *data = reinterpret_cast<TCONV_Data *>(opdatas);

  TfLiteTensor *output;
  const TfLiteTensor *output_shape;
  const TfLiteTensor *input;
  const TfLiteTensor *weights;
  const TfLiteTensor *bias;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &output_shape);
  GetInputSafe(context, inputs_[i][1], &weights);
  GetInputSafe(context, inputs_[i][2], &input);

  bool has_bias = (inputs_[i].size() == 4);
  TF_LITE_ENSURE(context, has_bias || tflite::NumInputs(node) == 3);

  if (has_bias) GetInputSafe(context, inputs_[i][3], &bias);
  else bias = nullptr;

  if (bias) {
    TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
    if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    }
    TF_LITE_ENSURE_EQ(context, tflite::NumElements(bias),
                      tflite::SizeOfDimension(weights, 0));
  }

  // Ensure that weights and inputs have the same channel dimension.
  // Note: TOCO will reorder weights in the following format: OHWI.
  TF_LITE_ENSURE_EQ(context, tflite::SizeOfDimension(input, 3),
                    tflite::SizeOfDimension(weights, 3));

  int stride_x = params->stride_width;
  int stride_y = params->stride_height;
  int filters = weights->dims->data[0];
  int kernel_size = weights->dims->data[1];
  int in1 = input->dims->data[1];
  int in2 = input->dims->data[2];
  int in3 = weights->dims->data[3];

  // Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;

  // Allocate Temporary Tensors (Output, Scratch, Transposed Weights, Col2Im)
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequiredTCONV(
      context, node, data, req_temp_out, outputs_[i][0], temp_out_id));

  TfLiteTensor *col2im = nullptr;
  if (data->has_col2im) {
    node->temporaries->data[data->col2im_index] = data->col2im_id;
    TF_LITE_ENSURE_OK(context, tflite::GetTemporarySafe(
                                   context, node, data->col2im_index, &col2im));
  }

  // Resize output tensor
  if (req_temp_out && !tflite::IsConstantTensor(output_shape)) {
    node->temporaries->data[temp_out_id] = outputs_[i][0];
    TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
    temp_out_tensor->type = kTfLiteInt8;
    temp_out_tensor->allocation_type = kTfLiteDynamic;
    TF_LITE_ENSURE_STATUS(ResizeTensor(context, output, temp_out_tensor));
  }

  // Weight tensor management
  if (data->weights_are_transposed) {
    node->temporaries->data[data->transposed_weights_index] =
        data->transposed_weights_id;
    TfLiteTensor *transposed_weights;
    TF_LITE_ENSURE_OK(context,
                      tflite::GetTemporarySafe(context, node,
                                               data->transposed_weights_index,
                                               &transposed_weights));
    if (!tflite::IsConstantTensor(weights)) {
      tflite::SetTensorToDynamic(transposed_weights);
    } else {
      tflite::ResizeAndTransposeWeights(context, weights, transposed_weights);
    }
  }

  // Scratch buffer management
  node->temporaries->data[data->scratch_tensor_index] = data->scratch_tensor_id;
  TfLiteTensor *scratch_buffer;
  TF_LITE_ENSURE_OK(
      context, tflite::GetTemporarySafe(
                   context, node, data->scratch_tensor_index, &scratch_buffer));
  scratch_buffer->type = kTfLiteInt32;
  scratch_buffer->allocation_type = kTfLiteDynamic;
  TF_LITE_ENSURE_STATUS(ResizeTensor(context, output, scratch_buffer));

  // Quantization Parameters Calculation
  TF_LITE_ENSURE_EQ(context, weights->quantization.type,
                    kTfLiteAffineQuantization);
  const auto *affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization *>(
          weights->quantization.params);
  const int channels_out = weights->dims->data[0];
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                           affine_quantization->scale->size == channels_out));
  data->per_channel_output_multiplier.resize(channels_out);
  data->per_channel_output_shift.resize(channels_out);
  TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
      context, input, weights, bias, output, kTfLiteActNone,
      &data->output_multiplier, &data->output_shift,
      &data->output_activation_min, &data->output_activation_max,
      data->per_channel_output_multiplier.data(),
      data->per_channel_output_shift.data(), channels_out));

  // Accelerator specific optimisations
  TfLiteTensor *transposed_weights;
  TF_LITE_ENSURE_OK(context, tflite::GetTemporarySafe(
                                 context, node, data->transposed_weights_index,
                                 &transposed_weights));
  precal_wsum(transposed_weights, wt_sum);
  return kTfLiteOk;
}

// tensorflow/lite/kernels/shape.cc
bool Prepare_SHAPE_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                        void *layers_params, void *opdatas,
                        vector<vector<int>> inputs_,
                        vector<vector<int>> outputs_, int &out_tid) {
  TF_LITE_ENSURE_EQ(context, inputs_[i].size(), 1);
  TF_LITE_ENSURE_EQ(context, outputs_[i].size(), 1);

  TfLiteTensor *output;
  const TfLiteTensor *input;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &input);

  TfLiteShapeParams *params =
      reinterpret_cast<TfLiteShapeParams *>(layers_params);
  switch (params->out_type) {
  case kTfLiteInt32: output->type = kTfLiteInt32; break;
  default:
    TF_LITE_KERNEL_LOG(context, "Unknown shape output data type: %d",
                       params->out_type);
    return kTfLiteError;
  }

  // By design, the input shape is always known at the time of Prepare, even
  // if the preceding op that generates |input| is dynamic. Thus, we can
  // always compute the shape immediately, without waiting for Eval.
  tflite::SetTensorToPersistentRo(output);

  // Shape always produces a 1-dimensional output tensor, where each output
  // element is the length of the corresponding input tensor's dimension.
  TfLiteIntArray *output_size = TfLiteIntArrayCreate(1);
  output_size->data[0] = tflite::NumDimensions(input);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));

  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));
  // Resize temporary output tensor
  ResizeTempOutTensorDefault(context, node, req_temp_out, temp_out_id, outputs_,
                             i, output_size);

  TFLITE_DCHECK_EQ(tflite::NumDimensions(output), 1);
  TFLITE_DCHECK_EQ(tflite::SizeOfDimension(output, 0),
                   tflite::NumDimensions(input));

  // Immediately propagate the known shape to the output tensor. This allows
  // downstream ops that rely on the value to use it during prepare.
  switch (output->type) {
  case kTfLiteInt32:
    ExtractShape(input, tflite::GetTensorData<int32_t>(output));
    break;
  default: return kTfLiteError;
  }

  return kTfLiteOk;
}

// tensorflow/lite/kernels/activations.cc
bool Prepare_SOFTMAX_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                          void *layers_params, void *opdatas,
                          vector<vector<int>> inputs_,
                          vector<vector<int>> outputs_, int &out_tid) {
  // TF_LITE_ENSURE_EQ(context, inputs_[i].size(), 1);
  // TF_LITE_ENSURE_EQ(context, outputs_[i].size(), 1);

  auto *params = reinterpret_cast<TfLiteSoftmaxParams *>(layers_params);
  SOFTMAX_Data *data = reinterpret_cast<SOFTMAX_Data *>(opdatas);

  TfLiteTensor *output;
  const TfLiteTensor *input;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &input);

  TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
  TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 256,
                      (0.001f * 1.f / 256));

  // const int kScaledDiffIntegerBits = 5;
  // int input_left_shift;
  // tflite::PreprocessSoftmaxScaling(
  //     static_cast<double>(params->beta),
  //     static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
  //     &data->params.input_multiplier, &input_left_shift);
  // data->params.input_left_shift = input_left_shift;
  // data->params.diff_min = -1.0 * tflite::CalculateInputRadius(
  //                                    kScaledDiffIntegerBits,
  //                                    input_left_shift);

  data->params.table = data->table;
  tflite::optimized_ops::PopulateSoftmaxLookupTable(
      &data->params, input->params.scale, params->beta);
  data->params.zero_point = output->params.zero_point;
  data->params.scale = output->params.scale;

  // Resize Output tensor
  TfLiteIntArray *output_size = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));

  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));
  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  return kTfLiteOk;
}

// tensorflow/lite/kernels/pad.cc
bool Prepare_PAD_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                      void *layers_params, void *opdatas,
                      vector<vector<int>> inputs_, vector<vector<int>> outputs_,
                      int &out_tid) {

  TF_LITE_ENSURE(context, inputs_[i].size() == 2 || inputs_[i].size() == 3);
  TF_LITE_ENSURE_EQ(context, outputs_[i].size(), 1);

  PadContext op_context(context, i, inputs_, outputs_);
  if (tflite::IsConstantTensor(op_context.paddings)) {
    TF_LITE_ENSURE_MSG(context, !CheckPaddingOverflow(&op_context),
                       "INT64 padding overflow. Only support value between "
                       "INT32_MIN and INT32_MAX.");
  }
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);
  if (op_context.constant_values != nullptr) {
    TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                            op_context.constant_values->type);
  }

  // Ensure we do not exceed maximum dimension count.
  TF_LITE_ENSURE(context, op_context.dims <= PadKernelMaxDimensionCount);

  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));

  // Exit early if paddings is a non-const tensor or the given input is an
  // unranked input. Set output tensor to dynamic so output size can be
  // determined in Eval.
  if (tflite::NumDimensions(op_context.input) == 0 ||
      !tflite::IsConstantOrPersistentTensor(op_context.paddings)) {
    tflite::SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }

  // Ensures the paddings array is dims x 2.
  TF_LITE_ENSURE_EQ(context, tflite::SizeOfDimension(op_context.paddings, 0),
                    op_context.dims);
  TF_LITE_ENSURE_EQ(context, tflite::SizeOfDimension(op_context.paddings, 1),
                    2);

  // Right now we only support paddings between INT32_MIN and INT32_MAX, so
  // we are using int here and below.
  TfLiteIntArray *input_size = op_context.input->dims;
  TfLiteIntArray *output_size = TfLiteIntArrayCopy(input_size);
  const int32_t *paddings_data =
      tflite::GetTensorData<int32_t>(op_context.paddings);
  for (int idx = 0; idx < op_context.dims; ++idx) {
    // Paddings are between INT32_MIN and INT32_MAX.
    int before_padding = static_cast<int>(*paddings_data++);
    int after_padding = static_cast<int>(*paddings_data++);
    TF_LITE_ENSURE_MSG(context, (before_padding >= 0 && after_padding >= 0),
                       "Pad value has to be greater than equal to 0.");
  }
  paddings_data = tflite::GetTensorData<int32_t>(op_context.paddings);
  for (int idx = 0; idx < op_context.dims; ++idx) {
    // Paddings are between INT32_MIN and INT32_MAX.
    int before_padding = static_cast<int>(*paddings_data++);
    int after_padding = static_cast<int>(*paddings_data++);
    output_size->data[idx] =
        (input_size->data[idx] + before_padding + after_padding);
  }

  // Output tensor management
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, op_context.output, output_size));

  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);
  return kTfLiteOk;
}

// tensorflow/lite/kernels/mean.cc
bool Prepare_MEAN_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                       void *layers_params, void *opdatas,
                       vector<vector<int>> inputs_,
                       vector<vector<int>> outputs_, int &out_tid,
                       vector<tuple<int, int>> &temp_tensor_ids) {

  TF_LITE_ENSURE_EQ(context, inputs_[i].size(), 2);
  TF_LITE_ENSURE_EQ(context, outputs_[i].size(), 1);

  REDUCE_Data *data = reinterpret_cast<REDUCE_Data *>(opdatas);
  TfLiteReducerParams *params =
      reinterpret_cast<TfLiteReducerParams *>(layers_params);

  ReduceOpContext op_context(context, params, i, inputs_, outputs_);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.axis->type, kTfLiteInt32);
  TfLiteIntArray *output_size;
  TF_LITE_ENSURE_OK(context,
                    GetReduceOutputShape(context, &op_context, &output_size));
  int output_num_elements = 1;
  for (int i = 0; i < output_size->size; ++i)
    output_num_elements *= output_size->data[i];

  // Quantization Parameters Calculation
  const double real_multiplier =
      static_cast<double>(op_context.input->params.scale) /
      static_cast<double>(op_context.output->params.scale);
  int exponent;
  tflite::QuantizeMultiplier(real_multiplier, &data->multiplier, &exponent);
  data->shift = exponent;

  // Creates a temp index to iterate through input data.
  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequiredMEAN(
      context, node, data, req_temp_out, outputs_[i][0], temp_out_id,
      temp_tensor_ids));

  // Resize output tensor
  context->ResizeTensor(context, op_context.output, output_size);

  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  int scratch_index = get<0>(temp_tensor_ids[0]);
  int resolved_axis_index = get<0>(temp_tensor_ids[1]);
  int temp_accum_index = get<0>(temp_tensor_ids[2]);
  int normalized_dims_index = get<0>(temp_tensor_ids[3]);

  node->temporaries->data[scratch_index] = get<1>(temp_tensor_ids[0]);
  node->temporaries->data[resolved_axis_index] = get<1>(temp_tensor_ids[1]);
  node->temporaries->data[temp_accum_index] = get<1>(temp_tensor_ids[2]);
  node->temporaries->data[normalized_dims_index] = get<1>(temp_tensor_ids[3]);

  TfLiteTensor *scratch_tensor;
  tflite::GetTemporarySafe(context, node, scratch_index, &scratch_tensor);
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray *index_size = TfLiteIntArrayCreate(1);
  index_size->data[0] = tflite::NumDimensions(op_context.input);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, scratch_tensor, index_size));

  // Creates a temp tensor to store resolved axis given input data.
  TfLiteTensor *resolved_axis;
  tflite::GetTemporarySafe(context, node, resolved_axis_index, &resolved_axis);
  resolved_axis->type = kTfLiteInt32;

  // Creates a temporary accumulation tensor to store temp sums when calculating
  // mean or temp prod when calculating reduce prod.
  TfLiteTensor *temp_accum;
  tflite::GetTemporarySafe(context, node, temp_accum_index, &temp_accum);
  temp_accum->type = kTfLiteInt32;

  // Creates a temp tensor to store normalized shape given input data.
  TfLiteTensor *normalized_dims;
  tflite::GetTemporarySafe(context, node, normalized_dims_index,
                           &normalized_dims);
  normalized_dims->type = kTfLiteInt32;

  data->noop = tflite::IsConstantOrPersistentTensor(op_context.input) &&
               tflite::IsConstantOrPersistentTensor(op_context.axis);

  if (data->noop)
    data->noop &= output_num_elements <= kMaxConstantOutputTensorSize;

  if (!tflite::IsConstantOrPersistentTensor(op_context.input)) {
    tflite::SetTensorToDynamic(normalized_dims);
  } else {
    TfLiteTensorDataFree(normalized_dims);
    normalized_dims->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      ResizeTempDims(context, &op_context, normalized_dims));
  }
  // Leaves work to Eval if axis is not constant; else resizes output.
  if (!tflite::IsConstantOrPersistentTensor(op_context.axis)) {
    tflite::SetTensorToDynamic(op_context.output);
    tflite::SetTensorToDynamic(resolved_axis);
    return kTfLiteOk;
  }
  TfLiteTensorDataFree(resolved_axis);
  resolved_axis->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    ResizeTempAxis(context, &op_context, resolved_axis));

  // reduce_mean requires a buffer to store intermediate sum result.
  TfLiteTensor *temp_sum;
  tflite::GetTemporarySafe(context, node, temp_accum_index, &temp_sum);
  if (!tflite::IsConstantOrPersistentTensor(op_context.axis)) {
    tflite::SetTensorToDynamic(temp_sum);
    return kTfLiteOk;
  }
  temp_sum->allocation_type = kTfLiteArenaRw;
  ResizeTempAccum(context, &op_context, temp_sum);

  return kTfLiteOk;
}

// tensorflow/lite/kernels/quantize.cc
bool Prepare_QUANTIZE_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                           void *layers_params, void *opdatas,
                           vector<vector<int>> inputs_,
                           vector<vector<int>> outputs_, int &out_tid,
                           vector<tuple<int, int>> &temp_tensor_ids) {

  QUANTIZE_Data *data = reinterpret_cast<QUANTIZE_Data *>(opdatas);
  TfLiteTensor *output;
  const TfLiteTensor *input;

  GetOutputSafe(context, outputs_[i][0], &output);
  GetInputSafe(context, inputs_[i][0], &input);

  // Currently this only support affine quantization.
  TF_LITE_ENSURE_EQ(context, output->quantization.type,
                    kTfLiteAffineQuantization);

  if (input->type == kTfLiteFloat32) {
    // Quantize use case.
    TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                output->type == kTfLiteInt8 ||
                                output->type == kTfLiteInt16);
  } else {
    // Requantize use case.
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, output->type == kTfLiteInt8 ||
                                  output->type == kTfLiteInt16 ||
                                  output->type == kTfLiteInt32);
    } else if (input->type == kTfLiteInt32) {
      TF_LITE_ENSURE(context, output->type == kTfLiteInt8 ||
                                  output->type == kTfLiteInt16);
    } else {
      TF_LITE_ENSURE(context,
                     input->type == kTfLiteInt8 || input->type == kTfLiteUInt8);
      TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                  output->type == kTfLiteInt8);
    }
    const double effective_output_scale =
        static_cast<double>(input->params.scale) /
        static_cast<double>(output->params.scale);
    tflite::QuantizeMultiplier(effective_output_scale, &data->output_multiplier,
                               &data->output_shift);
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  TfLiteIntArray *output_size = TfLiteIntArrayCopy(input->dims);
  // Creates a temp index to iterate through input data.
  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));

  // Resize output tensor
  context->ResizeTensor(context, output, output_size);

  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  return kTfLiteOk;
}

// tensorflow/lite/kernels/dequantize.cc
bool Prepare_DEQUANTIZE_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                             void *layers_params, void *opdatas,
                             vector<vector<int>> inputs_,
                             vector<vector<int>> outputs_, int &out_tid,
                             vector<tuple<int, int>> &temp_tensor_ids) {

  DEQUANTIZE_Data *data = reinterpret_cast<DEQUANTIZE_Data *>(opdatas);
  DequantizeOpContext op_context(context, i, inputs_, outputs_);

  TF_LITE_ENSURE(context, op_context.input->type == kTfLiteUInt8 ||
                              op_context.input->type == kTfLiteInt8 ||
                              op_context.input->type == kTfLiteInt16 ||
                              op_context.input->type == kTfLiteFloat16);

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
  }
  op_context.output->type = kTfLiteFloat32;
  if (tflite::IsConstantTensor(op_context.input)) {
    op_context.output->allocation_type = kTfLiteArenaRwPersistent;
  }

  TfLiteIntArray *output_size = TfLiteIntArrayCopy(op_context.input->dims);
  // Creates a temp index to iterate through input data.
  // Temporary Output tensor management
  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));

  // Resize output tensor
  context->ResizeTensor(context, op_context.output, output_size);

  // Resize temporary output tensor
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  return kTfLiteOk;
}
#endif