#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_PREP_VM_DELEGATE_VIT_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_PREP_VM_DELEGATE_VIT_DELEGATE_UTIL_H_

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "util.h"

using namespace std;

void precal_sum_load_padv3_vectorized(int8_t *data, int width, int depth,
                                      int8_t *shape_data,
                                      std::vector<int> &sums,
                                      int w_pad = 64, int d_pad = 64);

#define PadKernelMaxDimensionCount 5
const int kMaxConstantOutputTensorSize = 8;

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

bool IsIm2ColRequired(const TfLiteTensor *input, TfLiteConvParams *params,
                      const TfLiteTensor *filter, Conv2D_Data *data,
                      bool is_hybrid) {
  if (data->need_hwcn_weights) return false;

  const bool need_dilated_im2col =
      params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
  const bool need_non_dilated_im2col =
      params->stride_width != 1 || params->stride_height != 1 ||
      filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

  const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

  if (!need_im2col) return false;

  const bool is_hybrid_non_dilated = is_hybrid && need_non_dilated_im2col;
  const bool is_quantized =
      input->type == kTfLiteUInt8 || input->type == kTfLiteInt8;

  if (is_hybrid && !need_non_dilated_im2col) {
    return false;
  } else {
    return true;
  }
}

namespace tflite {
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

template <typename T>
void Im2col(ConvParams &params, int kheight, int kwidth, uint8 zero_byte,
            const RuntimeShape &input_shape, const T *input_data,
            const RuntimeShape &output_shape, T *output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  int buffer_id = 0;
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        tflite::optimized_ops::ExtractPatchIntoBufferColumn(
            input_shape, w, h, b, kheight, kwidth, stride_width, stride_height,
            pad_width, pad_height, input_width, input_height, input_depth,
            output_depth, buffer_id, input_data, output_data, zero_byte);
        ++buffer_id;
      }
    }
  }
}

template <typename T>
void DilatedIm2col(ConvParams &params, const RuntimeShape &input_shape,
                   const T *input_data, const RuntimeShape &filter_shape,
                   const RuntimeShape &output_shape, T *im2col_data,
                   const int32_t *zero_bytes, const int zero_bytes_len) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK(dilation_width_factor != 1 || dilation_height_factor != 1);
  TFLITE_DCHECK(im2col_data);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  MatchingDim(output_shape, 3, filter_shape, 0);

  const RuntimeShape row_shape({1, batches, output_height, output_width});
  const RuntimeShape col_shape({1, filter_height, filter_width, input_depth});
  const RuntimeShape im2col_shape(
      {1, 1, row_shape.FlatSize(), col_shape.FlatSize()});

  for (int batch = 0; batch < batches; ++batch) {
    const T zero_byte = zero_bytes_len > 1 ? static_cast<T>(zero_bytes[batch])
                                           : static_cast<T>(zero_bytes[0]);
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        int row_offset = Offset(row_shape, 0, batch, out_y, out_x);
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          if ((in_y >= 0) && (in_y < input_height)) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              int col_offset = Offset(col_shape, 0, filter_y, filter_x, 0);
              T *dst = im2col_data +
                       Offset(im2col_shape, 0, 0, row_offset, col_offset);
              if ((in_x >= 0) && (in_x < input_width)) {
                T const *src =
                    input_data + Offset(input_shape, batch, in_y, in_x, 0);
                memcpy(dst, src, input_depth * sizeof(T));
              } else {
                memset(dst, zero_byte, input_depth * sizeof(T));
              }
            }
          } else {
            int col_offset = Offset(col_shape, 0, filter_y, 0, 0);
            T *dst = im2col_data +
                     Offset(im2col_shape, 0, 0, row_offset, col_offset);
            memset(dst, zero_byte, filter_width * input_depth * sizeof(T));
          }
        }
      }
    }
  }
}

template <typename T>
void DilatedIm2col(ConvParams &params, uint8 zero_byte,
                   const RuntimeShape &input_shape, const T *input_data,
                   const RuntimeShape &filter_shape,
                   const RuntimeShape &output_shape, T *im2col_data) {
  const int32_t zero_point = static_cast<int32_t>(zero_byte);
  DilatedIm2col<T>(params, input_shape, input_data, filter_shape, output_shape,
                   im2col_data, &zero_point, 1);
}
} 

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

TfLiteStatus ResizeTempAxis(TfLiteContext *context, ReduceOpContext *op_context,
                            TfLiteTensor *resolved_axis) {
  TfLiteIntArray *axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = static_cast<int>(tflite::NumElements(op_context->axis));
  return context->ResizeTensor(context, resolved_axis, axis_size);
}

TfLiteStatus ResizeTempDims(TfLiteContext *context, ReduceOpContext *op_context,
                            TfLiteTensor *normalized_dims) {
  TfLiteIntArray *dims_size = TfLiteIntArrayCreate(1);
  dims_size->data[0] = (op_context->input->dims->size);
  return context->ResizeTensor(context, normalized_dims, dims_size);
}

TfLiteStatus ResizeOutputTensor(TfLiteContext *context,
                                ReduceOpContext *op_context) {
  TfLiteIntArray *output_dims;
  TF_LITE_ENSURE_OK(context,
                    GetReduceOutputShape(context, op_context, &output_dims));
  return context->ResizeTensor(context, op_context->output, output_dims);
}

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

static TfLiteStatus
AllocateTemporaryOutTensorsIfRequired(TfLiteContext *context, TfLiteNode *node,
                                      bool req_temp_out, int temp_out_tid,
                                      int &temp_out_id) {
  int temporaries_count = node->temporaries->size;
  AddTempOutTensor(context, node, req_temp_out, temporaries_count, temp_out_tid,
                   temp_out_id);
  return UpdateTempTensors(node, temporaries_count);
}

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

void preload_weights(int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4, int inpZeroPoint, int *bias) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
      } else {
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
      }
    }
    if (bias == nullptr) {
      wt_sum1.push_back(s0 * inpZeroPoint);
      wt_sum2.push_back(s1 * inpZeroPoint);
      wt_sum3.push_back(s2 * inpZeroPoint);
      wt_sum4.push_back(s3 * inpZeroPoint);
    } else {
      wt_sum1.push_back((s0 * inpZeroPoint) + bias[(i * 4) + 0]);
      wt_sum2.push_back((s1 * inpZeroPoint) + bias[(i * 4) + 1]);
      wt_sum3.push_back((s2 * inpZeroPoint) + bias[(i * 4) + 2]);
      wt_sum4.push_back((s3 * inpZeroPoint) + bias[(i * 4) + 3]);
    }
  }
}

void prepare_weights(int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4, int inpZeroPoint,
                     std::vector<int> *bias) {
  int *bias_ptr = bias ? bias->data() : nullptr;
  preload_weights(weight_data, dims, wb0, wb1, wb2, wb3, wt_sum1, wt_sum2,
                  wt_sum3, wt_sum4, inpZeroPoint, bias_ptr);
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
  if (result > std::numeric_limits<int32_t>::max())
    result = std::numeric_limits<int32_t>::max();
  if (result < std::numeric_limits<int32_t>::min())
    result = std::numeric_limits<int32_t>::min();
  int32_t result_32 = result;
  result_32 += out_offset;
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
}

void precal_sum_load_padv3_vectorized(int8_t *data, int width, int depth,
                                      int8_t *shape_data,
                                      std::vector<int> &sums,
                                      int w_pad, int d_pad) {
#ifdef ACC_NEON
  int w = roundUp(width, w_pad);
  int d = roundUp(depth, d_pad);
  int i_c = 0;
  for (int i = 0; i < w; i++) {
    int s0 = 0;
    if (i < width) {
      for (int j = 0; j < d; j += 16) {
        if (j < depth) {
          int8x16_t val;
          int8_t val_arr[16];
          val = vld1q_s8(data + (i * depth) + j);
          vst1q_s8(val_arr, val);
          for (int k = 0; k < 16; k++) {
            // Avoid ingesting bytes past valid depth; zero-fill tail.
            if ((j + k) < depth) {
              s0 += val_arr[k];
              shape_data[i_c++] = val_arr[k];
            } else {
              shape_data[i_c++] = 0;
            }
          }
        } else {
          for (int k = 0; k < 16; k++) {
            shape_data[i_c++] = 0;
          }
        }
      }
    } else {
      for (int j = 0; j < d; j++) {
        shape_data[i_c++] = 0;
      }
    }
    sums.push_back(s0);
  }
#else
  precal_sum_load_pad(data, width, depth, shape_data, sums, w_pad, d_pad);
#endif
}

#endif