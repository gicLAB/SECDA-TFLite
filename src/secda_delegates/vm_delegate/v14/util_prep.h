#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_PREP_VM_DELEGATE_VM_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_PREP_VM_DELEGATE_VM_DELEGATE_UTIL_H_

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "util.h"

using namespace std;

#define PadKernelMaxDimensionCount 5
const int kMaxConstantOutputTensorSize = 8;

// =========================================================
// Layer Specific Structs
// =========================================================
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

// template <typename T>
// inline void ExtractPatchIntoBufferColumn(const RuntimeShape &input_shape, int
// w,
//                                          int h, int b, int kheight, int
//                                          kwidth, int stride_width, int
//                                          stride_height, int pad_width, int
//                                          pad_height, int in_width, int
//                                          in_height, int in_depth, int
//                                          single_buffer_length, int buffer_id,
//                                          const T *in_data, T
//                                          *conv_buffer_data, uint8 zero_byte)
//                                          {
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   // This chunk of code reshapes all the inputs corresponding to
//   // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
//   const int kwidth_times_indepth = kwidth * in_depth;
//   const int inwidth_times_indepth = in_width * in_depth;
//   const int ih_ungated_start = h * stride_height - pad_height;
//   const int ih_ungated_end = (ih_ungated_start + kheight);
//   const int ih_end = std::min(ih_ungated_end, in_height);
//   const int iw_ungated_start = w * stride_width - pad_width;
//   const int iw_ungated_end = (iw_ungated_start + kwidth);
//   const int iw_end = std::min(iw_ungated_end, in_width);
//   // If the patch is off the edge of the input image, skip writing those rows
//   // and columns from the patch into the output array.
//   const int h_offset = std::max(0, -ih_ungated_start);
//   const int w_offset = std::max(0, -iw_ungated_start);
//   const int ih_start = std::max(0, ih_ungated_start);
//   const int iw_start = std::max(0, iw_ungated_start);
//   const int single_row_num =
//       std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
//   const int output_row_offset = (buffer_id * single_buffer_length);
//   int out_offset =
//       output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
//   int in_offset = Offset(input_shape, b, ih_start, iw_start, 0);

//   // Express all of the calculations as padding around the input patch.
//   const int top_padding = h_offset;
//   const int bottom_padding = (ih_ungated_end - ih_end);
//   const int left_padding = w_offset;
//   const int right_padding = (iw_ungated_end - iw_end);
//   assert(single_row_num ==
//          ((kwidth - (left_padding + right_padding)) * in_depth));

//   // Write out zeroes to the elements representing the top rows of the input
//   // patch that are off the edge of the input image.
//   if (top_padding > 0) {
//     const int top_row_elements = (top_padding * kwidth * in_depth);
//     memset(conv_buffer_data + output_row_offset, zero_byte,
//            (top_row_elements * sizeof(T)));
//   }

//   // If the patch is on the interior of the input image horizontally, just
//   copy
//   // over the rows sequentially, otherwise add zero padding at the start or
//   end. if ((left_padding == 0) && (right_padding == 0)) {
//     for (int ih = ih_start; ih < ih_end; ++ih) {
//       memcpy(conv_buffer_data + out_offset, in_data + in_offset,
//              single_row_num * sizeof(T));
//       out_offset += kwidth_times_indepth;
//       in_offset += inwidth_times_indepth;
//     }
//   } else {
//     for (int ih = ih_start; ih < ih_end; ++ih) {
//       if (left_padding > 0) {
//         const int left_start = (out_offset - (left_padding * in_depth));
//         memset(conv_buffer_data + left_start, zero_byte,
//                (left_padding * in_depth * sizeof(T)));
//       }
//       memcpy(conv_buffer_data + out_offset, in_data + in_offset,
//              single_row_num * sizeof(T));
//       if (right_padding > 0) {
//         const int right_start = (out_offset + single_row_num);
//         memset(conv_buffer_data + right_start, zero_byte,
//                (right_padding * in_depth * sizeof(T)));
//       }
//       out_offset += kwidth_times_indepth;
//       in_offset += inwidth_times_indepth;
//     }
//   }

//   // If the bottom of the patch falls off the input image, pad the values
//   // representing those input rows with zeroes.
//   if (bottom_padding > 0) {
//     const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
//     const int bottom_start =
//         output_row_offset +
//         ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
//     memset(conv_buffer_data + bottom_start, zero_byte,
//            (bottom_row_elements * sizeof(T)));
//   }
// }

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
  // Loop over the output nodes.
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

// Supports per-batch zero_byte for per-batch asymmetric quantized inputs.
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

  // For dilated convolution, the input pixels are not contiguous therefore we
  // can't use the same optimizations as Im2Col(). Though note this code would
  // work fine for the non-dilated case too (though likely a bit slower).
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

  // Construct the MxN sized im2col matrix.
  // The rows M, are sub-ordered B x H x W
  const RuntimeShape row_shape({1, batches, output_height, output_width});
  // The columns, N, are sub-ordered Kh x Kw x Din
  const RuntimeShape col_shape({1, filter_height, filter_width, input_depth});
  // Use dimensions M and N to construct dims for indexing directly into im2col
  const RuntimeShape im2col_shape(
      {1, 1, row_shape.FlatSize(), col_shape.FlatSize()});

  // Loop through the output rows (B x H x W)
  for (int batch = 0; batch < batches; ++batch) {
    const T zero_byte = zero_bytes_len > 1 ? static_cast<T>(zero_bytes[batch])
                                           : static_cast<T>(zero_bytes[0]);
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        // Each im2col row is an output pixel. Arrange the input data in this
        // row in an order we can conveniently multiply with the filter data.
        int row_offset = Offset(row_shape, 0, batch, out_y, out_x);
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        // Loop through all the pixels of the filter (Kh x Kw)
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          if ((in_y >= 0) && (in_y < input_height)) {
            // Filter row is within the input data.
            // Loop through all the filter pixels in this row.
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              int col_offset = Offset(col_shape, 0, filter_y, filter_x, 0);
              T *dst = im2col_data +
                       Offset(im2col_shape, 0, 0, row_offset, col_offset);
              if ((in_x >= 0) && (in_x < input_width)) {
                // Filter pixel is within the input, copy the input data.
                T const *src =
                    input_data + Offset(input_shape, batch, in_y, in_x, 0);
                memcpy(dst, src, input_depth * sizeof(T));
              } else {
                // Filter pixel is outside the input, zero it out.
                memset(dst, zero_byte, input_depth * sizeof(T));
              }
            }
          } else {
            // Filter row is outside the input, zero out the entire filter row.
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

static TfLiteStatus
AllocateTemporaryOutTensorsIfRequired(TfLiteContext *context, TfLiteNode *node,
                                      bool req_temp_out, int temp_out_tid,
                                      int &temp_out_id) {
  int temporaries_count = node->temporaries->size;
  AddTempOutTensor(context, node, req_temp_out, temporaries_count, temp_out_tid,
                   temp_out_id);
  return UpdateTempTensors(node, temporaries_count);
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

void prepare_weights(int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4, int inpZeroPoint,
                     std::vector<int> *bias) {
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
        int8_t weights[] = {w3, w2, w1, w0};
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
    // check bias is nullpointer or not
    if (bias == nullptr) {
      wt_sum1.push_back(s0 * inpZeroPoint);
      wt_sum2.push_back(s1 * inpZeroPoint);
      wt_sum3.push_back(s2 * inpZeroPoint);
      wt_sum4.push_back(s3 * inpZeroPoint);
    } else {
      wt_sum1.push_back((s0 * inpZeroPoint) + (*bias)[(i * 4) + 0]);
      wt_sum2.push_back((s1 * inpZeroPoint) + (*bias)[(i * 4) + 1]);
      wt_sum3.push_back((s2 * inpZeroPoint) + (*bias)[(i * 4) + 2]);
      wt_sum4.push_back((s3 * inpZeroPoint) + (*bias)[(i * 4) + 3]);
    }
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

// tensorflow/lite/kernels/conv.cc
bool Prepare_CONV2D_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                         void *layers_params, void *opdatas,
                         vector<vector<int>> inputs_,
                         vector<vector<int>> outputs_, int &out_tid,
                         vector<int8_t> &temp_im2col, vector<int8_t> &wb0,
                         vector<int8_t> &wb1, vector<int8_t> &wb2,
                         vector<int8_t> &wb3, vector<int> &wt_sum1,
                         vector<int> &wt_sum2, vector<int> &wt_sum3,
                         vector<int> &wt_sum4, vector<int> &biases,
                         vector<int> &crf, vector<int8_t> &crx) {

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
  // precal_wsum(filter, wt_sum);

  // copy crf and crx
  crf.resize(channels_out);
  crx.resize(channels_out);

  crf.assign(data->per_channel_output_multiplier.begin(),
             data->per_channel_output_multiplier.end());
  for (int j = 0; j < channels_out; j++) {
    crx[j] = 31 - data->per_channel_output_shift.data()[j];
  }

  // copy bias to biases vector
  biases.resize(channels_out);
  biases.assign(bias->data.i32, bias->data.i32 + channels_out);
  int *dims = filter->dims->data;
  int inpZeroPoint = -input->params.zero_point;
  prepare_weights(filter->data.int8, dims, wb0, wb1, wb2, wb3, wt_sum1, wt_sum2,
                  wt_sum3, wt_sum4, inpZeroPoint, &biases);

  return kTfLiteOk;
}

// tensorflow/lite/kernels/fully_connected.cc
bool Prepare_FC_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                     void *layers_params, void *opdatas,
                     vector<vector<int>> inputs_, vector<vector<int>> outputs_,
                     int &out_tid, vector<int8_t> &wb0, vector<int8_t> &wb1,
                     vector<int8_t> &wb2, vector<int8_t> &wb3,
                     vector<int> &wt_sum1, vector<int> &wt_sum2,
                     vector<int> &wt_sum3, vector<int> &wt_sum4,
                     vector<int> &biases, vector<int> &crf,
                     vector<int8_t> &crx) {
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

  // Resize output. -- why resizing output
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
  //////////////////////////////////////////////////////////////
  // Accelerator specific optimisations
  // precal_wsum(filter, wt_sum);

  // copy crf and crx
  // if it is per_channel_quantization copy from
  // per_channel_multiplier and per_channel_shift
  // otherwise
  // data->output_multiplier
  // data->output_shift
  int channels_out = filter->dims->data[0];
  if (is_per_channel) {
    if (channels_out != per_channel_quantization_size) {
      cout << "Prepare_FC_INT8(): channel size mismatch" << endl;
      return kTfLiteError;
    }
    crf.resize(per_channel_quantization_size);
    crx.resize(per_channel_quantization_size);

    crf.assign(data->per_channel_output_multiplier.begin(),
               data->per_channel_output_multiplier.end());
    for (int j = 0; j < per_channel_quantization_size; j++) {
      crx[j] = 31 - data->per_channel_output_shift.data()[j];
    }
  } else {
    crf.resize(channels_out);
    crx.resize(channels_out);
    for (int j = 0; j < channels_out; j++) {
      crf[j] = data->output_multiplier;
      crx[j] = 31 - data->output_shift;
    }
  }

  // pass the bias based on checking it is null pointer of not
  // create dim array
  int *dims = new int[4];
  dims[0] = filter->dims->data[0];
  dims[1] = 1;
  dims[2] = 1;
  dims[3] = filter->dims->data[1];

  int inpZeroPoint = -input->params.zero_point;
  if (isBias) {
    biases.resize(channels_out);
    biases.assign(bias->data.i32, bias->data.i32 + channels_out);
    prepare_weights(filter->data.int8, dims, wb0, wb1, wb2, wb3, wt_sum1,
                    wt_sum2, wt_sum3, wt_sum4, inpZeroPoint,
                    &biases); // pass address
  } else {
    prepare_weights(filter->data.int8, dims, wb0, wb1, wb2, wb3, wt_sum1,
                    wt_sum2, wt_sum3, wt_sum4, inpZeroPoint,
                    nullptr); // pass no bias
  }
  return kTfLiteOk;
}

#endif