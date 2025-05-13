#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_PREP_OMNI_DELEGATE_OMNI_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_PREP_OMNI_DELEGATE_OMNI_DELEGATE_UTIL_H_

#include "util.h"
using namespace std;

// =========================================================
// Layer Specfifc Helper functions
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

static TfLiteStatus AllocateTemporaryTensorsIfRequiredFC(
    TfLiteContext *context, TfLiteNode *node, bool req_temp_out,
    int temp_out_tid, int &temp_out_id, int input_tid, int filter_tid) {

  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  const TfLiteTensor *input;
  const TfLiteTensor *filter;

  GetInputSafe(context, input_tid, &input);
  GetInputSafe(context, filter_tid, &filter);
  int temporaries_count = node->temporaries->size;

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

int Quantised_Multiplier_S(int x, int qm, int shift, int out_offset,
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

// =========================================================
// Prepare function for all supported Ops
// =========================================================

// ADD INT8 Prepare
bool Prepare_ADD_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                      void *layers_params, void *opdatas,
                      vector<vector<int>> inputs_,
                      vector<vector<int>> outputs_) {

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
  context->ResizeTensor(context, output, output_size);
  return kTfLiteOk;
}

// CONV2D INT8 Prepare
bool Prepare_CONV2D_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                         void *layers_params, void *opdatas,
                         vector<vector<int>> inputs_,
                         vector<vector<int>> outputs_, int out_tid,
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
  int temp_o_id;
  int oi = outputs_[i][0];
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;

  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequiredCONV2D(
      context, node, is_hybrid, data->is_hybrid_per_channel, im2col_bytes,
      params, data, req_temp_out, outputs_[i][0], temp_o_id, inputs_[i][0],
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

  if (req_temp_out) {
    node->temporaries->data[temp_o_id] = outputs_[i][0];
    TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
    temp_out_tensor->type = kTfLiteInt8;
    temp_out_tensor->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray *temp_out_tensor_size = TfLiteIntArrayCreate(4);
    temp_out_tensor_size->data[0] = output_size->data[0];
    temp_out_tensor_size->data[1] = output_size->data[1];
    temp_out_tensor_size->data[2] = output_size->data[2];
    temp_out_tensor_size->data[3] = output_size->data[3];
    auto temp_out_tensor_status =
        context->ResizeTensor(context, temp_out_tensor, temp_out_tensor_size);
    if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
  }

  // Accelerator specific optimisations
  precal_wsum(filter, wt_sum);
  return kTfLiteOk;
}

bool Prepare_FC_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                     void *layers_params, void *opdatas,
                     vector<vector<int>> inputs_, vector<vector<int>> outputs_,
                     int out_tid, vector<int> &wt_sum) {
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
  // double real_multiplier = 0.0;
  // int exponent;
  // tflite::GetQuantizedConvolutionMultipler(context, input, filter, bias,
  // output,
  //                                          &real_multiplier);
  // tflite::QuantizeMultiplier(real_multiplier, &data->output_multiplier,
  //                            &data->output_shift);
  // tflite::CalculateActivationRangeQuantized(context, params->activation,
  // output,
  //                                           &data->output_activation_min,
  //                                           &data->output_activation_max);

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

  // TfLiteIntArray *output_size = TfLiteIntArrayCreate(2);
  // output_size->data[0] = out_dim1;
  // output_size->data[1] = out_dim2;
  // auto output_status = context->ResizeTensor(context, output, output_size);
  // if (output_status != kTfLiteOk) return output_status;

  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;

  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequiredFC(
      context, node, req_temp_out, outputs_[i][0], temp_out_id, inputs_[i][0],
      inputs_[i][1]));

  if (req_temp_out) {
    node->temporaries->data[temp_out_id] = outputs_[i][0];
    TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
    temp_out_tensor->type = kTfLiteInt8;
    temp_out_tensor->allocation_type = kTfLiteArenaRw;
    auto temp_out_tensor_status =
        context->ResizeTensor(context, temp_out_tensor, output_size);
    if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
  }

  // Accelerator specific optimisations
  int *dims = filter->dims->data;
  int dims_size = filter->dims->size;
  precal_wsum(filter, wt_sum);
  return kTfLiteOk;
}
#endif