#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_BERT_SIM_DELEGATE_BERT_SIM_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_BERT_SIM_DELEGATE_BERT_SIM_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/fully_connected_4bit.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"

#include <cassert>

using namespace std;

inline const char *const *EnumNamesBuiltinOperator() {
  static const char *const names[207] = {"ADD",
                                         "AVERAGE_POOL_2D",
                                         "CONCATENATION",
                                         "CONV_2D",
                                         "DEPTHWISE_CONV_2D",
                                         "DEPTH_TO_SPACE",
                                         "DEQUANTIZE",
                                         "EMBEDDING_LOOKUP",
                                         "FLOOR",
                                         "FULLY_CONNECTED",
                                         "HASHTABLE_LOOKUP",
                                         "L2_NORMALIZATION",
                                         "L2_POOL_2D",
                                         "LOCAL_RESPONSE_NORMALIZATION",
                                         "LOGISTIC",
                                         "LSH_PROJECTION",
                                         "LSTM",
                                         "MAX_POOL_2D",
                                         "MUL",
                                         "RELU",
                                         "RELU_N1_TO_1",
                                         "RELU6",
                                         "RESHAPE",
                                         "RESIZE_BILINEAR",
                                         "RNN",
                                         "SOFTMAX",
                                         "SPACE_TO_DEPTH",
                                         "SVDF",
                                         "TANH",
                                         "CONCAT_EMBEDDINGS",
                                         "SKIP_GRAM",
                                         "CALL",
                                         "CUSTOM",
                                         "EMBEDDING_LOOKUP_SPARSE",
                                         "PAD",
                                         "UNIDIRECTIONAL_SEQUENCE_RNN",
                                         "GATHER",
                                         "BATCH_TO_SPACE_ND",
                                         "SPACE_TO_BATCH_ND",
                                         "TRANSPOSE",
                                         "MEAN",
                                         "SUB",
                                         "DIV",
                                         "SQUEEZE",
                                         "UNIDIRECTIONAL_SEQUENCE_LSTM",
                                         "STRIDED_SLICE",
                                         "BIDIRECTIONAL_SEQUENCE_RNN",
                                         "EXP",
                                         "TOPK_V2",
                                         "SPLIT",
                                         "LOG_SOFTMAX",
                                         "DELEGATE",
                                         "BIDIRECTIONAL_SEQUENCE_LSTM",
                                         "CAST",
                                         "PRELU",
                                         "MAXIMUM",
                                         "ARG_MAX",
                                         "MINIMUM",
                                         "LESS",
                                         "NEG",
                                         "PADV2",
                                         "GREATER",
                                         "GREATER_EQUAL",
                                         "LESS_EQUAL",
                                         "SELECT",
                                         "SLICE",
                                         "SIN",
                                         "TRANSPOSE_CONV",
                                         "SPARSE_TO_DENSE",
                                         "TILE",
                                         "EXPAND_DIMS",
                                         "EQUAL",
                                         "NOT_EQUAL",
                                         "LOG",
                                         "SUM",
                                         "SQRT",
                                         "RSQRT",
                                         "SHAPE",
                                         "POW",
                                         "ARG_MIN",
                                         "FAKE_QUANT",
                                         "REDUCE_PROD",
                                         "REDUCE_MAX",
                                         "PACK",
                                         "LOGICAL_OR",
                                         "ONE_HOT",
                                         "LOGICAL_AND",
                                         "LOGICAL_NOT",
                                         "UNPACK",
                                         "REDUCE_MIN",
                                         "FLOOR_DIV",
                                         "REDUCE_ANY",
                                         "SQUARE",
                                         "ZEROS_LIKE",
                                         "FILL",
                                         "FLOOR_MOD",
                                         "RANGE",
                                         "RESIZE_NEAREST_NEIGHBOR",
                                         "LEAKY_RELU",
                                         "SQUARED_DIFFERENCE",
                                         "MIRROR_PAD",
                                         "ABS",
                                         "SPLIT_V",
                                         "UNIQUE",
                                         "CEIL",
                                         "REVERSE_V2",
                                         "ADD_N",
                                         "GATHER_ND",
                                         "COS",
                                         "WHERE",
                                         "RANK",
                                         "ELU",
                                         "REVERSE_SEQUENCE",
                                         "MATRIX_DIAG",
                                         "QUANTIZE",
                                         "MATRIX_SET_DIAG",
                                         "ROUND",
                                         "HARD_SWISH",
                                         "IF",
                                         "WHILE",
                                         "NON_MAX_SUPPRESSION_V4",
                                         "NON_MAX_SUPPRESSION_V5",
                                         "SCATTER_ND",
                                         "SELECT_V2",
                                         "DENSIFY",
                                         "SEGMENT_SUM",
                                         "BATCH_MATMUL",
                                         "PLACEHOLDER_FOR_GREATER_OP_CODES",
                                         "CUMSUM",
                                         "CALL_ONCE",
                                         "BROADCAST_TO",
                                         "RFFT2D",
                                         "CONV_3D",
                                         "IMAG",
                                         "REAL",
                                         "COMPLEX_ABS",
                                         "HASHTABLE",
                                         "HASHTABLE_FIND",
                                         "HASHTABLE_IMPORT",
                                         "HASHTABLE_SIZE",
                                         "REDUCE_ALL",
                                         "CONV_3D_TRANSPOSE",
                                         "VAR_HANDLE",
                                         "READ_VARIABLE",
                                         "ASSIGN_VARIABLE",
                                         "BROADCAST_ARGS",
                                         "RANDOM_STANDARD_NORMAL",
                                         "BUCKETIZE",
                                         "RANDOM_UNIFORM",
                                         "MULTINOMIAL",
                                         "GELU",
                                         "DYNAMIC_UPDATE_SLICE",
                                         "RELU_0_TO_1",
                                         "UNSORTED_SEGMENT_PROD",
                                         "UNSORTED_SEGMENT_MAX",
                                         "UNSORTED_SEGMENT_SUM",
                                         "ATAN2",
                                         "UNSORTED_SEGMENT_MIN",
                                         "SIGN",
                                         "BITCAST",
                                         "BITWISE_XOR",
                                         "RIGHT_SHIFT",
                                         "STABLEHLO_LOGISTIC",
                                         "STABLEHLO_ADD",
                                         "STABLEHLO_DIVIDE",
                                         "STABLEHLO_MULTIPLY",
                                         "STABLEHLO_MAXIMUM",
                                         "STABLEHLO_RESHAPE",
                                         "STABLEHLO_CLAMP",
                                         "STABLEHLO_CONCATENATE",
                                         "STABLEHLO_BROADCAST_IN_DIM",
                                         "STABLEHLO_CONVOLUTION",
                                         "STABLEHLO_SLICE",
                                         "STABLEHLO_CUSTOM_CALL",
                                         "STABLEHLO_REDUCE",
                                         "STABLEHLO_ABS",
                                         "STABLEHLO_AND",
                                         "STABLEHLO_COSINE",
                                         "STABLEHLO_EXPONENTIAL",
                                         "STABLEHLO_FLOOR",
                                         "STABLEHLO_LOG",
                                         "STABLEHLO_MINIMUM",
                                         "STABLEHLO_NEGATE",
                                         "STABLEHLO_OR",
                                         "STABLEHLO_POWER",
                                         "STABLEHLO_REMAINDER",
                                         "STABLEHLO_RSQRT",
                                         "STABLEHLO_SELECT",
                                         "STABLEHLO_SUBTRACT",
                                         "STABLEHLO_TANH",
                                         "STABLEHLO_SCATTER",
                                         "STABLEHLO_COMPARE",
                                         "STABLEHLO_CONVERT",
                                         "STABLEHLO_DYNAMIC_SLICE",
                                         "STABLEHLO_DYNAMIC_UPDATE_SLICE",
                                         "STABLEHLO_PAD",
                                         "STABLEHLO_IOTA",
                                         "STABLEHLO_DOT_GENERAL",
                                         "STABLEHLO_REDUCE_WINDOW",
                                         "STABLEHLO_SORT",
                                         "STABLEHLO_WHILE",
                                         "STABLEHLO_GATHER",
                                         "STABLEHLO_TRANSPOSE",
                                         "DILATE",
                                         "STABLEHLO_RNG_BIT_GENERATOR",
                                         "REDUCE_WINDOW",
                                         nullptr};
  return names;
}

enum QuantizeKernelType {
  kReference,
  kGenericOptimized,
};


const int kTensorNotAllocated = -1;
static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024; // 1GB
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

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;

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

int Generic_Quantised_Multiplier(int x, int qm, int shift) {
#define MAX32 2147483647
#define MIN32 -2147483648

  int nshift = shift;
  int total_shift = 31 - shift;
  int64_t x_64 = x;
  int64_t quantized_multiplier_64(qm);
  int64_t one = 1;
  int64_t round = one << (total_shift - 1);                // ALU ADD + ALU SHLI
  int64_t result = x_64 * quantized_multiplier_64 + round; // ALU ADD + ALU MUL
  result = result >> total_shift;                          // ALU SHRI
  int nresult = result;
  if (result > MAX32) result = MAX32; // ALU MIN
  if (result < MIN32) result = MIN32; // ALU MAX
  return static_cast<std::int32_t>(result);

#undef MAX32
#undef MIN32
}

void precal_sums(int8_t *data, int width, int depth, vector<int> &sums) {
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
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
            (i * (depth * 4) + j >= max) ? 0 : data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : data[i * (depth * 4) + j + depth * 3];
        int8_t g_data[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
      }
    }
    sums.push_back(s0);
    sums.push_back(s1);
    sums.push_back(s2);
    sums.push_back(s3);
  }
}

static TfLiteStatus AllocateTemporaryTensorsIfRequired(
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

bool IsNode_FC_INT8(const TfLiteRegistration *registration,
                    const TfLiteNode *node, TfLiteContext *context) {
  // Only supports FC ops
  if (kTfLiteBuiltinFullyConnected != registration->builtin_code) return false;

  // In this delegate implementation, FC N maps to the batch dimension.
  // const auto &output_tensor = context->tensors[node->outputs->data[0]];
  // if (output_tensor.dims != nullptr && output_tensor.dims->size > 0 &&
  //     output_tensor.dims->data[0] == 1) return false;

  if (node->inputs->size != 3 && node->inputs->size != 2) return false;
  // This delegate only supports int8 types.
  for (int i = 0; i < 2; ++i) {
    auto &tensor = context->tensors[node->inputs->data[i]];
    if (tensor.type != kTfLiteInt8) return false;
  }

  if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
    // Add this return: false back if you dont want FC with biases
    // return false;
    auto &tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32 && tensor.type <= 16) return false;
  }

  return true;
}

bool IsNode_CONV2D_INT8(const TfLiteRegistration *registration,
                        const TfLiteNode *node, TfLiteContext *context) {
  // Only supports CONV2D ops
  if (kTfLiteBuiltinConv2d != registration->builtin_code) return false;

  Conv2D_Data *data = reinterpret_cast<Conv2D_Data *>(node->user_data);
  TfLiteConvParams *params =
      reinterpret_cast<TfLiteConvParams *>(node->builtin_data);
  // if (params->padding == kTfLitePaddingValid) return false;
  // if (params->activation != kTfLiteActNone) return false;
  // if (params->activation != kTfLiteActNone) {
  //   cout << node->outputs->data[0] << ",\n";
  // }

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

#endif