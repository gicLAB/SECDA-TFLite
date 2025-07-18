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