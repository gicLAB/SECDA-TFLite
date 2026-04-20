#include "tensorflow/lite/delegates/utils/secda_delegates/vit_delegate/v9/vit_delegate.h"
#include <fstream>
#include <iostream>
#include <utility>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif

#include "accelerator/driver/vit_driver.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "secda_tools/secda_utils/acc_helpers.h"
#include "secda_tools/secda_utils/multi_threading.h"
#include "secda_tools/secda_utils/utils.h"
#include "util.h"
#include "util_prep.h"
#include "vit_delegate.h"

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#define DMA_BC 1

#ifndef VIT_DELEGATE_VERBOSE
#define VIT_DELEGATE_VERBOSE 1
#endif

bool Prepare_CONV2D_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                         void *layers_params, void *opdatas,
                         vector<vector<int>> &inputs_,
                         vector<vector<int>> &outputs_, int &out_tid,
                         vector<int8_t> &temp_im2col, vector<int8_t> &wb0,
                         vector<int8_t> &wb1, vector<int8_t> &wb2,
                         vector<int8_t> &wb3, vector<int> &wt_sum1,
                         vector<int> &wt_sum2, vector<int> &wt_sum3,
                         vector<int> &wt_sum4, vector<int> &biases,
                         vector<int> &crf, vector<int8_t> &crx,
                         std::vector<int8_t> &preloaded_weights,
                         std::vector<int> &preloaded_wt_sum);

bool Prepare_FC_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                     void *layers_params, void *opdatas,
                     vector<vector<int>> &inputs_, vector<vector<int>> &outputs_,
                     int &out_tid, vector<int8_t> &wb0, vector<int8_t> &wb1,
                     vector<int8_t> &wb2, vector<int8_t> &wb3,
                     vector<int> &wt_sum1, vector<int> &wt_sum2,
                     vector<int> &wt_sum3, vector<int> &wt_sum4,
                     vector<int> &biases, vector<int> &crf,
                     vector<int8_t> &crx, std::vector<int8_t> &preloaded_weights,
                     std::vector<int> &preloaded_wt_sum);

unsigned int dma_addrs[4] = {dma_addr0, dma_addr1, dma_addr2, dma_addr3};
unsigned int dma_addrs_in[4] = {dma_in0, dma_in1, dma_in2, dma_in3};
unsigned int dma_addrs_out[4] = {dma_out0, dma_out1, dma_out2, dma_out3};
struct acc_times a_t;
struct del_params dparams;
static struct Profile profile;
struct MultiThreadContext mt_context;
struct layer_details t;

#ifdef SYSC
struct sysC_sigs *scs;
#define SYSC_DMA_BL 563840 * 4
static struct s_mdma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                          SYSC_DMA_BL);
ACCNAME *acc;
struct dma_buffer_set dfs[4] = {
    {DMA_BC, (SYSC_DMA_BL / DMA_BC), dma_in0},
    {DMA_BC, (SYSC_DMA_BL / DMA_BC), dma_in1},
    {DMA_BC, (SYSC_DMA_BL / DMA_BC), dma_in2},
    {DMA_BC, (SYSC_DMA_BL / DMA_BC), dma_in3},
};
int recv_len = (SYSC_DMA_BL / DMA_BC);
#else
static struct s_mdma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out, DMA_BL);
int *acc;
struct dma_buffer_set dfs[4] = {
    {DMA_BC, (DMA_BL / DMA_BC), dma_in0},
    {DMA_BC, (DMA_BL / DMA_BC), dma_in1},
    {DMA_BC, (DMA_BL / DMA_BC), dma_in2},
    {DMA_BC, (DMA_BL / DMA_BC), dma_in3},
};
int recv_len = (DMA_BL / DMA_BC);
#endif

int Quantised_Multiplier_S(int x, int qm, int shift) {
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
  return result_32;
}

namespace tflite {
namespace vit_test {

class VitDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit VitDelegateKernel(const VitDelegateOptions &options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {
    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      // std::cout << "===========================" << std::endl;
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("VIT_ACC");
      sysC_init();
      sysC_binder(&_acc, &mdma, &scs1);
      acc = &_acc;
      scs = &scs1;
      std::cout << "Initialised the SystemC Modules" << std::endl;
#else
      dparams.acc = getAccBaseAddress<int>(acc_address, 65536);
      acc = dparams.acc;
      std::cout << "Initialised the DMA" << std::endl;
#endif
      std::cout << "VIT_ACC Accelerator v8";
#ifdef ACC_NEON
      std::cout << " with Neon";
#endif
      std::cout << std::endl;
      std::cout << "===========================" << std::endl;
      dparams.init = true;
    }

    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);

    biases.resize(params->nodes_to_replace->size);
    crf_arr.resize(params->nodes_to_replace->size);
    crx_arr.resize(params->nodes_to_replace->size);
    wb0.resize(params->nodes_to_replace->size);
    wb1.resize(params->nodes_to_replace->size);
    wb2.resize(params->nodes_to_replace->size);
    wb3.resize(params->nodes_to_replace->size);
    wt_sum1.resize(params->nodes_to_replace->size);
    wt_sum2.resize(params->nodes_to_replace->size);
    wt_sum3.resize(params->nodes_to_replace->size);
    wt_sum4.resize(params->nodes_to_replace->size);
    temp_im2col.resize(params->nodes_to_replace->size);
    // MOD: per-node cached weights/sums for Eval reuse.
    preloaded_weights.resize(params->nodes_to_replace->size);
    preloaded_weight_sums.resize(params->nodes_to_replace->size);

    opdatas.resize(params->nodes_to_replace->size);
    layers_params.resize(params->nodes_to_replace->size);
    is_global_output.resize(params->nodes_to_replace->size);
    output_dependencies.resize(params->nodes_to_replace->size);
    node_output_needed.resize(params->nodes_to_replace->size);

    int conv2d_count = 0;
    int fc_count = 0;

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode *delegated_node = nullptr;
      TfLiteRegistration *delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      for (int j = 0; j < delegated_node->inputs->size; j++)
        inputs_[i].push_back(delegated_node->inputs->data[j]);

      for (int j = 0; j < delegated_node->outputs->size; j++)
        outputs_[i].push_back(delegated_node->outputs->data[j]);

      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);
      layers_params[i] = delegated_node->builtin_data;
      opdatas[i] = delegated_node->user_data;

      if (builtin_code_[i] == kTfLiteBuiltinConv2d) conv2d_count++;
      if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) fc_count++;
    }
    return kTfLiteOk;
  }
  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) override {
    int node_count = inputs_.size();
    int out_tid = 0;
    prf_start(3);

    for (int i = 0; i < node_count; i++) {

      // =======================================================
      // Tracking Output dependencies
      // =======================================================
      int current_output = outputs_[i][0];
      bool isOG = false;
      vector<int> future_nodes;

      for (int j = 0; j < node->outputs->size; j++)
        isOG = isOG || (node->outputs->data[j] == current_output);
      is_global_output[i] = isOG;
      // Find all the remaining nodes that are dependent on this output tensor
      for (int j = i; j < node_count; j++)
        for (int k = 0; k < inputs_[j].size(); k++)
          if (inputs_[j][k] == current_output) future_nodes.push_back(j);
      output_dependencies[i] = future_nodes;
      node_output_needed[i] = output_dependencies[i].size() > 0;
      // =======================================================

      if (builtin_code_[i] == kTfLiteBuiltinConv2d) {
        // MOD: preload weights/sums during Prepare.
        Prepare_CONV2D_INT8(
            context, node, i, layers_params[i], opdatas[i], inputs_, outputs_,
            out_tid, temp_im2col[i], wb0[i], wb1[i], wb2[i], wb3[i], wt_sum1[i],
            wt_sum2[i], wt_sum3[i], wt_sum4[i], biases[i], crf_arr[i],
            crx_arr[i], preloaded_weights[i], preloaded_weight_sums[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) {
        // MOD: preload weights/sums during Prepare.
        Prepare_FC_INT8(context, node, i, layers_params[i], opdatas[i], inputs_,
                        outputs_, out_tid, wb0[i], wb1[i], wb2[i], wb3[i],
                        wt_sum1[i], wt_sum2[i], wt_sum3[i], wt_sum4[i],
                        biases[i], crf_arr[i], crx_arr[i], preloaded_weights[i],
                        preloaded_weight_sums[i]);
      }
    }
    prf_end(3, a_t.prep);
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    int node_count = inputs_.size();
    struct acc_container drv;
    drv.acc = acc;
    drv.profile = &profile;
    drv.mdma = &mdma;
    drv.mt_context = &mt_context;
    drv.thread_count = context->recommended_num_threads;
    prf_start(4);

    // MOD: ensure cached weights exist before Eval uses them.
    auto ensure_weight_cache = [&](int node_index, int rows, int cols,
                                   const int8_t *raw_weights, int padded_rows,
                                   int padded_cols) {
      if (!preloaded_weights[node_index].empty() &&
          preloaded_weights[node_index].size() ==
              static_cast<size_t>(padded_rows * padded_cols)) {
        return;
      }
      preloaded_weights[node_index].assign(padded_rows * padded_cols, 0);
      preloaded_weight_sums[node_index].clear();
      preloaded_weight_sums[node_index].reserve(padded_rows);
      precal_sum_load_padv3_vectorized(const_cast<int8_t *>(raw_weights), rows,
                                       cols,
                                       preloaded_weights[node_index].data(),
                                       preloaded_weight_sums[node_index], 64, 64);
    };

    for (int i = 0; i < node_count; i++) {
      drv.op_type = builtin_code_[i];
    #if VIT_DELEGATE_VERBOSE
      cout << "======================================================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i]
           << "      Type: " << EnumNamesBuiltinOperator()[builtin_code_[i]]
           << endl;
      cout << "======================================================" << endl;
    #endif

      if (builtin_code_[i] == kTfLiteBuiltinConv2d) { // CONV2D
        TfLiteConvParams *params =
            reinterpret_cast<TfLiteConvParams *>(layers_params[i]);
        Conv2D_Data *data = reinterpret_cast<Conv2D_Data *>(opdatas[i]);

        TfLiteTensor *output;
        const TfLiteTensor *input;
        const TfLiteTensor *filter;
        const TfLiteTensor *bias;

        GetInputSafe(context, inputs_[i][0], &input);
        GetInputSafe(context, inputs_[i][1], &filter);
        GetInputSafe(context, inputs_[i][2], &bias);
        GetOutputSafe(context, outputs_[i][0], &output);

        const int32_t input_offset = -input->params.zero_point;
        const int32_t filter_offset = -filter->params.zero_point;
        const int32_t output_offset = output->params.zero_point;

        int8_t *im2col_data = data->need_im2col ? &temp_im2col[i][0] : nullptr;
        ConvParams op_params;
        op_params.input_offset = -input->params.zero_point;
        op_params.output_offset = output->params.zero_point;
        op_params.stride_height = params->stride_height;
        op_params.stride_width = params->stride_width;
        op_params.dilation_height_factor = params->dilation_height_factor;
        op_params.dilation_width_factor = params->dilation_width_factor;
        op_params.padding_values.height = data->padding.height;
        op_params.padding_values.width = data->padding.width;
        op_params.quantized_activation_min = data->output_activation_min;
        op_params.quantized_activation_max = data->output_activation_max;
        
        int stride_height = params->stride_height;
        int stride_width = params->stride_width;
        int filter_height = filter->dims->data[1];
        int filter_width = filter->dims->data[2];
        int input_height = input->dims->data[1];
        int input_width = input->dims->data[2];
        int input_depth = input->dims->data[3];
        int output_height = output->dims->data[1];
        int batches = input->dims->data[0];
        int output_width = output->dims->data[2];
        int output_channel = output->dims->data[3];
        int filter_input_depth = filter->dims->data[3];
        int groups = input_depth / filter_input_depth;
        int dilation_width_factor = params->dilation_width_factor;
        int dilation_height_factor = params->dilation_height_factor;
        TFLITE_DCHECK_NE(groups, 0);
        TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
        int filters_per_group = output_channel / groups;
        TFLITE_DCHECK_NE(filters_per_group, 0);
        RuntimeShape input_shape =
            RuntimeShape(input->dims->size, input->dims->data);
        RuntimeShape filter_shape =
            RuntimeShape(filter->dims->size, filter->dims->data);
        RuntimeShape output_shape =
            RuntimeShape(output->dims->size, output->dims->data);
        TfLiteTensor *im2col =
            data->need_im2col
                ? &context->tensors[node->temporaries->data[data->im2col_index]]
                : nullptr;

        int pad_width = data->padding.height;
        int pad_height = data->padding.width;
        const int8 *input_data = input->data.int8;
        const int8 *filter_data = filter->data.int8;
        int8 *output_data = output->data.int8;

        const int32 output_activation_min = data->output_activation_min;
        const int32 output_activation_max = data->output_activation_max;

        const int8 *gemm_input_data = nullptr;
        const RuntimeShape *gemm_input_shape = nullptr;
        const bool need_dilated_im2col =
            dilation_width_factor != 1 || dilation_height_factor != 1;
        const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                                 filter_width != 1 || filter_height != 1;
        const int8 input_zero_point = -input_offset;
        const uint8 zero_point_byte =
            *reinterpret_cast<const uint8 *>(&input_zero_point);
        if (need_dilated_im2col) {
          TFLITE_DCHECK(im2col_data);
          RuntimeShape im2col_shape =
              RuntimeShape(im2col->dims->size, im2col->dims->data);
          DilatedIm2col<int8_t>(op_params, zero_point_byte, input_shape,
                                input_data, filter_shape, output_shape,
                                im2col_data);

          gemm_input_data = im2col_data;
          gemm_input_shape = &im2col_shape;
        } else if (need_im2col) {
          TFLITE_DCHECK(im2col_data);
          RuntimeShape im2col_shape =
              RuntimeShape(im2col->dims->size, im2col->dims->data);
          Im2col<int8_t>(op_params, filter_height, filter_width,
                         zero_point_byte, input_shape, input_data, im2col_shape,
                         im2col_data);
          gemm_input_data = im2col_data;
          gemm_input_shape = &im2col_shape;
        } else {
          TFLITE_DCHECK(!im2col_data);
          gemm_input_data = input_data;
          gemm_input_shape = &input_shape;
        }

        const int gemm_input_rows = gemm_input_shape->Dims(3);
        const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
        const int filter_rows = filter_shape.Dims(0);
        const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
        const int output_rows = output_shape.Dims(3);
        const int output_cols =
            output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);

        drv.N = gemm_input_cols;
        drv.M = filter_rows;
        drv.K = filter_cols;
        drv.layer_t = 1; // CONV layer

        int rfactor = 64;
        int unroll = 64;
        drv.pN = roundUp(drv.N, rfactor);
        drv.pM = roundUp(drv.M, rfactor);
        drv.pK = roundUp(drv.K, unroll);

        std::vector<int> in_sum;
        int8_t *padded_input = new int8_t[drv.pN * drv.pK];
        int8_t *padded_output = new int8_t[drv.pM * drv.pN];

        precal_sum_load_padv3_vectorized(const_cast<int8_t *>(gemm_input_data),
                                         drv.N, drv.K, padded_input, in_sum, 64, 64);
        
        ensure_weight_cache(i, drv.M, drv.K, filter->data.int8, drv.pM, drv.pK);

        drv.padded_input = padded_input;
        drv.padded_weights = preloaded_weights[i].empty()
                                 ? nullptr
                                 : preloaded_weights[i].data();
        drv.padded_output = padded_output;
        drv.in_sum = &in_sum[0];
        drv.wt_sum = preloaded_weight_sums[i].empty()
                         ? nullptr
                         : preloaded_weight_sums[i].data();
        TF_LITE_ENSURE(context, drv.padded_weights != nullptr);
        TF_LITE_ENSURE(context, drv.wt_sum != nullptr);
        
        drv.crx_a = crx_arr[i];
        drv.crf_a = crf_arr[i];
        drv.ra = output_offset;
        drv.rhs_offset = input_offset;
        drv.lhs_offset = filter_offset;
        drv.a_t = &a_t;

        drv.bias = biases[i].data();
        drv.is_bias = 1;

        drv.layer = associated_nodes[i];
        drv.start_count = dparams.start_count;

        prf_start(0);
        vit_sim::Entry(drv);
        prf_end(0, a_t.driver);

        const int out_batches = output->dims->data[0];
        const int out_rows = output->dims->data[1] * output->dims->data[2];
        const int out_cols = output->dims->data[3];
        
        store_unpad(padded_output, drv.N, drv.M, output->data.int8, 64, 64);

        delete[] padded_input;
        delete[] padded_output;

      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) { // FC
        t.layer = dparams.layer;
        t.conv_layer_no = dparams.layer;
        t.node = associated_nodes[i];

        TfLiteFullyConnectedParams *params =
            reinterpret_cast<TfLiteFullyConnectedParams *>(layers_params[i]);
        FC_Data *data = reinterpret_cast<FC_Data *>(opdatas[i]);
        const TfLiteTensor *input;
        const TfLiteTensor *filter;
        const TfLiteTensor *bias;
        TfLiteTensor *output;
        GetInputSafe(context, inputs_[i][0], &input);
        GetInputSafe(context, inputs_[i][1], &filter);
        GetOutputSafe(context, outputs_[i][0], &output);

        bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
        if (isBias) GetInputSafe(context, inputs_[i][2], &bias);
        else bias = nullptr;

        const int32_t output_offset = output->params.zero_point;
        const int32_t output_multiplier = data->output_multiplier;
        const int output_shift = data->output_shift;

        const int32_t input_offset = -input->params.zero_point;
        const int32_t filter_offset = -filter->params.zero_point;

        RuntimeShape output_shape = GetTensorShape(output);
        const int output_rank = output_shape.DimensionsCount();
        const int output_depth = output_shape.Dims(output_rank - 1);
        const int batches = FlatSizeSkipDim(output_shape, output_rank - 1);
        const int accum_depth = GetTensorShape(filter).Dims(1);

        drv.layer_t = 0; // FC layer
        drv.N = batches;
        drv.M = output_depth;
        drv.K = accum_depth;
        
        int rfactor = 64;
        int unroll = 64;
        drv.pN = roundUp(drv.N, rfactor);
        drv.pM = roundUp(drv.M, rfactor);
        drv.pK = roundUp(drv.K, unroll);

        std::vector<int> in_sum;
        int8_t *padded_input = new int8_t[drv.pN * drv.pK];
        int8_t *padded_output = new int8_t[drv.pM * drv.pN];

        precal_sum_load_padv3_vectorized(input->data.int8, drv.N, drv.K,
                                         padded_input, in_sum, 64, 64);
        
        ensure_weight_cache(i, drv.M, drv.K, filter->data.int8, drv.pM, drv.pK);

        drv.padded_input = padded_input;
        drv.padded_weights = preloaded_weights[i].empty()
                                 ? nullptr
                                 : preloaded_weights[i].data();
        drv.padded_output = padded_output;
        drv.in_sum = &in_sum[0];
        drv.wt_sum = preloaded_weight_sums[i].empty()
                         ? nullptr
                         : preloaded_weight_sums[i].data();
        TF_LITE_ENSURE(context, drv.padded_weights != nullptr);
        TF_LITE_ENSURE(context, drv.wt_sum != nullptr);
        
        drv.crx = data->output_shift;
        drv.crf = data->output_multiplier;
        drv.crx_a = crx_arr[i];
        drv.crf_a = crf_arr[i];
        drv.ra = output_offset;
        drv.rhs_offset = input_offset;
        drv.lhs_offset = filter_offset;
        drv.a_t = &a_t;

        drv.bias = biases[i].data();
        drv.is_bias = 1;

        drv.layer = associated_nodes[i];
        drv.start_count = dparams.start_count;
        prf_start(0);
        vit_sim::Entry(drv);
        prf_end(0, a_t.driver);
        dparams.start_count = drv.start_count;

        store_unpad(padded_output, drv.N, drv.M, output->data.int8, 64, 64);

        delete[] padded_input;
        delete[] padded_output;
      }

      for (int n = 0; n < i; n++) {
        for (int dep_node : output_dependencies[n]) {
          if (dep_node == i) {
            output_dependencies[n].erase(
                std::remove(output_dependencies[n].begin(),
                            output_dependencies[n].end(), dep_node),
                output_dependencies[n].end());
          }
          node_output_needed[n] = output_dependencies[n].size() > 0;
        }
      }
      dparams.layer++;
      dparams.delegated_nodes--;
    }
    prf_end(4, a_t.total_inference);
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<std::vector<int>> wt_sum1;
  std::vector<std::vector<int>> wt_sum2;
  std::vector<std::vector<int>> wt_sum3;
  std::vector<std::vector<int>> wt_sum4;
  std::vector<std::vector<int8_t>> wb0;
  std::vector<std::vector<int8_t>> wb1;
  std::vector<std::vector<int8_t>> wb2;
  std::vector<std::vector<int8_t>> wb3;
  std::vector<std::vector<int>> biases;
  std::vector<std::vector<int8_t>> temp_im2col;
  std::vector<std::vector<int8_t>> preloaded_weights;
  std::vector<std::vector<int>> preloaded_weight_sums;
  std::vector<std::vector<int>> crf_arr;
  std::vector<std::vector<int8_t>> crx_arr;

  std::vector<void *> opdatas;
  std::vector<void *> layers_params;

  std::vector<std::vector<int>> output_dependencies;
  std::vector<bool> node_output_needed;
  std::vector<bool> is_global_output;

private:
  const VitDelegateOptions options_;
};

class VitDelegate : public SimpleDelegateInterface {
public:
  explicit VitDelegate(const VitDelegateOptions &options) : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {

    bool isCONV2D = IsNode_CONV2D_INT8(registration, node, context);
    bool isFC = IsNode_FC_INT8(registration, node, context);
    
    std::vector<bool> supported_nodes = {isCONV2D, isFC};

    bool delegated_node = false;

    for (int i = 0; i < supported_nodes.size(); i++)
      if (supported_nodes[i]) delegated_node = true;

    if (delegated_node && isCONV2D) {
      const TfLiteTensor *input = &context->tensors[node->inputs->data[0]];
      const TfLiteTensor *filter = &context->tensors[node->inputs->data[1]];

      int input_depth = input->dims->data[3];
      int filter_input_depth = filter->dims->data[3];
      int groups = input_depth / filter_input_depth;
    }

    int output_tid = node->outputs->data[0];
    int forbidden_output_tid[] = {};
    for (int tid : forbidden_output_tid)
      if (output_tid == tid) delegated_node = false;

    if (delegated_node) {
      int pk = 0;
      if (isFC) {
        const TfLiteTensor &filter = context->tensors[node->inputs->data[1]];
        auto filter_shape = tflite::GetTensorShape(&filter);
        if (filter_shape.DimensionsCount() > 1) pk = filter_shape.Dims(1);
      } else if (isCONV2D) {
        const TfLiteTensor &filter = context->tensors[node->inputs->data[1]];
        auto filter_shape = tflite::GetTensorShape(&filter);
        pk = tflite::FlatSizeSkipDim(filter_shape, 0);
      }

      if (pk > 1024) {
        delegated_node = false;
      }
    }

    if (delegated_node) dparams.delegated_nodes++;
    return delegated_node;
  }
  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "VitDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<VitDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    return SimpleDelegateInterface::Options();
  }

private:
  const VitDelegateOptions options_;
};

} // namespace vit_test
} // namespace tflite

VitDelegateOptions TfLiteVitDelegateOptionsDefault() {
  VitDelegateOptions options = {0};
  options.allowed_builtin_code = -1;
  return options;
}

TfLiteDelegate *TfLiteVitDelegateCreate(const VitDelegateOptions *options) {
  std::unique_ptr<tflite::vit_test::VitDelegate> vit(
      new tflite::vit_test::VitDelegate(
          options ? *options : TfLiteVitDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(vit), kTfLiteDelegateFlagsAllowDynamicTensors);
}

void TfLiteVitDelegateDelete(TfLiteDelegate *delegate) {
  SYSC_ON(profile.saveProfile(acc->profiling_vars));   
  SYSC_ON(profile.saveCSVRecords(".data/vit_sim_v9")); 
#ifndef SYSC
  if (!dparams.unmap) {
    mdma.multi_free_dmas();
    munmap(dparams.acc, 65536);
    std::cout << "===========================" << std::endl;
    std::cout << "Unmapped DMA I/O Buffers" << std::endl;
    std::cout << "===========================" << std::endl;
    dparams.unmap = true;
  }
#endif
  a_t.print();
  a_t.save_prf();
  std::cout << "===========================" << std::endl;
  std::cout << "Deleted" << std::endl;
  std::cout << "===========================" << std::endl;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

bool Prepare_CONV2D_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                         void *layers_params, void *opdatas,
                         vector<vector<int>> &inputs_,
                         vector<vector<int>> &outputs_, int &out_tid,
                         vector<int8_t> &temp_im2col, vector<int8_t> &wb0,
                         vector<int8_t> &wb1, vector<int8_t> &wb2,
                         vector<int8_t> &wb3, vector<int> &wt_sum1,
                         vector<int> &wt_sum2, vector<int> &wt_sum3,
                         vector<int> &wt_sum4, vector<int> &biases,
                         vector<int> &crf, vector<int8_t> &crx,
                         std::vector<int8_t> &preloaded_weights,
                         std::vector<int> &preloaded_wt_sum) {

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
  bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
  if (isBias) GetInputSafe(context, inputs_[i][2], &bias);

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
      context, input, filter, isBias ? bias : nullptr, output, params->activation,
      &data->output_multiplier, &data->output_shift,
      &data->output_activation_min, &data->output_activation_max,
      data->per_channel_output_multiplier.data(),
      data->per_channel_output_shift.data(), channels_out));

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
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  int padded_channels = ((channels_out + 64 - 1) / 64) * 64;
  crf.resize(padded_channels, 0);
  crx.resize(padded_channels, 0);

  for (int j = 0; j < channels_out; j++) {
    crf[j] = data->per_channel_output_multiplier[j];
    int shift = data->per_channel_output_shift[j];
    crx[j] = shift > 0 ? (1 << shift) : shift;
  }

  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  int K = filter_height * filter_width * channels_in;
  int cross_term = K * input_offset * filter_offset;

  if (isBias) {
    biases.assign(bias->data.i32, bias->data.i32 + channels_out);
  } else {
    biases.assign(channels_out, 0);
  }

  for (int j = 0; j < channels_out; j++) {
    biases[j] += cross_term;
  }
  
  biases.resize(padded_channels, cross_term);

  int *dims = filter->dims->data;
  int inpZeroPoint = -input->params.zero_point;
  preload_weights(filter->data.int8, dims, wb0, wb1, wb2, wb3, wt_sum1,
                  wt_sum2, wt_sum3, wt_sum4, inpZeroPoint, biases.data());

  const int filter_rows = filter->dims->data[0];
  int filter_cols = 1;
  for (int dim = 1; dim < filter->dims->size; ++dim) {
    filter_cols *= filter->dims->data[dim];
  }
  const int padded_rows = ((filter_rows + 64 - 1) / 64) * 64;
  const int padded_cols = ((filter_cols + 64 - 1) / 64) * 64;
  preloaded_weights.assign(padded_rows * padded_cols, 0);
  preloaded_wt_sum.clear();
  preloaded_wt_sum.reserve(padded_rows);
  precal_sum_load_padv3_vectorized(
      const_cast<int8_t *>(filter->data.int8), filter_rows, filter_cols,
      preloaded_weights.data(), preloaded_wt_sum, 64, 64);

  return kTfLiteOk;
}

bool Prepare_FC_INT8(TfLiteContext *context, TfLiteNode *node, int i,
                     void *layers_params, void *opdatas,
                     vector<vector<int>> &inputs_, vector<vector<int>> &outputs_,
                     int &out_tid, vector<int8_t> &wb0, vector<int8_t> &wb1,
                     vector<int8_t> &wb2, vector<int8_t> &wb3,
                     vector<int> &wt_sum1, vector<int> &wt_sum2,
                     vector<int> &wt_sum3, vector<int> &wt_sum4,
                     vector<int> &biases, vector<int> &crf,
                     vector<int8_t> &crx, std::vector<int8_t> &preloaded_weights,
                     std::vector<int> &preloaded_wt_sum) {
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

  double real_multiplier = 0.0;
  tflite::GetQuantizedConvolutionMultipler(context, input, filter, bias, output,
                                           &real_multiplier);
  int exponent;
  tflite::QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                             &exponent);
  data->output_shift = exponent;

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
    TF_LITE_ENSURE(context,
                   input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
    TF_LITE_ENSURE(context, (filter->type == kTfLiteInt8));
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      per_channel_quantization_size);
    TF_LITE_ENSURE_EQ(
        context, per_channel_quantization_size,
        filter->dims->data[affine_quantization->quantized_dimension]);
    
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

  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++)
    input_size *= input->dims->data[i];
  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  TfLiteIntArray *output_size = nullptr;
  if (params->keep_num_dims) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1],
                      filter->dims->data[1]);
    output_size = TfLiteIntArrayCopy(input->dims);
    output_size->data[output_size->size - 1] = num_units;
  } else {
    output_size = TfLiteIntArrayCreate(2);
    output_size->data[0] = batch_size;
    output_size->data[1] = num_units;
  }
  auto output_status = context->ResizeTensor(context, output, output_size);
  if (output_status != kTfLiteOk) return output_status;

  int temp_out_id;
  bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
  if (!req_temp_out) out_tid++;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryOutTensorsIfRequired(
      context, node, req_temp_out, outputs_[i][0], temp_out_id));
  ResizeTempOutTensor(context, node, req_temp_out, temp_out_id, outputs_, i,
                      output_size);

  int channels_out = filter->dims->data[0];
  int padded_channels = ((channels_out + 64 - 1) / 64) * 64;
  
  crf.resize(padded_channels, 0);
  crx.resize(padded_channels, 0);

  if (is_per_channel) {
    for (int j = 0; j < per_channel_quantization_size; j++) {
      crf[j] = data->per_channel_output_multiplier[j];
      int shift = data->per_channel_output_shift.data()[j];
      crx[j] = shift > 0 ? (1 << shift) : shift;
    }
  } else {
    int shift = data->output_shift;
    data->output_shift = shift > 0 ? (1 << shift) : shift; 
    
    for (int j = 0; j < channels_out; j++) {
      crf[j] = data->output_multiplier;
      crx[j] = data->output_shift;
    }
  }

  int *dims = new int[4];
  dims[0] = filter->dims->data[0];
  dims[1] = 1;
  dims[2] = 1;
  dims[3] = filter->dims->data[1];

  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  int K = filter->dims->data[1];
  int cross_term = K * input_offset * filter_offset;

  if (isBias) {
    biases.assign(bias->data.i32, bias->data.i32 + channels_out);
  } else {
    biases.assign(channels_out, 0); 
  }

  for (int j = 0; j < channels_out; j++) {
    biases[j] += cross_term;
  }
  biases.resize(padded_channels, cross_term);

  int inpZeroPoint = -input->params.zero_point;
  preload_weights(filter->data.int8, dims, wb0, wb1, wb2, wb3, wt_sum1,
                  wt_sum2, wt_sum3, wt_sum4, inpZeroPoint,
                  biases.data()); 
  
  const int output_depth = filter->dims->data[0];
  const int accum_depth = filter->dims->data[1];
  const int padded_rows = ((output_depth + 64 - 1) / 64) * 64;
  const int padded_cols = ((accum_depth + 64 - 1) / 64) * 64;
  preloaded_weights.assign(padded_rows * padded_cols, 0);
  preloaded_wt_sum.clear();
  preloaded_wt_sum.reserve(padded_rows);
  precal_sum_load_padv3_vectorized(
      const_cast<int8_t *>(filter->data.int8), output_depth, accum_depth,
      preloaded_weights.data(), preloaded_wt_sum, 64, 64);
      
  return kTfLiteOk;
}

