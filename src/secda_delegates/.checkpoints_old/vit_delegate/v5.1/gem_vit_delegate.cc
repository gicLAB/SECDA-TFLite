#include "tensorflow/lite/delegates/utils/secda_delegates/vit_delegate/v5.1/vit_delegate.h"
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
      std::cout << "VIT_ACC Accelerator v5.1";
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
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    wb0.resize(params->nodes_to_replace->size);
    wb1.resize(params->nodes_to_replace->size);
    wb2.resize(params->nodes_to_replace->size);
    wb3.resize(params->nodes_to_replace->size);
    wt_sum1.resize(params->nodes_to_replace->size);
    wt_sum2.resize(params->nodes_to_replace->size);
    wt_sum3.resize(params->nodes_to_replace->size);
    wt_sum4.resize(params->nodes_to_replace->size);
    temp_im2col.resize(params->nodes_to_replace->size);

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
        Prepare_CONV2D_INT8(context, node, i, layers_params[i], opdatas[i],
                            inputs_, outputs_, out_tid, temp_im2col[i], wb0[i],
                            wb1[i], wb2[i], wb3[i], wt_sum1[i], wt_sum2[i],
                            wt_sum3[i], wt_sum4[i], biases[i], crf[i], crx[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) {
        Prepare_FC_INT8(context, node, i, layers_params[i], opdatas[i], inputs_,
                        outputs_, out_tid, wb0[i], wb1[i], wb2[i], wb3[i],
                        wt_sum1[i], wt_sum2[i], wt_sum3[i], wt_sum4[i],
                        biases[i], crf[i], crx[i]);
      }
    }
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

    for (int i = 0; i < node_count; i++) {
      drv.op_type = builtin_code_[i];
      cout << "======================================================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i]
           << "      Type: " << EnumNamesBuiltinOperator()[builtin_code_[i]]
           << endl;
      cout << "======================================================" << endl;

      if (builtin_code_[i] == kTfLiteBuiltinConv2d) { // CONV2D
        TfLiteConvParams *params =
            reinterpret_cast<TfLiteConvParams *>(layers_params[i]);
        Conv2D_Data *data = reinterpret_cast<Conv2D_Data *>(opdatas[i]);

        TfLiteTensor *im2col =
            data->need_im2col
                ? &context->tensors[node->temporaries->data[data->im2col_index]]
                : nullptr;

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
        //H: addded
        op_params.padding_values.height = data->padding.height;
        op_params.padding_values.width = data->padding.width;
        op_params.stride_height = params->stride_height;
        op_params.stride_width = params->stride_width;
        op_params.dilation_height_factor = params->dilation_height_factor;
        op_params.dilation_width_factor = params->dilation_width_factor;

        const int8 *gemm_input_data = nullptr;
        const int filter_height = filter->dims->data[1];
        const int filter_width = filter->dims->data[2];
        const bool need_im2col = params->stride_width != 1 ||
                                 params->stride_height != 1 ||
                                 filter_width != 1 || filter_height != 1;

        if (!need_im2col) {
          gemm_input_data = input->data.int8;
        } else {
          Im2col<int8_t>(op_params, filter_height, filter_width, input_offset,
                         GetTensorShape(input), input->data.int8,
                         GetTensorShape(im2col), im2col_data);
          gemm_input_data = im2col_data;
        }

        const int gemm_input_cols = FlatSizeSkipDim(
            need_im2col ? GetTensorShape(im2col) : GetTensorShape(input), 3);
        const int filter_rows = filter->dims->data[0];
        const int filter_cols = FlatSizeSkipDim(GetTensorShape(filter), 0);

        drv.N = gemm_input_cols;
        drv.M = filter_rows;
        drv.K = filter_cols;

        int rfactor = 16;
        int unroll = 64;
        drv.pN = roundUp(drv.N, rfactor);
        drv.pM = roundUp(drv.M, rfactor);
        drv.pK = roundUp(drv.K, unroll);

        std::vector<int> in_sum;
        std::vector<int> wt_sum;
        int8_t *padded_input = new int8_t[drv.pN * drv.pK];
        int8_t *padded_weights = new int8_t[drv.pM * drv.pK];
        int8_t *padded_output = new int8_t[drv.pM * drv.pN];

        precal_sum_load_padv3_vectorized(const_cast<int8_t *>(gemm_input_data),
                                         drv.N, drv.K, padded_input, in_sum);
        precal_sum_load_padv3_vectorized(
            const_cast<int8_t *>(filter->data.int8), drv.M, drv.K,
            padded_weights, wt_sum);

        drv.padded_input = padded_input;
        drv.padded_weights = padded_weights;
        drv.padded_output = padded_output;
        drv.in_sum = &in_sum[0];
        drv.wt_sum = &wt_sum[0];
        drv.crx = data->output_shift;
        drv.crf = data->output_multiplier;
        drv.ra = output_offset;
        drv.rhs_offset = input_offset;
        drv.lhs_offset = filter_offset;
        drv.a_t = &a_t;

        bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
        if (!isBias) {
          drv.bias = new int32_t[drv.pM]();
          drv.is_bias = 0;
        } else {
          drv.bias = biases[i].data();
          drv.is_bias = 1;
        }

        drv.layer = associated_nodes[i];
        drv.start_count = dparams.start_count;

        prf_start(0);
        vit_sim::Entry(drv);
        prf_end(0, a_t.driver);

        dparams.start_count = drv.start_count;

        //! Wrong according to AI
        const int out_batches = output->dims->data[0];
        const int out_rows = output->dims->data[1] * output->dims->data[2];
        const int out_cols = output->dims->data[3];
        store_unpad(padded_output, out_batches * out_rows, out_cols,
        output->data.int8, 16, 16);
        //! Ai says replace this with
        // const int batches = output->dims->data[0];
        // const int height = output->dims->data[1];
        // const int width = output->dims->data[2];
        // const int channels = output->dims->data[3];

        // // Calculate the total number of rows across ALL batches.
        // const int total_rows = batches * height * width;

        // store_unpad(padded_output, total_rows, channels, output->data.int8, 16,
        //             16);

        // //! end AI
        delete[] padded_input;
        delete[] padded_weights;
        delete[] padded_output;
        if (!isBias) delete[] drv.bias;

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
        const int output_depth = output_shape.Dims(1);
        const int batches = output_shape.Dims(0);
        const int accum_depth = GetTensorShape(filter).Dims(1);

        drv.N = batches;
        drv.M = output_depth;
        drv.K = accum_depth;
        int rfactor = 16;
        int unroll = 64;
        drv.pN = roundUp(drv.N, rfactor);
        drv.pM = roundUp(drv.M, rfactor);
        drv.pK = roundUp(drv.K, unroll);

        std::vector<int> in_sum;
        std::vector<int> wt_sum;
        int8_t *padded_input = new int8_t[drv.pN * drv.pK];
        int8_t *padded_weights = new int8_t[drv.pM * drv.pK];
        int8_t *padded_output = new int8_t[drv.pM * drv.pN];

        precal_sum_load_padv3_vectorized(input->data.int8, drv.N, drv.K,
                                         padded_input, in_sum);
        precal_sum_load_padv3_vectorized(filter->data.int8, drv.M, drv.K,
                                         padded_weights, wt_sum);

        drv.padded_input = padded_input;
        drv.padded_weights = padded_weights;
        drv.padded_output = padded_output;
        drv.in_sum = &in_sum[0];
        drv.wt_sum = &wt_sum[0];
        drv.crx = output_shift;
        drv.crf = output_multiplier;
        drv.ra = output_offset;
        drv.rhs_offset = input_offset;
        drv.lhs_offset = filter_offset;
        drv.a_t = &a_t;

        if (!isBias) {
          drv.bias = new int32_t[drv.pM]();
          drv.is_bias = 0;
        } else {
          drv.bias = biases[i].data();
          drv.is_bias = 1;
        }

        drv.layer = associated_nodes[i];
        drv.start_count = dparams.start_count;
        prf_start(0);
        vit_sim::Entry(drv);
        prf_end(0, a_t.driver);
        dparams.start_count = drv.start_count;

        store_unpad(padded_output, drv.N, drv.M, output->data.int8, 16, 16);

        delete[] padded_input;
        delete[] padded_weights;
        delete[] padded_output;
        if (!isBias) delete[] drv.bias;
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
  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int8_t>> crx;

  std::vector<void *> opdatas;
  std::vector<void *> layers_params;

  std::vector<std::vector<int>> output_dependencies;
  std::vector<bool> node_output_needed;
  std::vector<bool> is_global_output;

private:
  const VitDelegateOptions options_;
};

// VitDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
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

    int output_tid = node->outputs->data[0];
    int forbidden_output_tid[] = {
        // 110, //  restricts node with output tid 110 from being delegated
    };
    for (int tid : forbidden_output_tid)
      if (output_tid == tid) delegated_node = false;

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
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const VitDelegateOptions options_;
};

} // namespace vit_test
} // namespace tflite

VitDelegateOptions TfLiteVitDelegateOptionsDefault() {
  VitDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this vit test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteVitDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteVitDelegateCreate(const VitDelegateOptions *options) {
  std::unique_ptr<tflite::vit_test::VitDelegate> vit(
      new tflite::vit_test::VitDelegate(
          options ? *options : TfLiteVitDelegateOptionsDefault()));
  // return
  // tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(vit));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(vit), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `TfLiteVitDelegateCreate` call.
void TfLiteVitDelegateDelete(TfLiteDelegate *delegate) {
  SYSC_ON(profile.saveProfile(acc->profiling_vars));     // in-driver
  SYSC_ON(profile.saveCSVRecords(".data/vit_sim_v5.1")); // in-fpga
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