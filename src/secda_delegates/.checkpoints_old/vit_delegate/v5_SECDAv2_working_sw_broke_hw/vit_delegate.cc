
#include <fstream>
#include <iostream>
#include <utility>
#include "tensorflow/lite/delegates/utils/secda_delegates/vit_delegate/v5/vit_delegate.h"

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif

#include "secda_tools/secda_profiler/profiler.h"
#include "secda_tools/secda_utils/acc_helpers.h"
#include "secda_tools/secda_utils/multi_threading.h"
#include "secda_tools/secda_utils/utils.h"
#include "accelerator/driver/vit_driver.h"
#include "util.h"
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
      std::cout << "VIT_ACC Accelerator";
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
    opdatas.resize(params->nodes_to_replace->size);
    cparams.resize(params->nodes_to_replace->size);

    wgt_sum.resize(params->nodes_to_replace->size);
    biases.resize(params->nodes_to_replace->size);
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    weight_offsets.resize(params->nodes_to_replace->size);
    del_weights.resize(params->nodes_to_replace->size);

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
      // 3 inputs...
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      inputs_[i].push_back(delegated_node->inputs->data[2]);
      // 1 ouptut...
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      // Built-in code...
      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);
      TfLiteFullyConnectedParams *cparam =
          reinterpret_cast<TfLiteFullyConnectedParams *>(
              delegated_node->builtin_data);
      OpData *opdata = reinterpret_cast<OpData *>(delegated_node->user_data);
      cparams[i] = cparam;
      opdatas[i] = opdata;
    }
    return kTfLiteOk;
  }
  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) override {
    //! Id want ot do it here
    //! Also move checking the bias here
    int node_count = inputs_.size();
    int out_tid = 0;
    for (int i = 0; i < node_count; i++) {
      TfLiteFullyConnectedParams *params = cparams[i];
      OpData *data = opdatas[i];

      TfLiteTensor *output;
      const TfLiteTensor *input;
      const TfLiteTensor *filter;
      const TfLiteTensor *bias;

      GetOutputSafe(context, outputs_[i][0], &output);
      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      if (inputs_[i].size() == 3 && inputs_[i][2] >= 0) {
        GetInputSafe(context, inputs_[i][2], &bias);
        biases[i] = bias->data.i32;
      } else {
        biases[i] = nullptr;
        bias = nullptr;
      }

      // Get Qaunt Params.
      double real_multiplier = 0.0;
      int exponent;
      GetQuantizedConvolutionMultipler(context, input, filter, bias, output,
                                       &real_multiplier);
      QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                         &data->output_shift);
      CalculateActivationRangeQuantized(context, params->activation, output,
                                        &data->output_activation_min,
                                        &data->output_activation_max);

      // Resize output.
      int input_size = 1;
      for (int i = 0; i < input->dims->size; i++)
        input_size *= input->dims->data[i];
      const int batch_size = input_size / filter->dims->data[1];
      const int num_units = filter->dims->data[0];
      const int out_dim1 = batch_size;
      const int out_dim2 = num_units;
      TfLiteIntArray *output_size = TfLiteIntArrayCreate(2);
      output_size->data[0] = out_dim1;
      output_size->data[1] = out_dim2;
      auto output_status = context->ResizeTensor(context, output, output_size);
      if (output_status != kTfLiteOk) return output_status;

      int temp_out_id;
      bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      if (!req_temp_out) out_tid++;

      TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
          context, node, req_temp_out, outputs_[i][0], temp_out_id,
          inputs_[i][0], inputs_[i][1]));

      int k = node->outputs->data[out_tid];

      if (req_temp_out) {
        node->temporaries->data[temp_out_id] = outputs_[i][0];
        TfLiteIntArray *temp_out_tensor_size = TfLiteIntArrayCreate(2);
        temp_out_tensor_size->data[0] = output_size->data[0];
        temp_out_tensor_size->data[1] = output_size->data[1];

        TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteArenaRw;
        auto temp_out_tensor_status = context->ResizeTensor(
            context, temp_out_tensor, temp_out_tensor_size);
        if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
      }
    }
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  // This funciton executes the operations required by node by offloading the
  // computation to the fully_connected
  // For more info find default implementation at
  // "tensorflow/lite/kernels/fully_connected.cc"
  // OFFLOADS TO ACCELERATOR DRIVER
  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    // Evaluate the delegated graph
    // Loop over all of the delegated nodes
    // Number of nodes = inputs.size() and inputs[i] is a list of
    // tensor indices for the input node i while outputs[i] is the
    // list of ouptuts for that node

    prf_start(1); // cpu_total

    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      auto *params = cparams[i];
      OpData *data = opdatas[i];
      const TfLiteTensor *input;
      const TfLiteTensor *filter; // Weights
      TfLiteTensor *output;

      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      GetOutputSafe(context, outputs_[i][0], &output);

      const TfLiteTensor *bias;
      bool isBias = biases[i] ? true : false;
      if (isBias) GetInputSafe(context, inputs_[i][2], &bias);
      else bias = nullptr;

      const int8 *input_data = input->data.int8;
      const int8 *filter_data = filter->data.int8;
      int8 *output_data = output->data.int8;
      const int32_t *bias_data =
          (bias != nullptr ? reinterpret_cast<int32_t *>(bias->data.raw)
                           : nullptr);

      FullyConnectedParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.weights_offset = -filter->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.output_multiplier = data->output_multiplier;
      op_params.output_shift = data->output_shift;
      op_params.quantized_activation_min = data->output_activation_min;
      op_params.quantized_activation_max = data->output_activation_max;
      op_params.lhs_cacheable = IsConstantTensor(filter);
      op_params.rhs_cacheable = IsConstantTensor(input);
      const int32_t output_offset = op_params.output_offset;
      const int32_t lhs_offset = -op_params.weights_offset;
      const int32_t rhs_offset = -op_params.input_offset;
      const int32_t output_multiplier = op_params.output_multiplier;
      const int output_shift = op_params.output_shift;
      const int32_t output_activation_min = op_params.quantized_activation_min;
      const int32_t output_activation_max = op_params.quantized_activation_max;
      RuntimeShape input_shape =
          RuntimeShape(input->dims->size, input->dims->data);
      RuntimeShape filter_shape =
          RuntimeShape(filter->dims->size, filter->dims->data);
      RuntimeShape output_shape =
          RuntimeShape(output->dims->size, output->dims->data);
      const int output_dim_count = output_shape.DimensionsCount();
      const int filter_dim_count = filter_shape.DimensionsCount();
      const int output_depth = output_shape.Dims(1);
      const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
      const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
      const int batches = output_shape.Dims(0);
      const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

      // nmk = dimensions for vectors with (N,M) and (M,K)
      int N = batches;
      int M = output_depth;
      int K = accum_depth;
      int rfactor = 16;
      int pN = roundUp(N, rfactor); // Padded
      int pM = roundUp(M, rfactor);
      // int pK = roundUp(K, rfactor);
      int unroll = 64;
      int pK = roundUp(K, unroll);

      std::vector<int> in_sum; // Sums accross the layers or smth
      std::vector<int> wt_sum;
      int *idims = input->dims->data;
      int *wdims = filter->dims->data;
      int8_t *padded_input = new int8_t[pN * pK];
      int8_t *padded_weights = new int8_t[pM * pK];
      int8_t *padded_output = new int8_t[pM * pN];

      int8_t *input_data_p = input->data.int8;
      int8_t *filter_data_p = filter->data.int8;
      int8_t *output_data_p = output->data.int8;

      // Calls the fc_driver to re-shape TFLite input/weight tensor and also
      // produces vector of sums from the tensor'r rows (required for
      // re-quantization)
      prf_start(3);
      precal_sum_load_padv3_vectorized(input->data.int8, N, K, padded_input,
                                       in_sum);
      precal_sum_load_padv3_vectorized(filter->data.int8, M, K, padded_weights,
                                       wt_sum);
      prf_end(3, a_t.prep);

      struct acc_container drv;
      drv.profile = &profile;
      drv.acc = acc;
      drv.mdma = &mdma;
      drv.layer = associated_nodes[i];
      drv.pN = pN;
      drv.pM = pM;
      drv.pK = pK;
      drv.N = N;
      drv.M = M;
      drv.K = K;
      drv.padded_input = padded_input;
      drv.padded_weights = padded_weights;
      drv.padded_output = padded_output;
      drv.in_sum = &in_sum[0];
      drv.wt_sum = &wt_sum[0];
      drv.crx = output_shift;
      drv.crf = output_multiplier;
      drv.ra = output_offset;
      drv.rhs_offset = -rhs_offset;
      drv.lhs_offset = -lhs_offset;
      drv.a_t = &a_t;
      if (!isBias) {
        drv.bias = new int32_t[pM]();
        drv.is_bias = 0;
      } else {
        drv.bias = biases[i];
        drv.is_bias = 1;
      }

      // Debugging
#ifdef DELEGATE_VERBOSE
      cout << "===========================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i] << endl;
      cout << "===========================" << endl;
#endif

      // Calls fc_driver to offload the FC operation
      drv.start_count = dparams.start_count;
      prf_start(0); // driver
      vit_sim::Entry(drv);
      // a_t = drv.a_t;
      prf_end(0, a_t.driver);
      dparams.start_count = drv.start_count;

      // Calls the fc_driver to unpack/unpad result to TFLite tensor
      store_unpad(padded_output, N, M, output_data_p, 16, 16);
      if (!isBias) delete[] drv.bias;

      dparams.layer++;
      dparams.delegated_nodes--;
    }
    prf_end(1, a_t.cpu_total);

    return kTfLiteOk;
  };

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<OpData *> opdatas;
  std::vector<TfLiteFullyConnectedParams *> cparams;

  std::vector<std::vector<int>> wgt_sum;
  std::vector<int> weight_offsets;
  std::vector<uint32_t *> del_weights;
  std::vector<int *> biases;
  std::vector<int *> crf;
  std::vector<int8_t *> crx;

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

    // This delegate only supports FC ops
    if (kTfLiteBuiltinFullyConnected != registration->builtin_code)
      return false;

    // This delegate only supports nodes with inputs, filters and biases
    // (optional)
    if (node->inputs->size != 2 && node->inputs->size != 3) return false;

    // This delegate only supports int8 types.
    for (int i = 0; i < 2; ++i) {
      auto &tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    // This node only supports 32-bit biases (if they are there)
    if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
      // auto &tensor = context->tensors[node->inputs->data[2]];
      return false;
      // if (tensor.type != kTfLiteInt32 && tensor.type <= 16) return false;
    }

    // Added node to the list of nodes to be delegated.
    dparams.delegated_nodes++;
    return true;
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
  SYSC_ON(profile.saveProfile(acc->profiling_vars));   // in-driver
  SYSC_ON(profile.saveCSVRecords(".data/vit_sim_v5")); // in-fpga
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