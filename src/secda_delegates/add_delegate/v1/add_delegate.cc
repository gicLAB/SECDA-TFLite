
#include <fstream>
#include <iostream>
#include <utility>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif
#include "secda_tools/secda_profiler/profiler.h"
#include "accelerator/driver/toyadd_driver.h"
#include "add_delegate.h"
#include "util.h"

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#define DELEGATE_NAME "ADD"
#define DELEGATE_VERSION 1
// Some variables needs to be defined across multiple instances of the delegate
unsigned int dma_addrs[1] = {dma_addr0};
unsigned int dma_addrs_in[1] = {dma_in0};
unsigned int dma_addrs_out[1] = {dma_out0};
struct TOYADD_ACC_times p_t;
struct del_params dparams;
static struct Profile profile;
struct MultiThreadContext mt_context;

#ifdef SYSC
static struct s_mdma mdma(1, dma_addrs, dma_addrs_in, dma_addrs_out, 563840);
ACCNAME *acc;
struct sysC_sigs *scs;
#else
struct s_mdma mdma(1, dma_addrs, dma_addrs_in, dma_addrs_out, DMA_BL);
int *acc;
#endif

namespace tflite {
namespace add_test {

// Add delegate kernel.
class AddDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit AddDelegateKernel(const AddDelegateOptions &options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {
    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      std::cout << "===========================" << std::endl;
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("TOYADD_ACC");
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
      std::cout << "TOYADD_ACC Accelerator";
#ifdef ACC_NEON
      std::cout << " with Neon";
#endif
      std::cout << std::endl;
      std::cout << "===========================" << std::endl;
      dparams.init = true;
    }

    // Save Tensors input & outputs
    // Save other info (opdata)
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    opdatas.resize(params->nodes_to_replace->size);
    cparams.resize(params->nodes_to_replace->size);

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
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);
      TfLiteAddParams *cparam =
          reinterpret_cast<TfLiteAddParams *>(delegated_node->builtin_data);
      OpData *opdata = reinterpret_cast<OpData *>(delegated_node->user_data);
      cparams[i] = cparam;
      opdatas[i] = opdata;
    }
    return kTfLiteOk;
  }

  // Runs once per node before inference/invoke()
  // This function allocates additional tensors, calculates
  // quantization parameters For more info look into
  // "tensorflow/lite/kernels/add.cc" for the default implementation for Add
  // Nodes
  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) override {
    int node_count = inputs_.size();
    int out_tid = 0;
    for (int i = 0; i < node_count; i++) {
      TfLiteAddParams *params = cparams[i];
      OpData *data = opdatas[i];

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
          twice_max_input_scale /
          ((1 << data->left_shift) * output->params.scale);
      QuantizeMultiplierSmallerThanOneExp(real_input1_multiplier,
                                          &data->input1_multiplier,
                                          &data->input1_shift);
      QuantizeMultiplierSmallerThanOneExp(real_input2_multiplier,
                                          &data->input2_multiplier,
                                          &data->input2_shift);
      QuantizeMultiplierSmallerThanOneExp(real_output_multiplier,
                                          &data->output_multiplier,
                                          &data->output_shift);
      CalculateActivationRangeQuantized(context, params->activation, output,
                                        &data->output_activation_min,
                                        &data->output_activation_max);

      context->ResizeTensor(context, output, output_size);
    }
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  // This function executes the operations required by node by offloading the
  // computation to the toyadd_driver For more info look into
  // "tensorflow/lite/kernels/add.cc" for the default implementation for Add
  // Nodes
  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    prf_start(0); // Start the profiling delegate
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      auto *params = cparams[i];
      OpData *data = opdatas[i];
      const TfLiteTensor *input1;
      const TfLiteTensor *input2;
      TfLiteTensor *output;
      GetInputSafe(context, inputs_[i][0], &input1);
      GetInputSafe(context, inputs_[i][1], &input2);
      GetOutputSafe(context, outputs_[i][0], &output);
      tflite::ArithmeticParams op_params;
      op_params.left_shift = data->left_shift;
      op_params.input1_offset = data->input1_offset;
      op_params.input1_multiplier = data->input1_multiplier;
      op_params.input1_shift = data->input1_shift;
      op_params.input2_offset = data->input2_offset;
      op_params.input2_multiplier = data->input2_multiplier;
      op_params.input2_shift = data->input2_shift;
      op_params.output_offset = data->output_offset;
      op_params.output_multiplier = data->output_multiplier;
      op_params.output_shift = data->output_shift;
      SetActivationParams(data->output_activation_min,
                          data->output_activation_max, &op_params);

      // Prepare Inputs for Driver/Accelerator
      int size = 1;
      for (int i = 0; i < input1->dims->size; ++i) {
        size *= input1->dims->data[i];
      }
      const int8 *input1_data = input1->data.int8;
      const int8 *input2_data = input2->data.int8;
      int8 *output_data = output->data.int8;

      struct acc_container drv;
      drv.acc = acc;
      drv.profile = &profile;
      drv.mdma = &mdma;
      drv.mt_context = &mt_context;
      drv.thread_count = context->recommended_num_threads;

      // Accelerator Specific Parameters
      drv.input_A = input1_data;
      drv.input_B = input2_data;
      drv.output_C = output_data;
      drv.length = roundDown(size, 4);
      drv.lshift = op_params.left_shift;
      drv.in1_off = op_params.input1_offset;
      drv.in1_sv = op_params.input1_shift;
      drv.in1_mul = op_params.input1_multiplier;
      drv.in2_off = op_params.input2_offset;
      drv.in2_sv = op_params.input2_shift;
      drv.in2_mul = op_params.input2_multiplier;
      drv.out1_off = op_params.output_offset;
      drv.out1_sv = op_params.output_shift;
      drv.out1_mul = op_params.output_multiplier;
      drv.qa_max = op_params.quantized_activation_max;
      drv.qa_min = op_params.quantized_activation_min;

      // Debugging
      drv.t.layer = dparams.layer;
      drv.p_t = p_t;
#ifdef DELEGATE_VERBOSE
      cout << "===========================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i] << endl;
      cout << "===========================" << endl;
#endif

      // Enter the driver code
      tflite_addsim::Entry(drv);
      p_t = drv.p_t;
      dparams.layer++;
      dparams.delegated_nodes--;
    }

    prf_end(0, p_t.delegate_total); // Stop the profiling delegate
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<OpData *> opdatas;
  std::vector<TfLiteAddParams *> cparams;

  std::vector<std::vector<int>> wgt_sum;
  std::vector<int> weight_offsets;
  std::vector<uint32_t *> del_weights;
  std::vector<int *> biases;
  std::vector<int *> crf;
  std::vector<int8_t *> crx;

private:
  const AddDelegateOptions options_;
};

// AddDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class AddDelegate : public SimpleDelegateInterface {
public:
  explicit AddDelegate(const AddDelegateOptions &options) : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {
    // Only supports FC ops
    if (kTfLiteBuiltinAdd != registration->builtin_code) return false;

    if (node->inputs->size != 2) return false;

    // This delegate only supports int8 types.
    for (int i = 0; i < 2; ++i) {
      auto &tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    TfLiteTensor input1 = context->tensors[node->inputs->data[0]];
    TfLiteTensor input2 = context->tensors[node->inputs->data[1]];

    if (!TfLiteIntArrayEqual(input1.dims, input2.dims)) return false;

    // Added node to the list of nodes to be delegated.
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "AddDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<AddDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const AddDelegateOptions options_;
};

} // namespace add_test
} // namespace tflite

AddDelegateOptions TfLiteAddDelegateOptionsDefault() {
  AddDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this add test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteAddDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteAddDelegateCreate(const AddDelegateOptions *options) {
  std::unique_ptr<tflite::add_test::AddDelegate> add(
      new tflite::add_test::AddDelegate(
          options ? *options : TfLiteAddDelegateOptionsDefault()));
  // return
  // tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(add));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(add), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `TfLiteAddDelegateCreate` call.
void TfLiteAddDelegateDelete(TfLiteDelegate *delegate) {
  SYSC_ON(profile.saveProfile(acc->profiling_vars));
  time_t now = time(0);
  tm *ltm = localtime(&now);
  std::string date =
      std::to_string(1900 + ltm->tm_year) + "-" +
      std::to_string(1 + ltm->tm_mon) + "-" + std::to_string(ltm->tm_mday) +
      "-" + std::to_string(ltm->tm_hour) + "-" + std::to_string(ltm->tm_min) +
      "-" + std::to_string(ltm->tm_sec);
  SYSC_ON(profile.saveCSVRecords(".data/" + std::string(DELEGATE_NAME) + "_" +
                                 std::to_string(DELEGATE_VERSION) + "_" +
                                 date));
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
  p_t.print();
  p_t.save_prf();
  std::cout << "===========================" << std::endl;
  std::cout << "Deleted" << std::endl;
  std::cout << "===========================" << std::endl;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}