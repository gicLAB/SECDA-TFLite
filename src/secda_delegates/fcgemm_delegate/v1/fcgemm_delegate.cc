// FCGEMM Delegate (Sim tested with some errors || FPGA not tested or integrated)
#include "tensorflow/lite/delegates/utils/secda_delegates/fcgemm_delegate/v1/fcgemm_delegate.h"

#include <fstream>
#include <iostream>
#include <utility>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif
#include "accelerator/driver/fc_driver.h"
#include "fcgemm_delegate.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "util.h"

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#define DELEGATE_NAME "FCGEMM"
#define DELEGATE_VERSION 1
// Some variables needs to be defined across multiple instances of the delegate
struct FCGEMM_times fc_t;
struct del_params dparams;
static struct Profile profile;
struct MultiThreadContext mt_context;

#ifdef SYSC
ACCNAME *acc;
struct sysC_sigs *scs;
#else
int *acc;
unsigned long long *insn_mem;
unsigned long long *inp_mem;
unsigned long long *wgt_mem;
unsigned long long *bias_mem;
unsigned int *out_mem;
#endif

namespace tflite {
namespace fcgemm_test {

// FCGEMM delegate kernel.
class FCGEMMDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit FCGEMMDelegateKernel(const FCGEMMDelegateOptions &options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {

    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      std::cout << "===========================" << std::endl;
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("FCGEMM_ACC");
      sysC_init();
      sysC_binder(&_acc, &scs1);
      acc = &_acc;
      scs = &scs1;
      std::cout << "Initialised the SystemC Modules" << std::endl;
#else
      dparams.acc = getAccBaseAddress<int>(acc_address, 65536);
      insn_mem = mm_alloc_rw<unsigned long long>(insn_addr, MM_BL);
      inp_mem = mm_alloc_rw<unsigned long long>(in_addr, MM_BL);
      wgt_mem = mm_alloc_rw<unsigned long long>(wgt_addr, MM_BL * 4);
      bias_mem = mm_alloc_rw<unsigned long long>(bias_addr, MM_BL * 4);
      out_mem = mm_alloc_r<unsigned int>(out_addr, MM_BL * 4);
      // Update as required
      writeMappedReg<int>(dparams.acc, 0x14, 0);
      writeMappedReg<int>(dparams.acc, 0x24, 1);
      writeMappedReg<int>(dparams.acc, 0x34, insn_address / 8);
      writeMappedReg<int>(dparams.acc, 0x3c, in_address / 8);
      writeMappedReg<int>(dparams.acc, 0x44, wgt_address / 8);
      writeMappedReg<int>(dparams.acc, 0x4c, bias_address / 8);
      writeMappedReg<int>(dparams.acc, 0x54, out_address / 4);
      writeMappedReg<int>(dparams.acc, 0x24, 0);
      std::cout << "Memory Mapped Buffers" << std::endl;
#endif
      scs->sig_start_acc = 0;
      scs->sig_done_acc = 0;
      scs->sig_reset_acc = 0;
      scs->sig_insn_addr = 0;
      scs->sig_input_addr = 0;
      scs->sig_weight_addr = 0;
      scs->sig_bias_addr = 0;
      scs->sig_output_addr = 0;
      std::cout << "===========================" << std::endl;
      std::cout << "FC_ACC";
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
    biases.resize(params->nodes_to_replace->size);
    biases_d.resize(params->nodes_to_replace->size);

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
      inputs_[i].push_back(delegated_node->inputs->data[2]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
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
      biases_d[i].resize(filter->dims->data[0]);
      if (inputs_[i].size() == 3 && inputs_[i][2] >= 0) {
        GetInputSafe(context, inputs_[i][2], &bias);
        biases[i] = bias->data.i32;
      } else {
        for (int j = 0; j < filter->dims->data[0]; j++) {
          biases_d[i][j] = 0;
        }
        biases[i] = &biases_d[i][0];
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

      int N = batch_size;
      int M = num_units;
      int K = filter->dims->data[1];
      int rfactor = 16;
      int pN = roundUp(N, rfactor);
      int pM = roundUp(M, rfactor);
      int pK = roundUp(K, rfactor);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    prf_start(0);
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      auto *params = cparams[i];
      OpData *data = opdatas[i];
      const TfLiteTensor *input;
      const TfLiteTensor *filter;
      TfLiteTensor *output;
      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      GetOutputSafe(context, outputs_[i][0], &output);

      const TfLiteTensor *bias;
      bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
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

      int task_count = context->recommended_num_threads;
      int N = batches;
      int M = output_depth;
      int K = accum_depth;
      int rfactor = 16;
      int pN = roundUp(N, rfactor);
      int pM = roundUp(M, rfactor);
      int pK = roundUp(K, rfactor);

      std::vector<int> in_sum;
      std::vector<int> wt_sum;
      int *idims = input->dims->data;
      int *wdims = filter->dims->data;

#ifdef SYSC
      int8_t *padded_input = new int8_t[pN * pK];
      int8_t *padded_weights = new int8_t[pM * pK];
      int8_t *padded_output = new int8_t[pM * pN];
#else
      int8_t *padded_input = (int8_t *)&inp_mem[0];
      int8_t *padded_weights = (int8_t *)&wgt_mem[0];
      int8_t *padded_output = (int8_t *)&out_mem[0];
#endif

      prf_start(1);
      precal_sum_load_pad(input->data.int8, N, K, padded_input, in_sum);
      precal_sum_load_pad(filter->data.int8, M, K, padded_weights, wt_sum);
      prf_end(1, fc_t.p_pack);

      // acc_container is used to wrap all the paramters the
      // fc_driver/accelerator needs from the delegate
      struct acc_container drv;
      drv.acc = acc;
      drv.scs = scs;
      drv.profile = &profile;
      drv.mt_context = &mt_context;
      drv.thread_count = context->recommended_num_threads;

// Accelerator Specific Parameters
#ifndef SYSC
      drv.insn_mem = insn_mem;
      drv.bias_mem = bias_mem;
#endif
      drv.pN = pN;
      drv.pM = pM;
      drv.pK = pK;
      drv.N = N;
      drv.M = M;
      drv.K = K;
      drv.padded_input = padded_input;
      drv.padded_weights = padded_weights;
      drv.padded_output = padded_output;
      drv.output_data = output_data;
      drv.in_sum = &in_sum[0];
      drv.wt_sum = &wt_sum[0];
      drv.crx = output_shift;
      drv.crf = output_multiplier;
      drv.ra = output_offset;
      drv.rhs_offset = -rhs_offset;
      drv.lhs_offset = -lhs_offset;
      drv.bias = biases[i];

      // Debugging
      drv.t.layer = dparams.layer;
      drv.t2 = fc_t;
#ifdef DELEGATE_VERBOSE
      cout << "===========================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i] << endl;
      cout << "===========================" << endl;
#endif

      // Enter the driver code
      drv.start_count = dparams.start_count;
      tflite_fcgemm::Entry(drv);
      dparams.start_count = drv.start_count;
      fc_t = drv.t2;

      // Calls the fc_driver unpack/unpad result to TFLite tensor
      dparams.layer++;
      dparams.delegated_nodes--;
    }

    prf_end(0, fc_t.t_conv_total);
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;

  std::vector<OpData *> opdatas;
  std::vector<TfLiteFullyConnectedParams *> cparams;
  std::vector<int *> biases;
  std::vector<std::vector<int>> biases_d;

private:
  const FCGEMMDelegateOptions options_;
}; // namespace fcgemm_test

// FCGEMMDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class FCGEMMDelegate : public SimpleDelegateInterface {
public:
  explicit FCGEMMDelegate(const FCGEMMDelegateOptions &options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {
    // Only supports FC ops
    if (kTfLiteBuiltinFullyConnected != registration->builtin_code)
      return false;

    if (node->inputs->size != 3 && node->inputs->size != 2) return false;
    // This delegate only supports int8 types.
    for (int i = 0; i < 2; ++i) {
      auto &tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
      auto &tensor = context->tensors[node->inputs->data[2]];
      if (tensor.type != kTfLiteInt32 && tensor.type <= 16) return false;
    }

    // FC
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "FCGEMMDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<FCGEMMDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const FCGEMMDelegateOptions options_;
};

} // namespace fcgemm_test
} // namespace tflite

FCGEMMDelegateOptions TfLiteFCGEMMDelegateOptionsDefault() {
  FCGEMMDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this fcgemm test delegate
  // will not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteFCGEMMDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *
TfLiteFCGEMMDelegateCreate(const FCGEMMDelegateOptions *options) {
  std::unique_ptr<tflite::fcgemm_test::FCGEMMDelegate> fcgemm(
      new tflite::fcgemm_test::FCGEMMDelegate(
          options ? *options : TfLiteFCGEMMDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(fcgemm), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `TfLiteFCGEMMDelegateCreate` call.
void TfLiteFCGEMMDelegateDelete(TfLiteDelegate *delegate) {
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
  fc_t.print();
  fc_t.save_prf();
  std::cout << "===========================" << std::endl;
  std::cout << "Deleted" << std::endl;
  std::cout << "===========================" << std::endl;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
