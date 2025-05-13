
#include <fstream>
#include <iostream>
#include <utility>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif
#include "accelerator/driver/omni_driver.h"
#include "omni_delegate.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "util.h"
#include "util_prep.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

// Some variables needs to be defined across multiple instances of the delegate
unsigned int dma_addrs[1] = {dma_addr0};
unsigned int dma_addrs_in[1] = {dma_in0};
unsigned int dma_addrs_out[1] = {dma_out0};
struct OMNI_ACC_times p_t;
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
namespace omni_test {

// Omni delegate kernel.
class OmniDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit OmniDelegateKernel(const OmniDelegateOptions &options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {
    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      std::cout << "===========================" << std::endl;
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("OMNI_ACC");
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
      std::cout << "OMNI_ACC Accelerator";
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
    layers_params.resize(params->nodes_to_replace->size);

    int conv2d_count = 0;
    int fc_count = 0;
    int add_count = 0;
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
      // TfLiteAddParams *cparam =
      //     reinterpret_cast<TfLiteAddParams *>(delegated_node->builtin_data);
      // OpData *opdata = reinterpret_cast<OpData *>(delegated_node->user_data);
      layers_params[i] = delegated_node->builtin_data;
      opdatas[i] = delegated_node->user_data;
      if (builtin_code_[i] == kTfLiteBuiltinAdd) add_count++;
      if (builtin_code_[i] == kTfLiteBuiltinConv2d) conv2d_count++;
      if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) fc_count++;
    }

    // CONV2D/FC specific
    wt_sum.resize(conv2d_count + fc_count);
    temp_im2col.resize(conv2d_count);
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
      if (builtin_code_[i] == kTfLiteBuiltinAdd) {
        Prepare_ADD_INT8(context, node, i, layers_params[i], opdatas[i],
                         inputs_, outputs_);
      } else if (builtin_code_[i] == kTfLiteBuiltinConv2d) {
        Prepare_CONV2D_INT8(context, node, i, layers_params[i], opdatas[i],
                            inputs_, outputs_, out_tid, wt_sum[i],
                            temp_im2col[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) {
        Prepare_FC_INT8(context, node, i, layers_params[i], opdatas[i], inputs_,
                        outputs_, out_tid, wt_sum[i]);
      }
    }
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  // This function executes the operations required by node by offloading the
  // computation to the omni_driver For more info look into
  // "tensorflow/lite/kernels/add.cc" for the default implementation for Add
  // Nodes
  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    prf_start(0); // Start the profiling delegate
    int node_count = inputs_.size();
    struct acc_container drv;
    drv.acc = acc;
    drv.profile = &profile;
    drv.mdma = &mdma;
    drv.mt_context = &mt_context;
    drv.thread_count = context->recommended_num_threads;

    for (int i = 0; i < node_count; i++) {
      drv.op_type = builtin_code_[i];
      if (builtin_code_[i] == kTfLiteBuiltinAdd) { // ADD
        auto *params = layers_params[i];
        ADD_Data *data = reinterpret_cast<ADD_Data *>(opdatas[i]);
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

        const int8 *input1_data = input1->data.int8;
        const int8 *input2_data = input2->data.int8;
        int8 *output_data = output->data.int8;
        int size = 1;
        for (int i = 0; i < input1->dims->size; ++i) {
          size *= input1->dims->data[i];
        }

        for (int i = 0; i < size; ++i) {
          const int32 input1_val = op_params.input1_offset + input1_data[i];
          const int32 input2_val = op_params.input2_offset + input2_data[i];
          const int32 shifted_input1_val =
              input1_val * (1 << op_params.left_shift);
          const int32 shifted_input2_val =
              input2_val * (1 << op_params.left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, op_params.input1_multiplier,
                  op_params.input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, op_params.input2_multiplier,
                  op_params.input2_shift);
          const int32 raw_sum = scaled_input1_val + scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  raw_sum, op_params.output_multiplier,
                  op_params.output_shift) +
              op_params.output_offset;
          const int32 clamped_output = std::min(
              op_params.quantized_activation_max,
              std::max(op_params.quantized_activation_min, raw_output));
          output_data[i] = static_cast<int8>(clamped_output);
        }

        // Need to fix the accelerator for Add
        // Prepare Inputs for Driver/Accelerator
        // Accelerator Specific Parameters
        // drv.input_A = input1_data;
        // drv.input_B = input2_data;
        // drv.output_C = output_data;
        // int padded_size = roundUp(size, 4);
        // int8_t padded_input_A[padded_size];
        // int8_t padded_input_B[padded_size];
        // if (padded_size > size) {
        //   memcpy(padded_input_A, input1_data, size);
        //   memset(padded_input_A + size, 0, padded_size - size);
        //   memcpy(padded_input_B, input2_data, size);
        //   memset(padded_input_B + size, 0, padded_size - size);
        //   drv.input_A = padded_input_A;
        //   drv.input_B = padded_input_B;
        // }
        // drv.padded_size = padded_size;
        // drv.size = size;
        // drv.lshift = op_params.left_shift;
        // drv.in1_off = op_params.input1_offset;
        // drv.in1_sv = op_params.input1_shift;
        // drv.in1_mul = op_params.input1_multiplier;
        // drv.in2_off = op_params.input2_offset;
        // drv.in2_sv = op_params.input2_shift;
        // drv.in2_mul = op_params.input2_multiplier;
        // drv.out1_off = op_params.output_offset;
        // drv.out1_sv = op_params.output_shift;
        // drv.out1_mul = op_params.output_multiplier;
        // drv.qa_max = op_params.quantized_activation_max;
        // drv.qa_min = op_params.quantized_activation_min;

      } else if (builtin_code_[i] == kTfLiteBuiltinConv2d) { // CONV2D
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

        int8 *im2col_data = data->need_im2col ? &temp_im2col[i][0] : nullptr;

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

        // CONV2D Implementation algorithm
        int stride_height = params->stride_height;
        int stride_width = params->stride_width;
        int filter_height = filter->dims->data[1];
        int filter_width = filter->dims->data[2];
        int input_height = input->dims->data[1];
        int input_width = input->dims->data[2];
        int input_depth = input->dims->data[3];
        int output_height = output->dims->data[1];
        int output_width = output->dims->data[2];
        int output_channel = output->dims->data[3];
        const int8 *input_data = input->data.int8;
        const int8 *filter_data = filter->data.int8;
        int8 *output_data = output->data.int8;

        // Simple Convolution Algorithm
        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            for (int oc = 0; oc < output_channel; ++oc) {
              int32_t acc = 0;
              for (int fh = 0; fh < filter_height; ++fh) {
                for (int fw = 0; fw < filter_width; ++fw) {
                  for (int ic = 0; ic < input_depth; ++ic) {
                    int in_x = ow * stride_width + fw - data->padding.width;
                    int in_y = oh * stride_height + fh - data->padding.height;
                    if (in_x >= 0 && in_x < input_width && in_y >= 0 &&
                        in_y < input_height) {
                      int input_index =
                          ((in_y * input_width + in_x) * input_depth) + ic;
                      int filter_index =
                          (oc * filter_height * filter_width * input_depth) +
                          (fh * filter_width * input_depth) +
                          (fw * input_depth) + ic;

                      acc +=
                          input_data[input_index] * filter_data[filter_index];
                    }
                  }
                }
              }
              int wsum_offset = wt_sum[i][oc] * -input->params.zero_point;
              if (bias != nullptr) wsum_offset += bias->data.i32[oc];
              acc += wsum_offset;
              int out_shift = data->per_channel_output_shift.data()[oc];
              int out_mult = data->per_channel_output_multiplier.data()[oc];
              int out_offset = op_params.output_offset;
              int out_min = op_params.quantized_activation_min;
              int out_max = op_params.quantized_activation_max;
              acc = Quantised_Multiplier_S(acc, out_mult, out_shift, out_offset,
                                           out_min, out_max);
              int output_index =
                  ((oh * output_width) + ow) * output_channel + oc;
              output_data[output_index] = static_cast<int8_t>(acc);
            }
          }
        }
      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) { // FC
        auto *params = layers_params[i];
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
        const int32_t output_activation_min =
            op_params.quantized_activation_min;
        const int32_t output_activation_max =
            op_params.quantized_activation_max;

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
        const int o0 = output_shape.Dims(0);
        const int o1 = output_shape.Dims(1);

        const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
        const auto *affine_quantization =
            reinterpret_cast<TfLiteAffineQuantization *>(
                filter->quantization.params);
        TF_LITE_ENSURE(context, affine_quantization);
        TF_LITE_ENSURE(context, affine_quantization->scale);
        const int per_channel_quantization_size =
            affine_quantization->scale->size;
        const bool is_per_channel = per_channel_quantization_size > 1;

        int N = 1;
        int M = 1;
        for (int i = 0; i < input_shape.DimensionsCount() - 1; i++)
          N *= input_shape.Dims(i);
        for (int i = 0; i < filter_shape.DimensionsCount() - 1; i++)
          M *= filter_shape.Dims(i);
        int K = accum_depth;

        // MatMul + Bias + InSum + WgtSum + Requantize + Clamp
        for (int n = 0; n < N; n++) {
          for (int m = 0; m < M; m++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
              int in = input_data[n * K + k];
              int wt = filter_data[m * K + k];
              int mul = in * wt;
              sum += in * wt;
            }
            // add bias
            // sum += biases[i][m] + (wt_sum[m] * (-rhs_offset)) +
            //        (in_sum[n] * (-lhs_offset));
            sum += (wt_sum[i][m] * (-rhs_offset));
            if (bias != nullptr) sum += bias->data.i32[m];

            // re-quantize
            int out_shift = output_shift;
            int out_mult = output_multiplier;
            if (is_per_channel) {
              out_shift = data->per_channel_output_shift.data()[m];
              out_mult = data->per_channel_output_multiplier.data()[m];
            }
            int out_offset = op_params.output_offset;
            int out_min = op_params.quantized_activation_min;
            int out_max = op_params.quantized_activation_max;
            sum = Quantised_Multiplier_S(sum, out_mult, out_shift, out_offset,
                                         out_min, out_max);
            output_data[n * M + m] = sum;
          }
        }
      } else if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) { // DWCONV
      }

      // Debugging
      drv.t.layer = dparams.layer;
      drv.p_t = p_t;
#ifdef DELEGATE_VERBOSE
      cout << "===========================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i]
           << "      Type: " << builtin_code_[i] << endl;
      cout << "===========================" << endl;
#endif
      // Enter the driver code
      // if (builtin_code_[i] == kTfLiteBuiltinAdd) {
      //   tflite_omnisim::Entry(drv);
      // }
      p_t = drv.p_t;

      dparams.layer++;
      dparams.delegated_nodes--;
    }

    prf_end(0, p_t.delegate_total); // Stop the profiling delegate
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<void *> opdatas;
  std::vector<void *> layers_params;

  // Convolution specific variables
  std::vector<std::vector<int>> wt_sum;
  std::vector<std::vector<int8_t>> temp_im2col;

  // Add specific variables

private:
  const OmniDelegateOptions options_;
};

// OmniDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class OmniDelegate : public SimpleDelegateInterface {
public:
  explicit OmniDelegate(const OmniDelegateOptions &options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {

    bool isCONV2D = IsNode_CONV2D_INT8(registration, node, context);
    bool isDWCONV2D = IsNode_DWCONV2D_INT8(registration, node, context);
    bool isADD = IsNode_ADD_INT8(registration, node, context);
    bool isFC = IsNode_FC_INT8(registration, node, context);
    bool isTCONV = IsNode_TCONV_INT8(registration, node, context);

    // Check if the node is supported by the delegate
    // bool supported_nodes = [isCONV2D, isDWCONV2D, isADD, isFC, isTCONV];
    // std::vector<bool> supported_nodes = {isCONV2D, isDWCONV2D, isADD, isFC,
    // isTCONV};
    // std::vector<bool> supported_nodes = {isADD};
    // std::vector<bool> supported_nodes = {isCONV2D, isFC};
    // std::vector<bool> supported_nodes = {isCONV2D, isADD};

    std::vector<bool> supported_nodes = {isCONV2D, isFC, isADD};
    // std::vector<bool> supported_nodes = {isCONV2D};

    bool delegated_node = false;
    for (int i = 0; i < supported_nodes.size(); i++) {
      if (supported_nodes[i]) {
        delegated_node = true;
        break;
      }
    }

    if (delegated_node) dparams.delegated_nodes++;
    return delegated_node;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "OmniDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<OmniDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const OmniDelegateOptions options_;
};

} // namespace omni_test
} // namespace tflite

OmniDelegateOptions TfLiteOmniDelegateOptionsDefault() {
  OmniDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this omni test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteOmniDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteOmniDelegateCreate(const OmniDelegateOptions *options) {
  std::unique_ptr<tflite::omni_test::OmniDelegate> omni(
      new tflite::omni_test::OmniDelegate(
          options ? *options : TfLiteOmniDelegateOptionsDefault()));
  // return
  // tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(omni));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(omni), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `TfLiteOmniDelegateCreate` call.
void TfLiteOmniDelegateDelete(TfLiteDelegate *delegate) {
  SYSC_ON(profile.saveProfile(acc->profiling_vars));
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
