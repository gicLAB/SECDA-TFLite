#include <fstream>
#include <iostream>
#include <utility>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif
#include "accelerator/driver/omni_driver.h"
#include "omni_delegate.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "util.h"
#include "util_prep.h"

#define DELEGATE_NAME "OMNI"
#define DELEGATE_VERSION 1

// Some variables needs to be defined across multiple instances of the delegate
unsigned int dma_addrs[1] = {dma_addr0};
unsigned int dma_addrs_in[1] = {dma_in0};
unsigned int dma_addrs_out[1] = {dma_out0};
struct OMNI_ACC_times p_t;
struct del_params dparams;
static struct Profile profile;
struct MultiThreadContext mt_context;

// COPILOT NOTE: `dtm` tracks delegated tensors by TfLite tensor
// id and manages DMA-backed storage/mapping decisions across Prepare/Eval.
struct DMATensorManager dtm;

#ifdef SYSC
static struct s_mdma mdma(1, dma_addrs);
// Shared backing store for the DMA driver instruction stream
static struct mm_buf_int dbuf1(dma_in0, MM_BL / 4);
// Isolated backing store for the Tensor Manager to prevent collisions
static struct mm_buf_int dbuf2(dma_in1, MM_BL / 4); 

ACCNAME* acc;
struct sysC_sigs* scs;
#else
struct s_mdma mdma(1, dma_addrs);
struct mm_buf_int dbuf1(dma_in0, MM_BL / 4);
struct mm_buf_int dbuf2(dma_in1, MM_BL / 4); 

int* acc;
#endif

// COPILOT NOTE: When true, op kernels read/write through
// `DMATensor::data` instead of direct TfLite tensor buffers.
bool use_dma_buffer = true;

namespace tflite {
namespace omni_test {

// Omni delegate kernel.
class OmniDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit OmniDelegateKernel(const OmniDelegateOptions& options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      std::cout << "===========================" << std::endl;
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("OMNI_ACC");
      sysC_init();
      sysC_binder(&_acc, &mdma, &scs1);
      
      mdma.dmas[0].dma_bind_input_buffer(&dbuf1);
      mdma.dmas[0].dma_bind_output_buffer(&dbuf1);
      
      // Bind the tensor manager to the isolated buffer
      dtm.set_buffer(dbuf2.get_buffer(), dbuf2.size);
      
      acc = &_acc;
      scs = &scs1;
      std::cout << "Initialised the SystemC Modules" << std::endl;
#else
      dparams.acc = getAccBaseAddress<int>(acc_ctrl_address, 65536);
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

    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    opdatas.resize(params->nodes_to_replace->size);
    layers_params.resize(params->nodes_to_replace->size);
    is_global_output.resize(params->nodes_to_replace->size);
    output_dependencies.resize(params->nodes_to_replace->size);
    node_output_needed.resize(params->nodes_to_replace->size);
    omni_tensor_ids.resize(params->nodes_to_replace->size);

    int conv2d_count = 0;
    int fc_count = 0;
    int add_count = 0;
    int dwconv2d_count = 0;
    int tconv_count = 0;
    int shape_count = 0;
    int softmax_count = 0;
    int pad_count = 0;
    int mean_count = 0;
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
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
      
      // Allocate isolated memory to prevent TFLite heap corruption
      if (builtin_code_[i] == kTfLiteBuiltinAdd) opdatas[i] = new ADD_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinConv2d) opdatas[i] = new Conv2D_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) opdatas[i] = new FC_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) opdatas[i] = new DWCONV2D_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinTransposeConv) opdatas[i] = new TCONV_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinSoftmax) opdatas[i] = new SOFTMAX_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinMean) opdatas[i] = new REDUCE_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinQuantize) opdatas[i] = new QUANTIZE_Data();
      else if (builtin_code_[i] == kTfLiteBuiltinDequantize) opdatas[i] = new DEQUANTIZE_Data();
      else opdatas[i] = nullptr;

      if (builtin_code_[i] == kTfLiteBuiltinAdd) add_count++;
      if (builtin_code_[i] == kTfLiteBuiltinConv2d) conv2d_count++;
      if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) fc_count++;
      if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) dwconv2d_count++;
      if (builtin_code_[i] == kTfLiteBuiltinTransposeConv) tconv_count++;
      if (builtin_code_[i] == kTfLiteBuiltinShape) shape_count++;
      if (builtin_code_[i] == kTfLiteBuiltinSoftmax) softmax_count++;
      if (builtin_code_[i] == kTfLiteBuiltinPad) pad_count++;
      if (builtin_code_[i] == kTfLiteBuiltinMean) mean_count++;
    }
    wt_sum.resize(params->nodes_to_replace->size);
    omni_im2col.resize(params->nodes_to_replace->size);
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
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
      
      for (int j = i; j < node_count; j++)
        for (int k = 0; k < inputs_[j].size(); k++)
          if (inputs_[j][k] == current_output) future_nodes.push_back(j);
      output_dependencies[i] = future_nodes;
      node_output_needed[i] = output_dependencies[i].size() > 0;
      // =======================================================

      if (builtin_code_[i] == kTfLiteBuiltinAdd) {
        Prepare_ADD_INT8(context, node, i, layers_params[i], opdatas[i],
                         inputs_, outputs_, out_tid);
      } else if (builtin_code_[i] == kTfLiteBuiltinConv2d) {
        Prepare_CONV2D_INT8(context, node, i, layers_params[i], opdatas[i],
                            inputs_, outputs_, out_tid, wt_sum[i],
                            omni_im2col[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) {
        Prepare_FC_INT8(context, node, i, layers_params[i], opdatas[i], inputs_,
                        outputs_, out_tid, wt_sum[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) {
        Prepare_DWCONV2D_INT8(context, node, i, layers_params[i], opdatas[i],
                              inputs_, outputs_, out_tid, wt_sum[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinTransposeConv) {
        Prepare_TCONV_INT8(context, node, i, layers_params[i], opdatas[i],
                           inputs_, outputs_, out_tid, wt_sum[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinShape) {
        Prepare_SHAPE_INT8(context, node, i, layers_params[i], opdatas[i],
                           inputs_, outputs_, out_tid);
      } else if (builtin_code_[i] == kTfLiteBuiltinSoftmax) {
        Prepare_SOFTMAX_INT8(context, node, i, layers_params[i], opdatas[i],
                             inputs_, outputs_, out_tid);
      } else if (builtin_code_[i] == kTfLiteBuiltinPad) {
        Prepare_PAD_INT8(context, node, i, layers_params[i], opdatas[i],
                         inputs_, outputs_, out_tid);
      } else if (builtin_code_[i] == kTfLiteBuiltinMean) {
        Prepare_MEAN_INT8(context, node, i, layers_params[i], opdatas[i],
                          inputs_, outputs_, out_tid, omni_tensor_ids[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinQuantize) {
        Prepare_QUANTIZE_INT8(context, node, i, layers_params[i], opdatas[i],
                              inputs_, outputs_, out_tid, omni_tensor_ids[i]);
      } else if (builtin_code_[i] == kTfLiteBuiltinDequantize) {
        Prepare_DEQUANTIZE_INT8(context, node, i, layers_params[i], opdatas[i],
                                inputs_, outputs_, out_tid, omni_tensor_ids[i]);
      }

      // Tensor-manager registration pass
      for (int j = 0; j < inputs_[i].size(); j++) {
        unsigned int TID = inputs_[i][j];
        if (TID == -1) continue;
        const TfLiteTensor* tensor;
        GetInputSafe(context, TID, &tensor);
        int ranks = tensor->dims->size;
        vector<unsigned int> dims;
        for (int k = 0; k < ranks; k++) dims.push_back(tensor->dims->data[k]);
        int bitwidth = tensor->type == kTfLiteInt8 ? 8 : 32;
        bool isStatic = tensor->allocation_type == kTfLiteMmapRo;
        bool isExternalInput = !isStatic;
        bool isExternalOutput = false;
        if (dtm.check_exists(TID) == false) {
          DMATensor t(TID, ranks, dims, bitwidth, isStatic, isExternalInput,
                      isExternalOutput);
          t.input_nodes.push_back(associated_nodes[i]);
          dtm.add_allocate_tensor(t, tensor->data.raw);
        } else {
          DMATensor* t = dtm.get_tensor(TID);
          t->input_nodes.push_back(associated_nodes[i]);
        }
      }

      for (int j = 0; j < outputs_[i].size(); j++) {
        unsigned int TID = outputs_[i][j];
        TfLiteTensor* tensor;
        GetOutputSafe(context, TID, &tensor);
        int ranks = tensor->dims->size;
        vector<unsigned int> dims;
        for (int k = 0; k < ranks; k++) dims.push_back(tensor->dims->data[k]);
        int bitwidth = tensor->type == kTfLiteInt8 ? 8 : 32;
        bool isStatic = tensor->allocation_type == kTfLiteMmapRo || 
                        tensor->allocation_type == kTfLitePersistentRo;
        bool isExternalInput = false;
        bool isExternalOutput = false;
        for (int k = 0; k < node->outputs->size; k++)
          isExternalOutput =
              isExternalOutput || (node->outputs->data[k] == TID);
        if (dtm.check_exists(TID) == false) {
          DMATensor t(TID, ranks, dims, bitwidth, isStatic, isExternalInput,
                      isExternalOutput);
          t.output_nodes.push_back(associated_nodes[i]);
          dtm.add_allocate_tensor(t, tensor->data.raw);
        } else {
          DMATensor* t = dtm.get_tensor(TID);
          t->output_nodes.push_back(associated_nodes[i]);
        }
      }
    }
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    prf_start(0);  // Start the profiling delegate
    int node_count = inputs_.size();
    struct acc_container drv;
    drv.acc = acc;
    drv.profile = &profile;
    drv.mdma = &mdma;
    drv.mt_context = &mt_context;
    drv.thread_count = context->recommended_num_threads;

    int prv_dst_route = 0;

    for (int i = 0; i < node_count; i++) {
      drv.op_type = builtin_code_[i];
      cout << "======================================================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i]
           << "      Type: " << EnumNamesBuiltinOperator()[builtin_code_[i]]
           << endl;

      // =======================================================
      // Preload Stage & Logging
      // =======================================================
      if (use_dma_buffer) {
        for (int j = 0; j < inputs_[i].size(); j++) {
          unsigned int TID = inputs_[i][j];
          const TfLiteTensor* tensor;
          GetInputSafe(context, TID, &tensor);
          if (TID == -1) continue;
          DMATensor* t = dtm.get_tensor(TID);
          if (t->isStatic) {
            // Static Weights and Biases (silenced to keep log clean)
          } else if (!t->mmapped && t->isExternalInput) {
            cout << "  [Input]  TID " << TID << " : Mapped (Host CPU -> DMA Buffer)" << endl;
            dtm.copy_to_mmap(TID, tensor->data.raw);
          } else {
            cout << "  [Input]  TID " << TID << " : Read   (Resident in DMA Buffer)" << endl;
          }
        }
      }

      // =========================================================
      // Hardware Routing & Fusion Scheduler (Abstracted)
      // =========================================================
      drv.src_route = 0;
      drv.dst_route = 0;
      drv.t.layer = dparams.layer;

      bool is_hw_offloaded = (builtin_code_[i] == kTfLiteBuiltinAdd);

      if (i > 0 && is_hw_offloaded) {
        int primary_input = inputs_[i][0];
        int prev_output = outputs_[i-1][0];
        if (primary_input == prev_output && prv_dst_route == 1) {
          drv.src_route = 1;
        }
      }

      if (i + 1 < node_count && is_hw_offloaded) {
        int current_output = outputs_[i][0];
        bool next_needs_it = false;
        for (int next_in : inputs_[i+1]) {
          if (next_in == current_output) {
            next_needs_it = true;
            break;
          }
        }
        if (next_needs_it && !is_global_output[i]) {
          drv.dst_route = 1;
        }
      }

      prv_dst_route = drv.dst_route;
      
      cout << "  [Route]  Src: " << (drv.src_route == 1 ? "On-Chip BRAM" : "Host DMA") 
           << " | Dst: " << (drv.dst_route == 1 ? "On-Chip BRAM" : "Host DMA") << endl;

      // =======================================================================
      // Operation Evaluation
      // =======================================================================
      if (builtin_code_[i] == kTfLiteBuiltinAdd) {  // ADD
        TfLiteAddParams* params =
            reinterpret_cast<TfLiteAddParams*>(layers_params[i]);
        ADD_Data* data = reinterpret_cast<ADD_Data*>(opdatas[i]);
        const TfLiteTensor* input1;
        const TfLiteTensor* input2;
        TfLiteTensor* output;
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

        const int8* input1_data;
        const int8* input2_data;
        int8* output_data;
        if (use_dma_buffer) {
          DMATensor* t1 = dtm.get_tensor(inputs_[i][0]);
          DMATensor* t2 = dtm.get_tensor(inputs_[i][1]);
          DMATensor* t3 = dtm.get_tensor(outputs_[i][0]);
          input1_data = reinterpret_cast<const int8*>(t1->data);
          input2_data = reinterpret_cast<const int8*>(t2->data);
          output_data = reinterpret_cast<int8*>(t3->data);
        } else {
          input1_data = input1->data.int8;
          input2_data = input2->data.int8;
          output_data = output->data.int8;
        }

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

        drv.input_A = input1_data;
        drv.input_B = input2_data;
        drv.output_C = output_data;
        int padded_size = roundUp(size, 4);
        int8_t padded_input_A[padded_size];
        int8_t padded_input_B[padded_size];
        if (padded_size > size) {
          memcpy(padded_input_A, input1_data, size);
          memset(padded_input_A + size, 0, padded_size - size);
          memcpy(padded_input_B, input2_data, size);
          memset(padded_input_B + size, 0, padded_size - size);
          drv.input_A = padded_input_A;
          drv.input_B = padded_input_B;
        }
        drv.padded_size = padded_size;
        drv.size = size;
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

        drv.p_t = p_t;
        omni_driver_space::Entry(drv);
        p_t = drv.p_t;

      } else if (builtin_code_[i] == kTfLiteBuiltinConv2d) {  // CONV2D
        TfLiteConvParams* params =
            reinterpret_cast<TfLiteConvParams*>(layers_params[i]);
        Conv2D_Data* data = reinterpret_cast<Conv2D_Data*>(opdatas[i]);

        TfLiteTensor* output;
        const TfLiteTensor* input;
        const TfLiteTensor* filter;
        const TfLiteTensor* bias = nullptr;

        GetInputSafe(context, inputs_[i][0], &input);
        GetInputSafe(context, inputs_[i][1], &filter);
        GetOutputSafe(context, outputs_[i][0], &output);

        bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
        if (isBias) {
          GetInputSafe(context, inputs_[i][2], &bias);
        }

        int8* im2col_data = data->need_im2col ? &omni_im2col[i][0] : nullptr;

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

        int32_t input_offset = op_params.input_offset;
        int32_t output_offset = op_params.output_offset;
        
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

        int pad_width = data->padding.height;
        int pad_height = data->padding.width;

        const int8* input_data = nullptr;
        const int8* filter_data = nullptr;
        const int32* bias_data = nullptr;
        int8* output_data = nullptr;

        if (use_dma_buffer) {
          DMATensor* t1 = dtm.get_tensor(inputs_[i][0]);
          DMATensor* t2 = dtm.get_tensor(inputs_[i][1]);
          DMATensor* t4 = dtm.get_tensor(outputs_[i][0]);

          input_data = reinterpret_cast<const int8*>(t1->data);
          filter_data = reinterpret_cast<const int8*>(t2->data);
          output_data = reinterpret_cast<int8*>(t4->data);
          
          if (isBias) {
            DMATensor* t3 = dtm.get_tensor(inputs_[i][2]);
            bias_data = reinterpret_cast<const int32*>(t3->data);
          }
        } else {
          input_data = input->data.int8;
          filter_data = filter->data.int8;
          output_data = output->data.int8;
          bias_data = (bias != nullptr ? reinterpret_cast<const int32*>(bias->data.raw) : nullptr);
        }

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

                      int8_t input_val = input_data[input_index];
                      int8_t filter_val = filter_data[filter_index];

                      acc += (input_data[input_index] + input_offset) * filter_data[filter_index];
                    }
                  }
                }
              }

              if (bias_data != nullptr) acc += bias_data[oc];
              int out_shift = data->per_channel_output_shift.data()[oc];
              int out_mult = data->per_channel_output_multiplier.data()[oc];
              int out_offset = op_params.output_offset;
              int out_min = op_params.quantized_activation_min;
              int out_max = op_params.quantized_activation_max;
              acc = Quantised_Multiplier_V1(acc, out_mult, out_shift,
                                            out_offset, out_min, out_max);
              int output_index =
                  ((oh * output_width) + ow) * output_channel + oc;
              output_data[output_index] = static_cast<int8_t>(acc);
            }
          }
        }
        
        // drv.p_t = p_t;
        // omni_driver_space::Entry(drv);
        // p_t = drv.p_t;

      } else if (builtin_code_[i] == kTfLiteBuiltinFullyConnected) {  // FC
        TfLiteFullyConnectedParams* params =
            reinterpret_cast<TfLiteFullyConnectedParams*>(layers_params[i]);
        FC_Data* data = reinterpret_cast<FC_Data*>(opdatas[i]);
        const TfLiteTensor* input;
        const TfLiteTensor* filter;
        const TfLiteTensor* bias = nullptr;
        TfLiteTensor* output;
        GetInputSafe(context, inputs_[i][0], &input);
        GetInputSafe(context, inputs_[i][1], &filter);
        GetOutputSafe(context, outputs_[i][0], &output);

        bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
        if (isBias) {
          GetInputSafe(context, inputs_[i][2], &bias);
        }

        const int8* input_data = nullptr;
        const int8* filter_data = nullptr;
        const int32* bias_data = nullptr;
        int8* output_data = nullptr;

        if (use_dma_buffer) {
          DMATensor* t1 = dtm.get_tensor(inputs_[i][0]);
          DMATensor* t2 = dtm.get_tensor(inputs_[i][1]);
          DMATensor* t4 = dtm.get_tensor(outputs_[i][0]);
          input_data = reinterpret_cast<const int8*>(t1->data);
          filter_data = reinterpret_cast<const int8*>(t2->data);
          output_data = reinterpret_cast<int8*>(t4->data);
          if (isBias) {
            DMATensor* t3 = dtm.get_tensor(inputs_[i][2]);
            bias_data = reinterpret_cast<const int32*>(t3->data);
          }
        } else {
          input_data = input->data.int8;
          filter_data = filter->data.int8;
          output_data = output->data.int8;
          bias_data = (bias != nullptr ? reinterpret_cast<const int32*>(bias->data.raw) : nullptr);
        }

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
        const int32_t weight_offset = op_params.weights_offset;
        const int32_t input_offset = op_params.input_offset;
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
        const auto* affine_quantization =
            reinterpret_cast<TfLiteAffineQuantization*>(
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
              int in = input_data[n * K + k] + input_offset;
              int wt = filter_data[m * K + k] + weight_offset;
              sum += in * wt;
            }

            int out_shift = output_shift;
            int out_mult = output_multiplier;
            if (is_per_channel) {
              out_shift = data->per_channel_output_shift.data()[m];
              out_mult = data->per_channel_output_multiplier.data()[m];
            }
            if (bias_data != nullptr) sum += bias_data[m];

            int out_offset = op_params.output_offset;
            int out_min = op_params.quantized_activation_min;
            int out_max = op_params.quantized_activation_max;
            sum = Quantised_Multiplier_V1(sum, out_mult, out_shift, out_offset,
                                          out_min, out_max);

            output_data[n * M + m] = static_cast<int8_t>(sum);
          }
        }
        
        // drv.p_t = p_t;
        // omni_driver_space::Entry(drv);
        // p_t = drv.p_t;

      } else if (builtin_code_[i] == kTfLiteBuiltinDepthwiseConv2d) {  // DWCONV
        TfLiteDepthwiseConvParams* params =
            reinterpret_cast<TfLiteDepthwiseConvParams*>(layers_params[i]);
        DWCONV2D_Data* data = reinterpret_cast<DWCONV2D_Data*>(opdatas[i]);

        TfLiteTensor* output;
        const TfLiteTensor* input;
        const TfLiteTensor* filter;
        const TfLiteTensor* bias = nullptr;

        GetInputSafe(context, inputs_[i][0], &input);
        GetInputSafe(context, inputs_[i][1], &filter);
        GetOutputSafe(context, outputs_[i][0], &output);
        
        bool isBias = (inputs_[i].size() == 3 && inputs_[i][2] >= 0);
        if (isBias) {
          GetInputSafe(context, inputs_[i][2], &bias);
        }

        DepthwiseParams op_params;
        op_params.padding_type = PaddingType::kSame;
        op_params.padding_values.width = data->padding.width;
        op_params.padding_values.height = data->padding.height;
        op_params.stride_width = params->stride_width;
        op_params.stride_height = params->stride_height;
        op_params.dilation_width_factor = params->dilation_width_factor;
        op_params.dilation_height_factor = params->dilation_height_factor;
        op_params.input_offset = -input->params.zero_point;
        op_params.weights_offset = 0;
        op_params.output_offset = output->params.zero_point;
        op_params.quantized_activation_min = data->output_activation_min;
        op_params.quantized_activation_max = data->output_activation_max;
        TF_LITE_ENSURE_STATUS(ComputeDepthMultiplier(
            context, input, filter, &op_params.depth_multiplier));

        const int stride_width = op_params.stride_width;
        const int stride_height = op_params.stride_height;
        const int dilation_width_factor = op_params.dilation_width_factor;
        const int dilation_height_factor = op_params.dilation_height_factor;
        const int pad_width = op_params.padding_values.width;
        const int pad_height = op_params.padding_values.height;
        const int depth_multiplier = op_params.depth_multiplier;
        const int32_t input_offset = op_params.input_offset;
        const int32_t output_offset = op_params.output_offset;
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

        TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
        TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
        TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
        TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
        const int batches = MatchingDim(input_shape, 0, output_shape, 0);
        const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
        const int input_height = input_shape.Dims(1);
        const int input_width = input_shape.Dims(2);
        const int input_depth = input_shape.Dims(3);
        const int filter_height = filter_shape.Dims(1);
        const int filter_width = filter_shape.Dims(2);
        const int output_height = output_shape.Dims(1);
        const int output_width = output_shape.Dims(2);
        TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);

        const int8* input_data = nullptr;
        const int8* filter_data = nullptr;
        const int32* bias_data = nullptr;
        int8* output_data = nullptr;

        if (use_dma_buffer) {
          DMATensor* t1 = dtm.get_tensor(inputs_[i][0]);
          DMATensor* t2 = dtm.get_tensor(inputs_[i][1]);
          DMATensor* t4 = dtm.get_tensor(outputs_[i][0]);
          
          input_data = reinterpret_cast<const int8*>(t1->data);
          filter_data = reinterpret_cast<const int8*>(t2->data);
          output_data = reinterpret_cast<int8*>(t4->data);
          
          if (isBias) {
            DMATensor* t3 = dtm.get_tensor(inputs_[i][2]);
            bias_data = reinterpret_cast<const int32*>(t3->data);
          }
        } else {
          input_data = input->data.int8;
          filter_data = filter->data.int8;
          output_data = output->data.int8;
          bias_data = (bias != nullptr ? reinterpret_cast<const int32*>(bias->data.raw) : nullptr);
        }

        for (int batch = 0; batch < batches; ++batch) {
          for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                for (int m = 0; m < depth_multiplier; ++m) {
                  const int output_channel = m + in_channel * depth_multiplier;
                  const int in_x_origin = (out_x * stride_width) - pad_width;
                  const int in_y_origin = (out_y * stride_height) - pad_height;
                  int32_t acc = 0;
                  for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                    for (int filter_x = 0; filter_x < filter_width;
                         ++filter_x) {
                      const int in_x =
                          in_x_origin + dilation_width_factor * filter_x;
                      const int in_y =
                          in_y_origin + dilation_height_factor * filter_y;

                      const bool is_point_inside_image =
                          (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                          (in_y < input_height);
                      if (is_point_inside_image) {
                        int32_t input_val = input_data[Offset(
                            input_shape, batch, in_y, in_x, in_channel)];
                        int32_t filter_val =
                            filter_data[Offset(filter_shape, 0, filter_y,
                                               filter_x, output_channel)];
                        acc += filter_val * (input_val + input_offset);
                      }
                    }
                  }
                  if (bias_data != nullptr) acc += bias_data[output_channel];
                  int out_shift =
                      data->per_channel_output_shift.data()[output_channel];
                  int out_mult = data->per_channel_output_multiplier
                                     .data()[output_channel];
                  acc = Quantised_Multiplier_V2(
                      acc, out_mult, out_shift, output_offset,
                      output_activation_min, output_activation_max);
                  output_data[Offset(output_shape, batch, out_y, out_x,
                                     output_channel)] =
                      static_cast<int8_t>(acc);
                }
              }
            }
          }
        }
      } else if (builtin_code_[i] == kTfLiteBuiltinTransposeConv) {  // TCONV
        TfLiteTransposeConvParams* params =
            reinterpret_cast<TfLiteTransposeConvParams*>(layers_params[i]);
        TCONV_Data* data = reinterpret_cast<TCONV_Data*>(opdatas[i]);
        const TfLiteTensor* input;
        const TfLiteTensor* weights;
        TfLiteTensor* output;
        const TfLiteTensor* bias;
        const TfLiteTensor* output_shape_tensor;

        bool has_bias = inputs_[i][3] != 0;

        TfLiteTensor* transposed_weights =
            data->weights_are_transposed
                ? GetTemporary(context, node, data->transposed_weights_index)
                : nullptr;

        TfLiteTensor* col2im =
            data->has_col2im ? GetTemporary(context, node, data->col2im_index)
                             : nullptr;

        TfLiteTensor* scratch_buffer;
        TF_LITE_ENSURE_OK(
            context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                      &scratch_buffer));

        GetInputSafe(context, inputs_[i][0], &output_shape_tensor);
        GetInputSafe(context, inputs_[i][2], &input);
        GetInputSafe(context, inputs_[i][1], &weights);
        GetInputSafe(context, inputs_[i][3], &bias);
        GetOutputSafe(context, outputs_[i][0], &output);

        if (tflite::IsDynamicTensor(output)) {
          TF_LITE_ENSURE_OK(context, ResizeTensorOutShapeTensor(
                                         context, output_shape_tensor, output));
        }
        if (data->has_col2im && tflite::IsDynamicTensor(col2im)) {
          TF_LITE_ENSURE_OK(
              context, ResizeCol2ImTensor(context, output_shape_tensor, weights,
                                          input, col2im));
        }
        if (tflite::IsDynamicTensor(scratch_buffer)) {
          TF_LITE_ENSURE_OK(
              context, ResizeTensorOutShapeTensor(context, output_shape_tensor,
                                                  scratch_buffer));
        }

        const int width = SizeOfDimension(output, 2);
        const int height = SizeOfDimension(output, 1);
        const int filter_width = SizeOfDimension(weights, 2);
        const int filter_height = SizeOfDimension(weights, 1);
        int unused_output_height, unused_output_width;
        data->padding = ComputePaddingHeightWidth(
            params->stride_height, params->stride_width, 1, 1, height, width,
            filter_height, filter_width, params->padding, &unused_output_height,
            &unused_output_width);

        RuntimeShape input_shape =
            RuntimeShape(input->dims->size, input->dims->data);
        RuntimeShape transposed_weights_shape = RuntimeShape(
            transposed_weights->dims->size, transposed_weights->dims->data);
        RuntimeShape filter_shape =
            RuntimeShape(weights->dims->size, weights->dims->data);
        RuntimeShape output_shape =
            RuntimeShape(output->dims->size, output->dims->data);
        const RuntimeShape& scratch_shape = GetTensorShape(scratch_buffer);
        
        TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
        TFLITE_DCHECK_EQ(transposed_weights_shape.DimensionsCount(), 4);
        TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
        TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

        const int scratch_cols = scratch_shape.Dims(1) * scratch_shape.Dims(2);
        const int scratch_rows = scratch_shape.Dims(0) * scratch_shape.Dims(3);

        const int batches = MatchingDim(input_shape, 0, output_shape, 0);
        const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
        const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
        const int stride_width = params->stride_width;
        const int stride_height = params->stride_height;
        const int pad_width = data->padding.width;
        const int pad_height = data->padding.height;
        const int input_height = input_shape.Dims(1);
        const int input_width = input_shape.Dims(2);

        const int output_height = output_shape.Dims(1);
        const int output_width = output_shape.Dims(2);
        const int32_t input_offset = -input->params.zero_point;
        const int32_t output_offset = output->params.zero_point;
        const int32_t output_activation_min = data->output_activation_min;
        const int32_t output_activation_max = data->output_activation_max;

        const int8_t* input_data = GetTensorData<int8>(input);
        const int8_t* transposed_weight_data =
            GetTensorData<int8>(transposed_weights);
        const int8_t* filter_data = GetTensorData<int8>(weights);
        int8_t* output_data = GetTensorData<int8>(output);
        int32_t* col2im_data = GetTensorData<int32>(col2im);
        int32_t* scratch_data = GetTensorData<int32>(scratch_buffer);
        const int32_t* bias_data =
            (has_bias ? GetTensorData<int32>(bias) : nullptr);
        const int num_elements = output_shape.FlatSize();

        memset(scratch_data, 0, num_elements * sizeof(int32_t));
        
        for (int batch = 0; batch < batches; ++batch) {
          for (int in_y = 0; in_y < input_height; ++in_y) {
            for (int in_x = 0; in_x < input_width; ++in_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int out_x_origin = (in_x * stride_width) - pad_width;
                const int out_y_origin = (in_y * stride_height) - pad_height;
                for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    for (int out_channel = 0; out_channel < output_depth;
                         ++out_channel) {
                      const int out_x = out_x_origin + filter_x;
                      const int out_y = out_y_origin + filter_y;
                      if ((out_x >= 0) && (out_x < output_width) &&
                          (out_y >= 0) && (out_y < output_height)) {
                        const int8_t input_value = input_data[Offset(
                            input_shape, batch, in_y, in_x, in_channel)];
                        const int8_t filter_value =
                            filter_data[Offset(filter_shape, out_channel,
                                               filter_y, filter_x, in_channel)];
                        scratch_data[Offset(output_shape, batch, out_y, out_x,
                                            out_channel)] +=
                            (input_value + input_offset) * filter_value;
                      }
                    }
                  }
                }
              }
            }
          }
        }

        for (int batch = 0; batch < batches; ++batch) {
          for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                int32_t acc = scratch_data[Offset(output_shape, batch, out_y,
                                                  out_x, out_channel)];
                if (bias_data != nullptr) acc += bias_data[out_channel];
                int out_shift =
                    data->per_channel_output_shift.data()[out_channel];
                int out_mult =
                    data->per_channel_output_multiplier.data()[out_channel];
                acc = Quantised_Multiplier_V1(
                    acc, out_mult, out_shift, output_offset,
                    output_activation_min, output_activation_max);
                output_data[Offset(output_shape, batch, out_y, out_x,
                                   out_channel)] = static_cast<int8_t>(acc);
              }
            }
          }
        }
      } else if (builtin_code_[i] == kTfLiteBuiltinShape) {  // SHAPE
        // Nothing to do during eval
      } else if (builtin_code_[i] == kTfLiteBuiltinSoftmax) {  // SOFTMAX
        SOFTMAX_Data* data = reinterpret_cast<SOFTMAX_Data*>(opdatas[i]);
        tflite::SoftmaxParams params = data->params;
        const TfLiteTensor* input;
        TfLiteTensor* output;
        GetInputSafe(context, inputs_[i][0], &input);
        GetOutputSafe(context, outputs_[i][0], &output);

        const int8* input_data;
        int8* output_data;
        if (use_dma_buffer) {
          DMATensor* t1 = dtm.get_tensor(inputs_[i][0]);
          DMATensor* t4 = dtm.get_tensor(outputs_[i][0]);
          input_data = reinterpret_cast<const int8*>(t1->data);
          output_data = reinterpret_cast<int8*>(t4->data);
        } else {
          input_data = input->data.int8;
          output_data = output->data.int8;
        }

        RuntimeShape input_shape =
            RuntimeShape(input->dims->size, input->dims->data);
        RuntimeShape output_shape =
            RuntimeShape(output->dims->size, output->dims->data);

        const int trailing_dim = input_shape.DimensionsCount() - 1;
        const int excluding_last_dim = tflite::MatchingFlatSizeSkipDim(
            input_shape, trailing_dim, output_shape);
        const int last_dim = tflite::MatchingDim(input_shape, trailing_dim,
                                                 output_shape, trailing_dim);

        const int32_t clamp_max = std::numeric_limits<int8_t>::max();
        const int32_t clamp_min = std::numeric_limits<int8_t>::min();
        for (int i = 0; i < excluding_last_dim; ++i) {
          int32_t max_val = std::numeric_limits<int8_t>::min();
          for (int j = 0; j < last_dim; ++j) {
            max_val = std::max(max_val, static_cast<int32_t>(input_data[j]));
          }

          float sum_exp = 0.0f;
          const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
          const float* table_offset = &params.table[max_uint8 - max_val];
          for (int j = 0; j < last_dim; ++j) {
            sum_exp += table_offset[input_data[j]];
          }

          const float inv_sum_exp = 1.0f / (sum_exp * params.scale);
          for (int j = 0; j < last_dim; ++j) {
            const float prob_rescaled =
                table_offset[input_data[j]] * inv_sum_exp;
            const int32_t prob_quantized =
                optimized_ops::QuantizeSoftmaxOutput<int8_t>(prob_rescaled,
                                                             params.zero_point);
            output_data[j] = static_cast<int8_t>(
                std::max(std::min(clamp_max, prob_quantized), clamp_min));
          }
          input_data += last_dim;
          output_data += last_dim;
        }
        
        // drv.p_t = p_t;
        // omni_driver_space::Entry(drv);
        // p_t = drv.p_t;
        
      } else if (builtin_code_[i] == kTfLiteBuiltinPad) {  // PAD
        PadContext op_context(context, i, inputs_, outputs_);
        if (op_context.constant_values != nullptr) {
          TF_LITE_ENSURE_EQ(context,
                            tflite::NumElements(op_context.constant_values), 1);
        }
        if (tflite::IsDynamicTensor(op_context.output)) {
          TfLiteIntArray* input_size = op_context.input->dims;
          TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);
          const int32_t* paddings_data =
              tflite::GetTensorData<int32_t>(op_context.paddings);
          for (int idx = 0; idx < op_context.dims; ++idx) {
            int before_padding = static_cast<int>(*paddings_data++);
            int after_padding = static_cast<int>(*paddings_data++);
            TF_LITE_ENSURE_MSG(context,
                               (before_padding >= 0 && after_padding >= 0),
                               "Pad value has to be greater than equal to 0.");
          }
          paddings_data = tflite::GetTensorData<int32_t>(op_context.paddings);
          for (int idx = 0; idx < op_context.dims; ++idx) {
            int before_padding = static_cast<int>(*paddings_data++);
            int after_padding = static_cast<int>(*paddings_data++);
            output_size->data[idx] =
                (input_size->data[idx] + before_padding + after_padding);
          }

          TF_LITE_ENSURE_STATUS(
              context->ResizeTensor(context, op_context.output, output_size));
        }

        tflite::PadParams op_params;
        const int32_t* paddings_data =
            tflite::GetTensorData<int32_t>(op_context.paddings);
        op_params.left_padding_count = op_context.dims;
        op_params.right_padding_count = op_context.dims;
        for (int idx = op_context.dims - 1; idx >= 0; --idx) {
          op_params.left_padding[idx] =
              static_cast<int32_t>(paddings_data[idx * 2]);
          op_params.right_padding[idx] =
              static_cast<int32_t>(paddings_data[idx * 2 + 1]);
        }

        int8_t pad_value;
        if (op_context.constant_values == nullptr) {
          TF_LITE_ENSURE(context, op_context.output->params.zero_point >=
                                      std::numeric_limits<int8_t>::min());
          TF_LITE_ENSURE(context, op_context.output->params.zero_point <=
                                      std::numeric_limits<int8_t>::max());
          pad_value = static_cast<int8_t>(op_context.output->params.zero_point);
        } else {
          TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point,
                            op_context.constant_values->params.zero_point);
          TF_LITE_ENSURE_EQ(context, op_context.output->params.scale,
                            op_context.constant_values->params.scale);
          pad_value =
              *tflite::GetTensorData<int8_t>(op_context.constant_values);
        }
        const int8_t pad_value_copy = pad_value;

        RuntimeShape input_shape = RuntimeShape(op_context.input->dims->size,
                                                op_context.input->dims->data);
        RuntimeShape output_shape = RuntimeShape(op_context.output->dims->size,
                                                 op_context.output->dims->data);

        const int8_t* input_data = op_context.input->data.int8;
        int8_t* output_data = op_context.output->data.int8;

        const RuntimeShape ext_input_shape = RuntimeShape::ExtendedShape(
            PadKernelMaxDimensionCount, input_shape);
        const RuntimeShape ext_output_shape = RuntimeShape::ExtendedShape(
            PadKernelMaxDimensionCount, output_shape);

        TFLITE_DCHECK_LE(op_params.left_padding_count,
                         PadKernelMaxDimensionCount);
        TFLITE_DCHECK_LE(op_params.right_padding_count,
                         PadKernelMaxDimensionCount);

        int left_padding_copy[PadKernelMaxDimensionCount];
        for (int i = 0; i < PadKernelMaxDimensionCount; i++) {
          left_padding_copy[i] = 0;
        }
        for (int i = 0; i < op_params.left_padding_count; ++i) {
          left_padding_copy[i + PadKernelMaxDimensionCount -
                            op_params.left_padding_count] =
              op_params.left_padding[i];
        }
        int right_padding_copy[PadKernelMaxDimensionCount];
        for (int i = 0; i < PadKernelMaxDimensionCount; i++) {
          right_padding_copy[i] = 0;
        }
        for (int i = 0; i < op_params.right_padding_count; ++i) {
          right_padding_copy[i + PadKernelMaxDimensionCount -
                             op_params.right_padding_count] =
              op_params.right_padding[i];
        }

        const int output_batch = ext_output_shape.Dims(0);
        const int output_plane = ext_output_shape.Dims(1);
        const int output_height = ext_output_shape.Dims(2);
        const int output_width = ext_output_shape.Dims(3);
        const int output_depth = ext_output_shape.Dims(4);

        const int left_b_padding = left_padding_copy[0];
        const int left_p_padding = left_padding_copy[1];
        const int left_h_padding = left_padding_copy[2];
        const int left_w_padding = left_padding_copy[3];
        const int left_d_padding = left_padding_copy[4];

        const int right_b_padding = right_padding_copy[0];
        const int right_p_padding = right_padding_copy[1];
        const int right_h_padding = right_padding_copy[2];
        const int right_w_padding = right_padding_copy[3];
        const int right_d_padding = right_padding_copy[4];

        const int8_t* in_ptr = input_data;
        int8_t* out_ptr = output_data;

        for (int out_b = 0; out_b < output_batch; ++out_b) {
          for (int out_p = 0; out_p < output_plane; ++out_p) {
            for (int out_h = 0; out_h < output_height; ++out_h) {
              for (int out_w = 0; out_w < output_width; ++out_w) {
                for (int out_d = 0; out_d < output_depth; ++out_d) {
                  if (out_b < left_b_padding ||
                      out_b >= output_batch - right_b_padding ||
                      out_p < left_p_padding ||
                      out_p >= output_plane - right_p_padding ||
                      out_h < left_h_padding ||
                      out_h >= output_height - right_h_padding ||
                      out_w < left_w_padding ||
                      out_w >= output_width - right_w_padding ||
                      out_d < left_d_padding ||
                      out_d >= output_depth - right_d_padding) {
                    *out_ptr++ = pad_value;
                  } else {
                    *out_ptr++ = *in_ptr++;
                  }
                }
              }
            }
          }
        }

      } else if (builtin_code_[i] == kTfLiteBuiltinMean) {  // MEAN
        REDUCE_Data* data = reinterpret_cast<REDUCE_Data*>(opdatas[i]);
        TfLiteReducerParams* params =
            reinterpret_cast<TfLiteReducerParams*>(layers_params[i]);
        ReduceOpContext op_context(context, params, i, inputs_, outputs_);

        int num_axis = static_cast<int>(NumElements(op_context.axis));
        TfLiteTensor* omni_index;
        TfLiteTensor* resolved_axis;
        TfLiteTensor* omni_sum;
        TfLiteTensor* normalized_dims;

        int scratch_index = get<0>(omni_tensor_ids[i][0]);
        int resolved_axis_index = get<0>(omni_tensor_ids[i][1]);
        int omni_accum_index = get<0>(omni_tensor_ids[i][2]);
        int normalized_dims_index = get<0>(omni_tensor_ids[i][3]);

        GetTemporarySafe(context, node, scratch_index, &omni_index);
        GetTemporarySafe(context, node, resolved_axis_index, &resolved_axis);
        GetTemporarySafe(context, node, omni_accum_index, &omni_sum);
        GetTemporarySafe(context, node, normalized_dims_index,
                         &normalized_dims);

        if (IsDynamicTensor(op_context.output)) {
          TF_LITE_ENSURE_OK(
              context, ResizeTempAxis(context, &op_context, resolved_axis));
          TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
          TF_LITE_ENSURE_OK(context,
                            ResizeTempAccum(context, &op_context, omni_sum));
        }

        if (IsDynamicTensor(normalized_dims)) {
          TF_LITE_ENSURE_OK(
              context, ResizeTempDims(context, &op_context, normalized_dims));
        }

        const TfLiteTensor* input = op_context.input;
        RuntimeShape input_shape = GetTensorShape(input);
        if (input_shape.FlatSize() == 0) {
          TF_LITE_ENSURE_OK(context,
                            InitializeMeanOutputTyped(op_context.output));
          continue;
        }

        TfLiteTensor* output = op_context.output;
        tflite::reference_ops::QuantizedMeanOrSum(
            tflite::GetTensorData<int8_t>(input), input->params.zero_point,
            input->dims->data, input->dims->size,
            tflite::GetTensorData<int8_t>(output), data->multiplier,
            data->shift, output->params.zero_point, output->dims->data,
            output->dims->size, tflite::GetTensorData<int>(op_context.axis),
            num_axis, op_context.params->keep_dims,
            tflite::GetTensorData<int>(omni_index),
            tflite::GetTensorData<int>(resolved_axis),
            tflite::GetTensorData<int32_t>(omni_sum),
            /*compute_sum=*/false);

      } else if (builtin_code_[i] == kTfLiteBuiltinQuantize) {  // QUANTIZE
        QUANTIZE_Data* data = reinterpret_cast<QUANTIZE_Data*>(opdatas[i]);
        TfLiteTensor* output;
        const TfLiteTensor* input;

        GetOutputSafe(context, outputs_[i][0], &output);
        GetInputSafe(context, inputs_[i][0], &input);

        const RuntimeShape input_shape = GetTensorShape(input);
        const RuntimeShape output_shape = GetTensorShape(output);
        QuantizeKernelType kernel_type = kGenericOptimized;

        switch (input->type) {
          case kTfLiteFloat32: {
            const float* input_data = GetTensorData<float>(input);

            if (IsQuantizedPerChannel(output)) {
              const auto* quantization_params =
                  reinterpret_cast<const TfLiteAffineQuantization*>(
                      output->quantization.params);
              PerChannelQuantizationParams per_channel_op_params;
              per_channel_op_params.quantized_dimension =
                  quantization_params->quantized_dimension;
              per_channel_op_params.scale = quantization_params->scale->data;
              per_channel_op_params.zero_point =
                  quantization_params->zero_point->data;

              switch (output->type) {
                case kTfLiteInt8:
                  reference_ops::PerChannelQuantize(
                      per_channel_op_params, input_shape, input_data,
                      output_shape, GetTensorData<int8_t>(output));
                  break;
                case kTfLiteUInt8:
                  reference_ops::PerChannelQuantize(
                      per_channel_op_params, input_shape, input_data,
                      output_shape, GetTensorData<uint8_t>(output));
                  break;
                case kTfLiteInt16:
                  reference_ops::PerChannelQuantize(
                      per_channel_op_params, input_shape, input_data,
                      output_shape, GetTensorData<int16_t>(output));
                  break;
                default:
                  cout << "Error: " << __LINE__ << endl;
                  return kTfLiteError;
              }
            } else {
              tflite::QuantizationParams op_params;
              op_params.zero_point = output->params.zero_point;
              op_params.scale = output->params.scale;

              switch (output->type) {
                case kTfLiteInt8:
                  AffineQuantize<kGenericOptimized>(
                      op_params, input_shape, input_data, output_shape,
                      GetTensorData<int8_t>(output));
                  break;
                case kTfLiteUInt8:
                  AffineQuantize<kGenericOptimized>(
                      op_params, input_shape, input_data, output_shape,
                      GetTensorData<uint8_t>(output));
                  break;
                case kTfLiteInt16:
                  AffineQuantize<kGenericOptimized>(
                      op_params, input_shape, input_data, output_shape,
                      GetTensorData<int16_t>(output));
                  break;
                default:
                  cout << "Error: " << __LINE__ << endl;
                  return kTfLiteError;
              }
            }
            break;
          }
          case kTfLiteInt32: {
            switch (output->type) {
              case kTfLiteInt8:
                Requantize<kGenericOptimized>(
                    GetTensorData<int32_t>(input),
                    MatchingFlatSize(input_shape, output_shape),
                    data->output_multiplier, data->output_shift,
                    input->params.zero_point, output->params.zero_point,
                    GetTensorData<int8_t>(output));
                break;
              case kTfLiteInt16:
                Requantize<kGenericOptimized>(
                    GetTensorData<int32_t>(input),
                    MatchingFlatSize(input_shape, output_shape),
                    data->output_multiplier, data->output_shift,
                    input->params.zero_point, output->params.zero_point,
                    GetTensorData<int16_t>(output));
                break;
              default:
                cout << "Error: " << __LINE__ << endl;
                return kTfLiteError;
            }
            break;
          }
          case kTfLiteInt16: {
            switch (output->type) {
              case kTfLiteInt8:
                Requantize<kGenericOptimized>(
                    GetTensorData<int16_t>(input),
                    MatchingFlatSize(input_shape, output_shape),
                    data->output_multiplier, data->output_shift,
                    input->params.zero_point, output->params.zero_point,
                    GetTensorData<int8_t>(output));
                break;
              case kTfLiteInt16:
                Requantize<kGenericOptimized>(
                    GetTensorData<int16_t>(input),
                    MatchingFlatSize(input_shape, output_shape),
                    data->output_multiplier, data->output_shift,
                    input->params.zero_point, output->params.zero_point,
                    GetTensorData<int16_t>(output));
                break;
              case kTfLiteInt32:
                Requantize<kGenericOptimized>(
                    GetTensorData<int16_t>(input),
                    MatchingFlatSize(input_shape, output_shape),
                    data->output_multiplier, data->output_shift,
                    input->params.zero_point, output->params.zero_point,
                    GetTensorData<int32_t>(output));
                break;
              default:
                cout << "Error: " << __LINE__ << endl;
                return kTfLiteError;
            }
            break;
          }
          case kTfLiteInt8: {
            const int32_t size = MatchingFlatSize(input_shape, output_shape);
            const int8_t* input_data = GetTensorData<int8_t>(input);
            switch (output->type) {
              case kTfLiteInt8:
                Requantize<kGenericOptimized>(
                    input_data, size, data->output_multiplier,
                    data->output_shift, input->params.zero_point,
                    output->params.zero_point, GetTensorData<int8_t>(output));
                break;
              case kTfLiteUInt8:
                Requantize<kGenericOptimized>(
                    input_data, size, data->output_multiplier,
                    data->output_shift, input->params.zero_point,
                    output->params.zero_point, GetTensorData<uint8_t>(output));
                break;
              default:
                cout << "Error: " << __LINE__ << endl;
                return kTfLiteError;
            }
            break;
          }
          case kTfLiteUInt8: {
            const int32_t size = MatchingFlatSize(input_shape, output_shape);
            const uint8_t* input_data = GetTensorData<uint8_t>(input);
            switch (output->type) {
              case kTfLiteInt8:
                Requantize<kGenericOptimized>(
                    input_data, size, data->output_multiplier,
                    data->output_shift, input->params.zero_point,
                    output->params.zero_point, GetTensorData<int8_t>(output));
                break;
              case kTfLiteUInt8:
                Requantize<kGenericOptimized>(
                    input_data, size, data->output_multiplier,
                    data->output_shift, input->params.zero_point,
                    output->params.zero_point, GetTensorData<uint8_t>(output));
                break;
              default:
                cout << "Error: " << __LINE__ << endl;
                return kTfLiteError;
            }
            break;
          }
          default:
            cout << "Error: " << __LINE__ << endl;
            return kTfLiteError;
        }
      } else if (builtin_code_[i] == kTfLiteBuiltinDequantize) {  // DEQUANTIZE
        DEQUANTIZE_Data* data = reinterpret_cast<DEQUANTIZE_Data*>(opdatas[i]);
        DequantizeOpContext op_context(context, i, inputs_, outputs_);
        const TfLiteTensor* input = op_context.input;
        TfLiteTensor* output = op_context.output;

        if (tflite::IsConstantTensor(op_context.input) &&
            data->float_dequantized_weights_initialized) {
          continue;
        }

        if (IsQuantizedPerChannel(input)) {
          const auto* quantization_params =
              reinterpret_cast<const TfLiteAffineQuantization*>(
                  input->quantization.params);
          PerChannelDequantizationParams per_channel_op_params;
          per_channel_op_params.quantized_dimension =
              quantization_params->quantized_dimension;
          per_channel_op_params.scale = quantization_params->scale->data;
          per_channel_op_params.zero_point =
              quantization_params->zero_point->data;

          tflite::reference_ops::PerChannelDequantize<int8_t>(
              per_channel_op_params, GetTensorShape(input),
              GetTensorData<int8_t>(input), GetTensorShape(output),
              GetTensorData<float>(output));
        }

        DequantizationParams op_params;
        op_params.zero_point = input->params.zero_point;
        op_params.scale = input->params.scale;

        tflite::optimized_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));

        if (IsConstantTensor(op_context.input)) {
          data->float_dequantized_weights_initialized = true;
        }
      }
      // =======================================================================
      // End of All Operator Evals
      // =======================================================================

      // =======================================================
      // Store-Back Stage & Logging
      // =======================================================
      if (use_dma_buffer) {
        for (int j = 0; j < outputs_[i].size(); j++) {
          unsigned int TID = outputs_[i][j];
          TfLiteTensor* tensor;
          GetOutputSafe(context, TID, &tensor);
          if (TID == -1) continue;
          DMATensor* t = dtm.get_tensor(TID);
          if (!t->mmapped && t->isExternalOutput) {
            cout << "  [Output] TID " << TID << " : Stored (DMA Buffer -> Host CPU)" << endl;
            dtm.copy_from_mmap(TID, tensor->data.raw);
          } else {
            cout << "  [Output] TID " << TID << " : Kept   (Resident in DMA Buffer)" << endl;
          }
        }
      }
      cout << "======================================================" << endl;

      // =======================================================================
      // Delegate Management Code
      // =======================================================================
      // Pops output dependencies
      for (int n = 0; n < i; n++) {
        for (int dep_node : output_dependencies[n]) {
          if (dep_node == i) {
#ifdef DELEGATE_VERBOSE
            cout << "Popping node: " << associated_nodes[i]
                 << " from layer dependency: " << associated_nodes[n] << endl;
#endif
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
    prf_end(0, p_t.delegate_total);  // Stop the profiling delegate
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<void*> opdatas;
  std::vector<void*> layers_params;

  std::vector<std::vector<int>> output_dependencies;
  std::vector<bool> node_output_needed;
  std::vector<bool> is_global_output;
  std::vector<std::vector<std::tuple<int, int>>> omni_tensor_ids;

  // Convolution specific variables
  std::vector<std::vector<int>> wt_sum;
  std::vector<std::vector<int8_t>> omni_im2col;

 private:
  const OmniDelegateOptions options_;
};

class OmniDelegate : public SimpleDelegateInterface {
 public:
  explicit OmniDelegate(const OmniDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    bool isCONV2D = IsNode_CONV2D_INT8(registration, node, context);
    bool isDWCONV2D = IsNode_DWCONV2D_INT8(registration, node, context);
    bool isADD = IsNode_ADD_INT8(registration, node, context);
    bool isFC = IsNode_FC_INT8(registration, node, context);
    bool isTCONV = IsNode_TCONV_INT8(registration, node, context);
    bool isSHAPE = IsNode_SHAPE_INT8(registration, node, context);
    bool isSOFTMAX = IsNode_SOFTMAX_INT8(registration, node, context);
    bool isPAD = IsNode_PAD_INT8(registration, node, context);
    bool isMEAN = IsNode_MEAN_INT8(registration, node, context);
    bool isQUANTIZE = IsNode_QUANTIZE_INT8(registration, node, context);
    bool isDEQUANTIZE = IsNode_DEQUANTIZE_INT8(registration, node, context);

    std::vector<bool> supported_nodes = {isADD,     isCONV2D,   isFC,
                                         isSOFTMAX, isDWCONV2D, isSHAPE};

    bool delegated_node = false;
    for (int i = 0; i < supported_nodes.size(); i++)
      if (supported_nodes[i]) delegated_node = true;

    int output_tid = node->outputs->data[0];
    int forbidden_output_tid[] = {};
    for (int tid : forbidden_output_tid)
      if (output_tid == tid) delegated_node = false;

    if (delegated_node) dparams.delegated_nodes++;
    return delegated_node;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "OmniDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<OmniDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    return SimpleDelegateInterface::Options();
  }

 private:
  const OmniDelegateOptions options_;
};

}  // namespace omni_test
}  // namespace tflite

OmniDelegateOptions TfLiteOmniDelegateOptionsDefault() {
  OmniDelegateOptions options = {0};
  options.allowed_builtin_code = -1;
  return options;
}

TfLiteDelegate* TfLiteOmniDelegateCreate(const OmniDelegateOptions* options) {
  std::unique_ptr<tflite::omni_test::OmniDelegate> omni(
      new tflite::omni_test::OmniDelegate(
          options ? *options : TfLiteOmniDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(omni), kTfLiteDelegateFlagsAllowDynamicTensors);
}

void TfLiteOmniDelegateDelete(TfLiteDelegate* delegate) {
  SYSC_ON(profile.saveProfile(acc->profiling_vars));
  time_t now = time(0);
  tm* ltm = localtime(&now);
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
  dtm.print_tensors();
  std::cout << "===========================" << std::endl;
  std::cout << "Deleted" << std::endl;
  std::cout << "===========================" << std::endl;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}