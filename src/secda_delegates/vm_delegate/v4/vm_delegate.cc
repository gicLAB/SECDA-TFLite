// #define SYSC

#include "vm_delegate.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#include <utility>

#ifdef SYSC
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_integrator/systemc_integrate.h"
#endif

#include "accelerator/driver/gemm_driver.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "util.h"

#define DMA_BC 1 // threaded execution does not work with multiple blocks
#define DELEGATE_VERSION 3

static unsigned int dma_addrs[4] = {dma_addr0, dma_addr1, dma_addr2, dma_addr3};
static unsigned int dma_addrs_in[4] = {dma_in0, dma_in1, dma_in2, dma_in3};
static unsigned int dma_addrs_out[4] = {dma_out0, dma_out1, dma_out2, dma_out3};
static struct vm_times vm_t;

#ifdef SYSC
#define SYSC_DMA_BL 563840 * 2
static struct multi_dma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out,
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
static struct multi_dma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out, DMA_BL);
int *acc;
struct dma_buffer_set dfs[4] = {
    {DMA_BC, (DMA_BL / DMA_BC), dma_in0},
    {DMA_BC, (DMA_BL / DMA_BC), dma_in1},
    {DMA_BC, (DMA_BL / DMA_BC), dma_in2},
    {DMA_BC, (DMA_BL / DMA_BC), dma_in3},
};
int recv_len = (DMA_BL / DMA_BC);
#endif

struct store_params st_params[DMA_BC];
struct del_params dparams;
static struct Profile profile;

namespace tflite {
namespace vm_test {

// VM delegate kernel
class VMDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit VMDelegateKernel(const VMDelegateOptions &options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {
    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      std::cout << "===========================" << std::endl;
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("VM");
      sysC_init();
      sysC_binder(&_acc, &mdma, &scs1);
      acc = &_acc;
      std::cout << "Initialised the SystemC Modules" << std::endl;
#else
      dparams.acc = getAccBaseAddress<int>(acc_address, 65536);
      acc = dparams.acc;
      std::cout << "Initialised the DMA" << std::endl;
#endif

      std::cout << "Vector MAC (v4)";
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
    // temp_output.resize(params->nodes_to_replace->size);
    // temp_out_id.resize(params->nodes_to_replace->size);
    // for (int i = 0; i < temp_out_id.size(); i++) temp_out_id[i] = -1;

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
      TfLiteConvParams *cparam =
          reinterpret_cast<TfLiteConvParams *>(delegated_node->builtin_data);
      OpData *opdata = reinterpret_cast<OpData *>(delegated_node->user_data);
      cparams[i] = cparam;
      opdatas[i] = opdata;
    }
    return kTfLiteOk;
  }

  // Runs once per node before inference/invoke()
  // This function preloads weights, allocates additional tensors, calculates
  // quantization parameters For more info look into
  // "tensorflow/lite/kernels/conv.cc" for the default implementation for Conv2D
  // Nodes
  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) override {
    int node_count = inputs_.size();
    int out_tid = 0;

    for (int i = 0; i < node_count; i++) {
      TfLiteConvParams *params = cparams[i];
      OpData *data = opdatas[i];

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
      data->padding = ComputePaddingHeightWidth(
          params->stride_height, params->stride_width,
          params->dilation_height_factor, params->dilation_width_factor, height,
          width, filter_height, filter_width, padding, &out_height, &out_width);

      size_t im2col_type_size = sizeof(int8_t);

      const size_t im2col_bytes = static_cast<size_t>(batches) * out_height *
                                  out_width * channels_in * filter_height *
                                  filter_width * im2col_type_size;

      int temp_o_id;
      int nod = node->outputs->data[out_tid];
      int oi = outputs_[i][0];
      bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      if (!req_temp_out) out_tid++;

      TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
          context, node, is_hybrid, data->is_hybrid_per_channel, im2col_bytes,
          params, data, req_temp_out, outputs_[i][0], temp_o_id, inputs_[i][0],
          inputs_[i][1]));

      TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                        kTfLiteAffineQuantization);
      const auto *affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization *>(
              filter->quantization.params);
      TF_LITE_ENSURE(context, affine_quantization);
      TF_LITE_ENSURE(context, affine_quantization->scale);
      TF_LITE_ENSURE(context,
                     (affine_quantization->scale->size == 1 ||
                      affine_quantization->scale->size == channels_out));
      data->per_channel_output_shift.resize(channels_out);
      crf[i].resize(channels_out);
      crx[i].resize(channels_out);

      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
          context, input, filter, bias, output, params->activation,
          &data->output_multiplier, &data->output_shift,
          &data->output_activation_min, &data->output_activation_max,
          &crf[i][0], data->per_channel_output_shift.data(), channels_out));

      for (int j = 0; j < channels_out; j++)
        crx[i][j] = (int8_t)data->per_channel_output_shift.data()[j];

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
        auto im2col_status =
            context->ResizeTensor(context, im2col, im2col_size);
        if (im2col_status != kTfLiteOk) return im2col_status;
        temp_im2col[i].resize(im2col_bytes);
      }

      if (data->need_hwcn_weights) {
        node->temporaries->data[data->hwcn_weights_index] =
            data->hwcn_weights_id;
        TfLiteIntArray *hwcn_weights_size = TfLiteIntArrayCreate(2);

        int input_depth = input->dims->data[3];
        hwcn_weights_size->data[0] =
            (filter_height * filter_width * input_depth);
        hwcn_weights_size->data[1] = channels_out;

        TfLiteTensor *hwcn_weights =
            &context
                 ->tensors[node->temporaries->data[data->hwcn_weights_index]];
        hwcn_weights->type = input->type;
        hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;
        auto hwcn_weights_status =
            context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
        if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

        data->have_weights_been_transposed = false;
      }

      if (req_temp_out) {
        // temp_output[i].resize(output->bytes);
        // temp_out_id[i] = outputs_[i][0];
        node->temporaries->data[temp_o_id] = outputs_[i][0];
        TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteArenaRw;
        TfLiteIntArray *temp_out_tensor_size = TfLiteIntArrayCreate(4);
        temp_out_tensor_size->data[0] = output_size->data[0];
        temp_out_tensor_size->data[1] = output_size->data[1];
        temp_out_tensor_size->data[2] = output_size->data[2];
        temp_out_tensor_size->data[3] = output_size->data[3];
        auto temp_out_tensor_status = context->ResizeTensor(
            context, temp_out_tensor, temp_out_tensor_size);
        if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
      }

      biases[i] = bias->data.i32;
      int *dims = filter->dims->data;
      preload_weights(filter->data.int8, dims, wb0[i], wb1[i], wb2[i], wb3[i],
                      wt_sum1[i], wt_sum2[i], wt_sum3[i], wt_sum4[i]);
    }
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  // This function executes the operations required by node by offloading the
  // computation to the gemm_driver For more info look into
  // "tensorflow/lite/kernels/conv.cc" for the default implementation for Conv2D
  // Nodes
  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    prf_start(0);
    int out_tid = 0;
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      prf_start(1);
      auto *params = cparams[i];
      OpData *data = opdatas[i];
      const TfLiteTensor *input;
      const TfLiteTensor *filter;
      TfLiteTensor *output;

      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      GetOutputSafe(context, outputs_[i][0], &output);

      // bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      // if (!req_temp_out) out_tid++;
      // bool req_temp_in = false;
      // int temp_in_id = 0;
      // for (int j=0; j< temp_out_id.size(); j++){
      //   if (inputs_[i][0] == temp_out_id[j]){
      //     req_temp_in = true;
      //     temp_in_id = j;
      //     break;
      //   }
      // }

      // // const int8 *input_data = input->data.int8;
      // const int8 *input_data;
      // if (req_temp_in) {
      //   input_data = &temp_output[temp_in_id][0];
      // } else {
      //   input_data = input->data.int8;
      // }
      // // int8 *output_data = output->data.int8;
      // int8 *output_data;
      // if (req_temp_out) {
      //   output_data = &temp_output[i][0];
      //   // output_data = output->data.int8;
      // }else{
      //   output_data = output->data.int8;
      // }

      TfLiteTensor *im2col =
          data->need_im2col
              ? &context->tensors[node->temporaries->data[data->im2col_index]]
              : nullptr;

      const int8 *input_data = input->data.int8;
      const int8 *filter_data = filter->data.int8;
      // int8 *im2col_data = data->need_im2col ? im2col->data.int8 : nullptr;
      int8 *im2col_data = data->need_im2col ? &temp_im2col[i][0] : nullptr;
      int8 *output_data = output->data.int8;

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

      RuntimeShape input_shape =
          RuntimeShape(input->dims->size, input->dims->data);
      RuntimeShape filter_shape =
          RuntimeShape(filter->dims->size, filter->dims->data);
      RuntimeShape output_shape =
          RuntimeShape(output->dims->size, output->dims->data);

      const int stride_width = params->stride_width;
      const int stride_height = params->stride_height;

      const int dilation_width_factor = params->dilation_width_factor;
      const int dilation_height_factor = params->dilation_height_factor;
      const int32 input_offset = -input->params.zero_point;
      const int32 output_offset = output->params.zero_point;
      // Set min and max value of the output.
      const int32 output_activation_min = data->output_activation_min;
      const int32 output_activation_max = data->output_activation_max;

      const int8 *gemm_input_data = nullptr;
      const RuntimeShape *gemm_input_shape = nullptr;
      const int filter_width = filter_shape.Dims(2);
      const int filter_height = filter_shape.Dims(1);
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
        Im2col<int8_t>(op_params, filter_height, filter_width, zero_point_byte,
                       input_shape, input_data, im2col_shape, im2col_data);
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

      //  Load & Reshape Input data to temporary buffers before offloading to
      //  DMA inbuffers
      int width = gemm_input_cols;
      int w = ((width + 3) - ((width + 3) % 4));
      int depth = filter_cols;
      int d = ((depth + 15) - ((depth + 15) % 16));
      int s_need = w * d / 4 + 1;
      int in_sum1[w / 4];
      int in_sum2[w / 4];
      int in_sum3[w / 4];
      int in_sum4[w / 4];
      int8_t inb0[s_need];
      int8_t inb1[s_need];
      int8_t inb2[s_need];
      int8_t inb3[s_need];
      precal_sum_load_pad(gemm_input_data, width, depth, inb0, inb1, inb2, inb3,
                          in_sum1, in_sum2, in_sum3, in_sum4);

      int *wb_0 = reinterpret_cast<int *>(&wb0[i][0]);
      int *wb_1 = reinterpret_cast<int *>(&wb1[i][0]);
      int *wb_2 = reinterpret_cast<int *>(&wb2[i][0]);
      int *wb_3 = reinterpret_cast<int *>(&wb3[i][0]);

      // acc_container is used to wrap all the paramters the
      // gemm_driver/accelerator needs from the delegate
      struct acc_container drv(wb_0, wb_1, wb_2, wb_3, wt_sum1[i], wt_sum2[i],
                               wt_sum3[i], wt_sum4[i], crf[i], crx[i]);
      drv.acc = acc;
      drv.mdma = &mdma;
      drv.profile = &profile;
      drv.st_params = st_params;
      drv.dfs = dfs;
      drv.mt_context = &dparams.mt_context;
      drv.thread_count = context->recommended_num_threads;
      drv.in_id = 0;
      int *inb_0 = reinterpret_cast<int *>(inb0);
      int *inb_1 = reinterpret_cast<int *>(inb1);
      int *inb_2 = reinterpret_cast<int *>(inb2);
      int *inb_3 = reinterpret_cast<int *>(inb3);
      drv.inb_0 = inb_0;
      drv.inb_1 = inb_1;
      drv.inb_2 = inb_2;
      drv.inb_3 = inb_3;
      drv.in_sum1 = in_sum1;
      drv.in_sum2 = in_sum2;
      drv.in_sum3 = in_sum3;
      drv.in_sum4 = in_sum4;
      drv.bias = biases[i];
      drv.ra = output_offset;
      drv.inp_offset = input_offset;
      drv.wgt_offset = 0;
      drv.t.layer = associated_nodes[i];
      drv.recv_len = recv_len;
      drv.rows = gemm_input_cols;
      drv.cols = filter_rows;
      drv.depth = filter_cols;

      // drv.use_sim = options_.use_simmode;
      drv.use_sim = false;
      // drv.clear_traces();

      prf_end(1, vm_t.p_ipack);
      // Calls the gemm_driver to offload the CONV2D operation
      drv.t2 = vm_t;
      tflite_vm::Entry(drv, output_data);
      vm_t = drv.t2;
#ifdef DELEGATE_VERBOSE
      cout << "===========================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i] << endl;
      cout << "===========================" << endl;
#endif
      // saveMatrixCSV("aData/conv/" + std::to_string(associated_nodes[i]) +
      //                   "_wgt_acc.csv",
      //               filter_data, filter_rows, filter_cols);
      // saveMatrixCSV("aData/conv/" + std::to_string(associated_nodes[i]) +
      //                   "_inp_acc.csv",
      //               gemm_input_data, gemm_input_cols, gemm_input_rows);
      // saveMatrixCSV("aData/conv/" + std::to_string(associated_nodes[i]) +
      //                   "_out_acc.csv",
      //               output_data, gemm_input_cols, filter_rows);
      dparams.layer++;
      dparams.delegated_nodes--;
    }

    // for (int j = 0; j < temp_output_tensors.size(); j++)
    //   temp_output_tensors.at(j).free();
    prf_end(0, vm_t.t_conv_total);
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  // std::vector<temp_tensor> temp_output_tensors;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<std::vector<int>> wt_sum1;
  std::vector<std::vector<int>> wt_sum2;
  std::vector<std::vector<int>> wt_sum3;
  std::vector<std::vector<int>> wt_sum4;
  std::vector<std::vector<int8_t>> wb0;
  std::vector<std::vector<int8_t>> wb1;
  std::vector<std::vector<int8_t>> wb2;
  std::vector<std::vector<int8_t>> wb3;
  std::vector<int *> biases;
  std::vector<std::vector<int8_t>> temp_im2col;
  // std::vector<std::vector<int8_t>> temp_output;
  // std::vector<int> temp_out_id;
  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int8_t>> crx;
  std::vector<OpData *> opdatas;
  std::vector<TfLiteConvParams *> cparams;

private:
  const VMDelegateOptions options_;
};

// VMDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class VMDelegate : public SimpleDelegateInterface {
public:
  explicit VMDelegate(const VMDelegateOptions &options) : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {
    // Only supports CONV2D op
    if (kTfLiteBuiltinConv2d != registration->builtin_code) return false;

    // This delegate only supports int8 types.
    if (node->inputs->size != 3) return false;
    for (int i = 0; i < 2; ++i) {
      auto &tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
      // if (tensor.dims->data[0] == 1000) return false;
    }

    // Ensures bias tensor is supports int32 type
    auto &tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32) return false;

    // if (dparams.delegated_nodes > 3) return false;
    // Adds node for delegation
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "VMDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<VMDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const VMDelegateOptions options_;
};

} // namespace vm_test
} // namespace tflite

VMDelegateOptions TfLiteVMDelegateOptionsDefault() {
  VMDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this vm test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteVMDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteVMDelegateCreate(const VMDelegateOptions *options) {
  std::cout << "===========================" << std::endl;
  std::cout << "Created" << std::endl;
  std::cout << "===========================" << std::endl;
  // std::unique_ptr<tflite::vm_test::VMDelegate> vm(
  //     new tflite::vm_test::VMDelegate(
  //         options ? *options : TfLiteVMDelegateOptionsDefault()));
  // return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(vm));
  std::unique_ptr<tflite::vm_test::VMDelegate> vm(
      new tflite::vm_test::VMDelegate(
          options ? *options : TfLiteVMDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(vm), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `TfLiteVMDelegateCreate` call.
void TfLiteVMDelegateDelete(TfLiteDelegate *delegate) {
  // Saves profilier records once all delegated nodes are executed
  // SYSC_ON(profile.saveProfile(acc->profiling_vars));
  time_t now = time(0);
  tm *ltm = localtime(&now);
  std::string date =
      std::to_string(1900 + ltm->tm_year) + "-" +
      std::to_string(1 + ltm->tm_mon) + "-" + std::to_string(ltm->tm_mday) +
      "-" + std::to_string(ltm->tm_hour) + "-" + std::to_string(ltm->tm_min) +
      "-" + std::to_string(ltm->tm_sec);
  SYSC_ON(profile.saveCSVRecords(".data/vm_profs/vm_" + date));
#ifndef SYSC
  if (!dparams.unmap) {
    mdma.multi_free_dmas();
    munmap(dparams.acc, 65536);
    std::cout << "===========================" << std::endl;
    std::cout << "Unmapped DMA I/O Buffers" << std::endl;
    std::cout << "===========================" << std::endl;
    dparams.unmap = true;
    for (int i = 0; i < 4; i++) dfs[i].free();
  }
#endif
  vm_t.print();
  vm_t.save_prf();
  std::cout << "===========================" << std::endl;
  std::cout << "Deleted" << std::endl;
  std::cout << "===========================" << std::endl;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
