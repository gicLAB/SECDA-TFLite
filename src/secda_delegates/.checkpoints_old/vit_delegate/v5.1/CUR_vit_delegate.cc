
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

    //! ?
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

    // opdatas.resize(params->nodes_to_replace->size);
    // cparams.resize(params->nodes_to_replace->size);
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
  // This funciton executes the operations required by node by offloading the
  // computation to the fully_connected
  // For more info find default implementation at
  // "tensorflow/lite/kernels/fully_connected.cc"
  // OFFLOADS TO ACCELERATOR DRIVER
  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    int node_count = inputs_.size();
    struct acc_container drv;
    drv.acc = acc;
    drv.profile = &profile;
    drv.mdma = &mdma;
    drv.mt_context = &dparams.mt_context;
    drv.thread_count = context->recommended_num_threads;

    for (int i = 0; i < node_count; i++) {
      // =======================================================
      // Delegate Mangement Code
      drv.op_type = builtin_code_[i];
      // #ifdef DELEGATE_VERBOSE
      cout << "======================================================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i]
           << "      Type: " << EnumNamesBuiltinOperator()[builtin_code_[i]]
           << endl;
      cout << "======================================================" << endl;

      if (builtin_code_[i] == kTfLiteBuiltinConv2d) { // CONV2D
        prf_start(1);
        // // rpp -code
        // t.layer = dparams.layer;
        // t.conv_layer_no = dparams.layer;
        // t.node = associated_nodes[i];
        // rpp -code -end

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
        // int32_t input_offset = op_params.input_offset;
        // int32_t output_offset = op_params.output_offset;
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

        /// new code two add driver code -- rpp
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

        const int32 input_offset = -input->params.zero_point;
        const int32 output_offset = output->params.zero_point;
        // Set min and max value of the output.
        const int32 output_activation_min = data->output_activation_min;
        const int32 output_activation_max = data->output_activation_max;

        const int8 *gemm_input_data = nullptr;
        const RuntimeShape *gemm_input_shape = nullptr;
        // const int filter_width = filter_shape.Dims(2);
        // const int filter_height = filter_shape.Dims(1);
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

        //  Load & Reshape Input data to temporary buffers before offloading
        //  to DMA inbuffers
        int width = gemm_input_cols;
        int w = ((width + 3) - ((width + 3) % 4));
        int depth = filter_cols;
        int d = ((depth + 15) - ((depth + 15) % 16));
        int s_need = w * d / 4 + 1;
        int8_t inb0[s_need];
        int8_t inb1[s_need];
        int8_t inb2[s_need];
        int8_t inb3[s_need];
        precal_sum_load_pad(gemm_input_data, width, depth, inb0, inb1, inb2,
                            inb3);

        int *inb_0 = reinterpret_cast<int *>(inb0);
        int *inb_1 = reinterpret_cast<int *>(inb1);
        int *inb_2 = reinterpret_cast<int *>(inb2);
        int *inb_3 = reinterpret_cast<int *>(inb3);

        int *wb_0 = reinterpret_cast<int *>(&wb0[i][0]);
        int *wb_1 = reinterpret_cast<int *>(&wb1[i][0]);
        int *wb_2 = reinterpret_cast<int *>(&wb2[i][0]);
        int *wb_3 = reinterpret_cast<int *>(&wb3[i][0]);

        // acc_container is used to wrap all the paramters the
        // gemm_driver/accelerator needs from the delegate
        drv.acc = acc;
        drv.mdma = &mdma;
        drv.profile = &profile;
        drv.st_params = st_params;
        drv.dfs = dfs;
        drv.mt_context = &dparams.mt_context;
        drv.thread_count = context->recommended_num_threads;
        drv.in_id = 0;
        drv.wb_0 = wb_0;
        drv.wb_1 = wb_1;
        drv.wb_2 = wb_2;
        drv.wb_3 = wb_3;
        drv.wt_sum1 = wt_sum1[i];
        drv.wt_sum2 = wt_sum2[i];
        drv.wt_sum3 = wt_sum3[i];
        drv.wt_sum4 = wt_sum4[i];
        drv.crf = crf[i];
        drv.crx = crx[i];
        drv.inb_0 = inb_0;
        drv.inb_1 = inb_1;
        drv.inb_2 = inb_2;
        drv.inb_3 = inb_3;
        drv.ra = output_offset;
        drv.t = &t;
        // drv.t->layer = associated_nodes[i];
        drv.recv_len = recv_len;
        drv.rows = gemm_input_cols;
        drv.cols = filter_rows;
        drv.depth = filter_cols;
        drv.use_sim = false;
        prf_end(1, vm_t.ipack);
        // drv.t2 = vm_t;
        vit_sim::Entry(drv, output_data);
        // vm_t = drv.t2;

        // saveMatrixCSV("aData/conv_fc/" +
        // std::to_string(associated_nodes[i])
        // +
        //                   "_out_acc.csv",
        // output_data, gemm_input_cols, filter_rows);
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
  // std::vector<int *> biases;
  std::vector<std::vector<int>> biases;
  std::vector<std::vector<int8_t>> temp_im2col;
  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int8_t>> crx;

  // std::vector<OpData *> opdatas;
  // std::vector<TfLiteConvParams *> cparams;

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
