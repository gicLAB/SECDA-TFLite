
#include <iomanip>
#include <iostream>
#include <utility>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif
#include "secda_tools/secda_profiler/profiler.h"

#include "accelerator/driver/mm2im_driver.h"
#include "mm2im_delegate.h"
#include "util.h"

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#define DELEGATE_NAME "MM2IM"
#define DELEGATE_VERSION 10
// Some variables needs to be defined across multiple instances of the delegate
unsigned int dma_addrs[1] = {dma_addr0};
unsigned int dma_addrs_in[1] = {dma_in0};
unsigned int dma_addrs_out[1] = {dma_out0};
struct mm2im_times p_t;
static struct preloader preloader;
#ifdef SYSC
#define SYSC_DMA_BL 563840
// static struct s_mdma mdma(1, dma_addrs, dma_addrs_in, dma_addrs_out,
//                           SYSC_DMA_BL);
static struct s_mdma mdma(1, dma_addrs, dma_addrs_in, dma_addrs_out,
                          DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);
ACCNAME *acc;
#else
// struct s_mdma mdma(1, dma_addrs, dma_addrs_in, dma_addrs_out, DMA_BL);
static s_mdma mdma(1, dma_addrs, dma_addrs_in, dma_addrs_out, DMA_IN_BUF_SIZE,
                   DMA_OUT_BUF_SIZE);
int *acc;
#endif

struct del_params dparams;
static struct Profile profile;
struct MultiThreadContext mt_context;

// #define DELEGATE_VERBOSE
#ifdef DELEGATE_VERBOSE
#define DTOG(X) X
#else
#define DTOG(X)
#endif

namespace tflite {
namespace mm2im_test {

// MM2IM delegate kernel
class MM2IMDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit MM2IMDelegateKernel(const MM2IMDelegateOptions &options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {
    // // Init SystemC Modules & Profilier
    if (!dparams.init) {
      DTOG(std::cerr << "===========================" << std::endl);
#ifdef SYSC
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("MM2IM");
      sysC_init();
      sysC_binder(&_acc, &mdma, &scs1);
      acc = &_acc;
      DTOG(std::cerr << "Initialised the SystemC Modules" << std::endl);
#else
      dparams.acc = getAccBaseAddress<int>(acc_address, 65536);
      acc = dparams.acc;
      DTOG(std::cerr << "Initialised the DMA" << std::endl);
#endif
      preloader.init(mdma.dmas[0].dma_get_inbuffer());
      DTOG(std::cerr << "MM2IM Accelerator v10");
#ifdef ACC_NEON
      DTOG(std::cerr << " with Neon");
#endif
      DTOG(std::cerr << std::endl);
      DTOG(std::cerr << "===========================" << std::endl);
      dparams.init = true;
    }

    // Save Tensors input & outputs
    // Save other info (opdata)
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    opdatas.resize(params->nodes_to_replace->size);
    tparams.resize(params->nodes_to_replace->size);
    biases.resize(params->nodes_to_replace->size);
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    crx_8.resize(params->nodes_to_replace->size);
    acc_weights.resize(params->nodes_to_replace->size);
    acc_inputs.resize(params->nodes_to_replace->size);
    acc_wt_sums.resize(params->nodes_to_replace->size);
    n_weights.resize(params->nodes_to_replace->size);
    swapped_weights.resize(params->nodes_to_replace->size);
    acc_dex_maps.resize(params->nodes_to_replace->size);
    // mm2im_params.resize(params->nodes_to_replace->size);

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
      inputs_[i].push_back(delegated_node->inputs->data[3]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);
      TfLiteTransposeConvParams *tparam =
          reinterpret_cast<TfLiteTransposeConvParams *>(
              delegated_node->builtin_data);
      OpData *opdata = reinterpret_cast<OpData *>(delegated_node->user_data);
      tparams[i] = tparam;
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
      TfLiteTransposeConvParams *params = tparams[i];
      OpData *data = opdatas[i];

      TfLiteTensor *output;
      const TfLiteTensor *output_shape;
      const TfLiteTensor *input;
      const TfLiteTensor *weights;
      const TfLiteTensor *bias;

      GetOutputSafe(context, outputs_[i][0], &output);
      GetInputSafe(context, inputs_[i][0], &output_shape);
      GetInputSafe(context, inputs_[i][1], &weights);
      GetInputSafe(context, inputs_[i][2], &input);
      GetInputSafe(context, inputs_[i][3], &bias);

      int stride_x = params->stride_width;
      int stride_y = params->stride_height;
      int filters = weights->dims->data[0];
      int kernel_size = weights->dims->data[1];
      int in1 = input->dims->data[1];
      int in2 = input->dims->data[2];
      int in3 = weights->dims->data[3];
      // string padding = params->padding == kTfLitePaddingSame ? "same" :
      // "valid";
      string padding = params->padding == kTfLitePaddingSame ? "same" : "same";

      int out1, out2, out3, rows, cols, depth;
      int pt, pb, pl, pr;
      // cerr << "====================================" << endl;
      calParams(stride_x, stride_y, filters, kernel_size, in1, in2, in3,
                padding, rows, cols, depth, out1, out2, out3, pt, pb, pl, pr);
      // cerr << "====================================" << endl;
      // Input Params
      int ra = 26; // TODO: Make this dynamic

      int rhs_offset = -input->params.zero_point;
      // int rhs_offset = 0;
      int lhs_offset = 0;
      int output_height = out1;
      int output_width = out2;
      int output_depth = out3;
      int filter_height = kernel_size;
      int filter_width = kernel_size;
      int padding_top = pt;
      int padding_left = pl;
      int padding_bottom = pb;
      int padding_right = pr;
      int stride_height = stride_x;
      int stride_width = stride_y;
      int rounded_depth = roundUp(in3, UF);

      TfLiteIntArray *oshape = TfLiteIntArrayCreate(output->dims->size);
      oshape->data[0] = 1;
      oshape->data[1] = out1;
      oshape->data[2] = out2;
      oshape->data[3] = out3;

      if (SizeOfDimension(input, 3) != SizeOfDimension(weights, 3))
        return kTfLiteError;

      OpData *user_data = opdatas[i];
      int temp_out_id;
      bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      if (!req_temp_out) out_tid++;

      TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
          context, node, params, data, req_temp_out, outputs_[i][0],
          temp_out_id));

      TfLiteTensor *col2im = nullptr;
      if (data->has_col2im) {
        node->temporaries->data[data->col2im_index] = data->col2im_id;
        TF_LITE_ENSURE_OK(
            context,
            GetTemporarySafe(context, node, user_data->col2im_index, &col2im));
      }

      // Output & Col2Im
      if (!IsConstantTensor(output_shape) && false) {
        // Defer resizing until Eval().
        SetTensorToDynamic(output);
        if (data->has_col2im) SetTensorToDynamic(col2im);
      } else {
        TF_LITE_ENSURE_STATUS(ResizeTensor3(context, oshape, output));
        if (data->has_col2im) {
          TF_LITE_ENSURE_STATUS(ResizeCol2ImTensor(context, output_shape,
                                                   weights, input, col2im));
        }
      }

      // Temporary output
      if (req_temp_out && !IsConstantTensor(output_shape)) {
        node->temporaries->data[temp_out_id] = outputs_[i][0];
        TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteDynamic;
        // temp_out_tensor->allocation_type = kTfLiteArenaRw;
        TF_LITE_ENSURE_STATUS(ResizeTensor2(context, output, temp_out_tensor));
        // TF_LITE_ENSURE_STATUS(ResizeTensor3(context, oshape,
        // temp_out_tensor));
      }

      // Weights
      if (data->weights_are_transposed) {
        node->temporaries->data[data->transposed_weights_index] =
            data->transposed_weights_id;
        TfLiteTensor *transposed_weights;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node,
                                           user_data->transposed_weights_index,
                                           &transposed_weights));
        if (!IsConstantTensor(weights)) {
          SetTensorToDynamic(transposed_weights);
        } else {
          ResizeAndTransposeWeights(context, weights, transposed_weights);
        }
      }

      // Scratch buffer
      node->temporaries->data[data->scratch_tensor_index] =
          data->scratch_tensor_id;
      TfLiteTensor *scratch_buffer;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                                  data->scratch_tensor_index,
                                                  &scratch_buffer));

      scratch_buffer->type = kTfLiteInt32;
      scratch_buffer->allocation_type = kTfLiteDynamic;
      // scratch_buffer->allocation_type = kTfLiteArenaRw;
      if (!IsConstantTensor(output_shape) && false) {
        SetTensorToDynamic(scratch_buffer);
      } else {
        TF_LITE_ENSURE_STATUS(ResizeTensor2(context, output, scratch_buffer));
      }

      TF_LITE_ENSURE_EQ(context, weights->quantization.type,
                        kTfLiteAffineQuantization);
      const auto *affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization *>(
              weights->quantization.params);
      const int channels_out = weights->dims->data[0];
      TF_LITE_ENSURE(context, affine_quantization);
      TF_LITE_ENSURE(context, affine_quantization->scale);
      TF_LITE_ENSURE(context,
                     (affine_quantization->scale->size == 1 ||
                      affine_quantization->scale->size == channels_out));

      data->per_channel_output_multiplier.resize(channels_out);
      data->per_channel_output_shift.resize(channels_out);

      crf[i].resize(channels_out);
      crx[i].resize(channels_out);
      crx_8[i].resize(channels_out);
      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
          context, input, weights, bias, output, kTfLiteActNone,
          &data->output_multiplier, &data->output_shift,
          &data->output_activation_min, &data->output_activation_max,
          &crf[i][0], data->per_channel_output_shift.data(), channels_out));
      for (int j = 0; j < channels_out; j++) {
        crx[i][j] = data->per_channel_output_shift.data()[j];
        crx_8[i][j] = (int8_t)data->per_channel_output_shift.data()[j];
      }
      biases[i] = bias->data.i32;

      TfLiteTensor *transposed_weights;
      TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node,
                                         user_data->transposed_weights_index,
                                         &transposed_weights));

      acc_weights[i].resize(rounded_depth * rows);
      acc_inputs[i].resize(rounded_depth * cols);
      acc_wt_sums[i].resize(filters * kernel_size * kernel_size);
      acc_dex_maps[i].resize(in1 * in2 * filters * kernel_size * kernel_size);
      n_weights[i].resize(in3 * filters * kernel_size * kernel_size);
      swapped_weights[i].resize(in3 * filters * kernel_size * kernel_size);
      int8_t *acc_weight = &acc_weights[i][0];
      int *acc_wt_sum = &acc_wt_sums[i][0];
      int32_t *acc_dex_map = &acc_dex_maps[i][0];
      int8_t *n_wgt = &n_weights[i][0];
      int8_t *swapped_weight = &swapped_weights[i][0];
      struct mm2im_params amp(stride_x, stride_y, filters, kernel_size, in1,
                              in2, in3, rows, cols, depth, out1, out2, out3,
                              padding_top, padding_bottom, padding_left,
                              padding_right, true);

      memcpy(n_wgt, transposed_weights->data.int8,
             in3 * filters * kernel_size * kernel_size * sizeof(int8_t));
      swap_weights(transposed_weights->data.int8, swapped_weight, filters,
                   kernel_size, in3);
      preload_weights(swapped_weight, depth, rows, acc_wt_sum, acc_weight);

      bool has_bias = inputs_[i][3] != 0;
      int *bias_send = nullptr;
      vector<int> fake_bias(output_depth, 0);
      ;
      if (has_bias) bias_send = biases[i];
      else bias_send = &fake_bias[0];
      int32_t *loaded_wgts = reinterpret_cast<int32_t *>(acc_weight);
      preloader.dma_preload(loaded_wgts, kernel_size, filters, rounded_depth,
                            rhs_offset, acc_wt_sum, &crf[i][0], &crx_8[i][0],
                            bias_send);
      if (preloader.layer == dparams.delegated_nodes ||
          preloader.allocation_done) {
        preloader.allocation_done = true;
        DTOG(std::cerr << "=============================" << std::endl);
        DTOG(std::cerr << "Preloaded weights for " << preloader.layer << " / "
                       << dparams.delegated_nodes << " layers" << std::endl);
        DTOG(std::cerr << "Allocated size: " << preloader.alloced_size * 4
                       << " bytes" << std::endl);
        DTOG(std::cerr << "Loaded tiles: " << preloader.loaded_tiles
                       << std::endl);
        DTOG(std::cerr << "=============================" << std::endl);
      }

      col2im_mapping_v3(amp.oc, amp.oh, amp.ow, amp.ks, amp.ks, amp.pt, amp.pl,
                        amp.pb, amp.pr, amp.sx, amp.sy, acc_dex_map);
      amp.dex_map = acc_dex_map;
      amp.MM2IM_oh_map();
      for (int j = 0; j < amp.mm2im_map.size(); j++) {
        // free vector
        amp.mm2im_map[j].clear();
        // shrink vector
        amp.mm2im_map[j].shrink_to_fit();
      }
      amp.mm2im_map.clear();
      amp.mm2im_map.shrink_to_fit();
      mm2im_params.push_back(amp);
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
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      prf_start(1);
      auto *params = tparams[i];
      OpData *data = opdatas[i];
      const TfLiteTensor *input;
      const TfLiteTensor *weights;
      TfLiteTensor *output;
      const TfLiteTensor *bias;
      const TfLiteTensor *output_shape_tensor;
      bool has_bias = inputs_[i][3] != 0;

      TfLiteTensor *transposed_weights =
          data->weights_are_transposed
              ? GetTemporary(context, node, data->transposed_weights_index)
              : nullptr;

      TfLiteTensor *col2im =
          data->has_col2im ? GetTemporary(context, node, data->col2im_index)
                           : nullptr;

      TfLiteTensor *scratch_buffer;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                                  data->scratch_tensor_index,
                                                  &scratch_buffer));

      GetInputSafe(context, inputs_[i][0], &output_shape_tensor);
      GetInputSafe(context, inputs_[i][2], &input);
      GetInputSafe(context, inputs_[i][1], &weights);
      GetInputSafe(context, inputs_[i][3], &bias);
      GetOutputSafe(context, outputs_[i][0], &output);

      // Resize any deferred dynamic tensors
      if (tflite::IsDynamicTensor(output)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeTensor(context, output_shape_tensor, output));
      }
      if (data->has_col2im && tflite::IsDynamicTensor(col2im)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeCol2ImTensor(context, output_shape_tensor,
                                             weights, input, col2im));
      }
      if (tflite::IsDynamicTensor(scratch_buffer)) {
        TF_LITE_ENSURE_OK(context, ResizeTensor(context, output_shape_tensor,
                                                scratch_buffer));
      }

      // Get height and width of the output image.
      const int o_width = SizeOfDimension(output, 2);
      const int o_height = SizeOfDimension(output, 1);
      const int filter_w = SizeOfDimension(weights, 2);
      const int filter_h = SizeOfDimension(weights, 1);

      int unused_output_height, unused_output_width;
      data->padding = tflite::ComputePaddingHeightWidth(
          params->stride_height, params->stride_width, 1, 1, o_height, o_width,
          filter_h, filter_w, params->padding, &unused_output_height,
          &unused_output_width);

      tflite::ConvParams op_params;
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = data->padding.width;
      op_params.padding_values.height = data->padding.height;
      op_params.padding_values.width_offset = data->padding.width_offset;
      op_params.padding_values.height_offset = data->padding.height_offset;
      op_params.stride_width = params->stride_width;
      op_params.stride_height = params->stride_height;
      // Need to flip the sign of input offset to add it directly to the
      // quantized buffer.
      op_params.input_offset = -input->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.quantized_activation_min = data->output_activation_min;
      op_params.quantized_activation_max = data->output_activation_max;

      ConvParams &cparams = op_params;
      const RuntimeShape &input_shape = GetTensorShape(input);
      const RuntimeShape &hwoi_ordered_filter_shape =
          GetTensorShape(transposed_weights);

      const RuntimeShape &output_shape = GetTensorShape(output);
      const RuntimeShape &col2im_shape = GetTensorShape(col2im);
      const RuntimeShape &scratch_shape = GetTensorShape(scratch_buffer);
      int32 *output_multiplier = &crf[i][0];
      int32 *output_shift = &crx[i][0];
      const int8_t *input_data = GetTensorData<int8>(input);
      const int8_t *hwoi_ordered_filter_data =
          GetTensorData<int8>(transposed_weights);
      int8_t *output_data = GetTensorData<int8>(output);
      int32_t *col2im_data = GetTensorData<int32>(col2im);
      int32_t *scratch_data = GetTensorData<int32>(scratch_buffer);
      CpuBackendContext *cpu_backend_context =
          CpuBackendContext::GetFromContext(context);

      const int batch_size =
          tflite::MatchingDim(input_shape, 0, output_shape, 0);
      const int input_image_size = input_shape.Dims(1) * input_shape.Dims(2);
      const int output_height = output_shape.Dims(1);
      const int output_width = output_shape.Dims(2);
      const int output_image_size = output_height * output_width;
      const int input_depth =
          tflite::MatchingDim(input_shape, 3, hwoi_ordered_filter_shape, 3);
      const int output_depth =
          tflite::MatchingDim(output_shape, 3, hwoi_ordered_filter_shape, 2);
      const int input_offset = input_image_size * input_depth;
      const int output_offset = output_image_size * output_depth;

      const int filter_height = hwoi_ordered_filter_shape.Dims(0);
      const int filter_width = hwoi_ordered_filter_shape.Dims(1);
      const int padding_top = cparams.padding_values.height;
      const int padding_bottom =
          cparams.padding_values.height + cparams.padding_values.height_offset;
      const int padding_left = cparams.padding_values.width;
      const int padding_right =
          cparams.padding_values.width + cparams.padding_values.width_offset;
      const int stride_height = cparams.stride_height;
      const int stride_width = cparams.stride_width;

      const int hwoi_ordered_filter_total_size =
          filter_height * filter_width * output_depth;

      const RuntimeShape &bias_shape = GetTensorShape(bias);
      const int32 *bias_data = GetTensorData<int32>(bias);

      cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
      lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
      lhs_params.rows = hwoi_ordered_filter_total_size;
      lhs_params.cols = input_depth;
      // Since our weight is symmetric quantized, the zp will always be 0.
      lhs_params.zero_point = 0;

      int32_t *scratch_data_p = scratch_data;
      std::fill_n(scratch_data, output_offset * batch_size,
                  static_cast<int32>(0));

      cpu_backend_gemm::MatrixParams<int8_t> rhs_params;
      rhs_params.order = cpu_backend_gemm::Order::kColMajor;
      rhs_params.rows = input_depth;
      rhs_params.cols = input_image_size;
      rhs_params.zero_point = -cparams.input_offset;

      cpu_backend_gemm::MatrixParams<int32_t> dst_params;
      dst_params.order = cpu_backend_gemm::Order::kColMajor;
      dst_params.rows = hwoi_ordered_filter_total_size;
      dst_params.cols = input_image_size;

      cpu_backend_gemm::GemmParams<int32_t, int32_t> gemm_params;

      const int gemm_input_rows = input_shape.Dims(3);
      const int gemm_input_cols = FlatSizeSkipDim(input_shape, 3);
      const int filter_rows = hwoi_ordered_filter_shape.Dims(0);
      const int filter_cols = FlatSizeSkipDim(hwoi_ordered_filter_shape, 0);
      const int output_rows = output_shape.Dims(3);
      const int output_cols =
          output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
      const int scratch_cols = scratch_shape.Dims(1) * scratch_shape.Dims(2);
      const int scratch_rows = scratch_shape.Dims(0) * scratch_shape.Dims(3);

      // ========================================= //
      // ========================================= //
      // Accelerator config
      struct mm2im_params *par = &mm2im_params[i];
      int rounded_depth = roundUp(par->ic, UF);
      int8_t *acc_input = &acc_inputs[i][0];
      bool input_preloaded = false;
      if (preloader.allocation_done) {
        input_preloaded = preload_inputs_dma(input_data, par->depth, par->cols,
                                             mdma.dmas[0].dma_get_inbuffer());
      }
      if (!input_preloaded)
        preload_inputs(input_data, par->depth, par->cols, acc_input);

      int32_t *acc_loaded_wgts =
          reinterpret_cast<int32_t *>(&acc_weights[i][0]);
      int32_t *acc_loaded_inps = reinterpret_cast<int32_t *>(acc_input);

      struct acc_container drv;
      drv.acc = acc;
      drv.mdma = &mdma;
      drv.profile = &profile;
      drv.mt_context = &mt_context;
      drv.preloader = &preloader;
      drv.thread_count = context->recommended_num_threads;
      if (drv.oh * drv.ow * drv.oc <= 3136) drv.thread_count = 1;

      drv.lhs_offset = 0;
      drv.ih = par->ih;
      drv.iw = par->iw;
      drv.ic = par->ic;
      drv.f = par->f;
      drv.ks = par->ks;
      drv.oh = par->oh;
      drv.ow = par->ow;
      drv.oc = par->oc;
      drv.sx = par->sx;
      drv.sy = par->sy;
      drv.pt = par->pt;
      drv.pl = par->pl;
      drv.pb = par->pb;
      drv.pr = par->pr;
      drv.width_col = par->width_col;
      drv.rows = par->rows;
      drv.cols = par->cols;
      drv.depth = par->depth;
      drv.loaded_weights = acc_loaded_wgts;
      drv.loaded_inputs = acc_loaded_inps;
      drv.weights = &swapped_weights[i][0];
      drv.inputs = input_data;
      drv.input_preloaded = input_preloaded;

      vector<int> fake_bias(output_depth, 0);
      // int32_t fake_bias[output_depth];
      if (has_bias) drv.bias = biases[i];
      else drv.bias = &fake_bias[0];
      drv.crf = &crf[i][0];
      drv.crx = &crx_8[i][0];
      drv.ra = par->ra;
      drv.oh_ends = &par->oh_ends[0];

      drv.output_data = output_data;
      drv.acc_wt_sum = &acc_wt_sums[i][0];
      drv.ra = cparams.output_offset;
      drv.rhs_offset = -rhs_params.zero_point;
      drv.lhs_offset = 0;
      drv.t.layer = dparams.layer;
      prf_end(1, p_t.p_ipack);

// CPU implemented here does provide correct results
#ifdef RUN_CPU_TCONV
      cpu_backend_gemm::Gemm(lhs_params, hwoi_ordered_filter_data, rhs_params,
                             input_data, dst_params, col2im_data, gemm_params,
                             cpu_backend_context);
      optimized_ops::Col2im(
          col2im_data, output_depth, output_height, output_width, filter_height,
          filter_width, padding_top, padding_left, padding_bottom,
          padding_right, stride_height, stride_width, scratch_data_p);
      // saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
      //                   "_col2im_cpu.csv",
      //               scratch_data_p, scratch_cols, scratch_rows);
      if (has_bias)
        optimized_ops::BiasAdd(scratch_data_p, bias_data, batch_size,
                               output_height, output_width, output_depth);
      const int32_t output_min = std::numeric_limits<int8_t>::min();
      const int32_t output_max = std::numeric_limits<int8_t>::max();
      optimized_ops::Quantize(output_multiplier, output_shift, output_depth,
                              output_shape.FlatSize(), cparams.output_offset,
                              output_min, output_max, scratch_data,
                              output_data);
      // DirectMM2IM(hwoi_ordered_filter_data, input_data, output_data);
      // saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
      //                   "_wgt.csv",
      //               hwoi_ordered_filter_data, lhs_params.rows,
      //               lhs_params.cols);

      // saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
      //                   "_inp.csv",
      //               input_data, rhs_params.rows, rhs_params.cols);
      // saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
      //                   "_gemm_cpu.csv",
      //               col2im_data, dst_params.cols, dst_params.rows);
      // saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
      //                   "_biasadded_cpu.csv",
      //               scratch_data_p, scratch_cols, scratch_rows);
      saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
                        "_out_cpu.csv",
                    output_data, scratch_cols, scratch_rows);
#endif

#ifdef DELEGATE_VERBOSE
      // DTOG(cerr << "===========================" << endl);
      // DTOG(cerr << "Layer: " << dparams.layer
      //           << "      Node: " << associated_nodes[i] << endl);
      // DTOG(cerr << "===========================" << endl);
#endif
      drv.p_t = p_t;
      mm2im_driver::Entry(drv);
      p_t = drv.p_t;

#ifdef SYSC
      saveMatrixCSV("aData/mm2im/" + std::to_string(associated_nodes[i]) +
                        "_out_acc.csv",
                    output_data, scratch_cols, scratch_rows);
#endif
      dparams.layer++;
      if (dparams.layer % dparams.delegated_nodes == 0) {
        p_t.call_count++;
        preloader.reset_loading();
      }
    }
    prf_end(0, p_t.t_tconv);
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;

  std::vector<std::vector<int8_t>> n_weights;
  std::vector<std::vector<int8_t>> swapped_weights;
  std::vector<std::vector<int8_t>> acc_weights;
  std::vector<std::vector<int8_t>> acc_inputs;
  std::vector<std::vector<int>> acc_wt_sums;
  std::vector<std::vector<int>> acc_dex_maps;
  std::vector<struct mm2im_params> mm2im_params;

  std::vector<int *> biases;
  std::vector<OpData *> opdatas;
  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int>> crx;
  std::vector<std::vector<int8_t>> crx_8;
  std::vector<TfLiteTransposeConvParams *> tparams;

private:
  const MM2IMDelegateOptions options_;
};

// MM2IMDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class MM2IMDelegate : public SimpleDelegateInterface {
public:
  explicit MM2IMDelegate(const MM2IMDelegateOptions &options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {
    // Only supports MM2IM op
    if (kTfLiteBuiltinTransposeConv != registration->builtin_code) return false;

    // This delegate requires at least 3 inputs.
    if (node->inputs->size < 3) return false;

    // This delegate only supports int8 types.
    for (int i = 1; i < 3; ++i) {
      auto &tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    // Ensures output shape tensor is supports int32 type
    auto &tensor = context->tensors[node->inputs->data[0]];
    if (tensor.type != kTfLiteInt32) return false;

    if (node->inputs->size == 4) {
      // Ensures bias tensor is supports int32 type
      auto &tensor2 = context->tensors[node->inputs->data[3]];
      if (tensor2.type != kTfLiteInt32) return false;
    }

    auto &tensor0 = context->tensors[node->inputs->data[1]];
    int filter_dim = tensor0.dims->data[0];

    // TODO JUDE: Support any filter dim (make this better)
    // if (filter_dim < PE_COUNT) return false;

    // Adds node for delegation
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "MM2IMDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<MM2IMDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const MM2IMDelegateOptions options_;
};

} // namespace mm2im_test
} // namespace tflite

MM2IMDelegateOptions TfLiteMM2IMDelegateOptionsDefault() {
  MM2IMDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this mm2im test delegate
  // will not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteMM2IMDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteMM2IMDelegateCreate(const MM2IMDelegateOptions *options) {
  std::unique_ptr<tflite::mm2im_test::MM2IMDelegate> mm2im(
      new tflite::mm2im_test::MM2IMDelegate(
          options ? *options : TfLiteMM2IMDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(mm2im), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// TfLiteDelegate *
// TfLiteMM2IMDelegateCreate(const MM2IMDelegateOptions *options) {
//   std::unique_ptr<tflite::mm2im_test::MM2IMDelegate> mm2im(
//       new tflite::mm2im_test::MM2IMDelegate(
//           options ? *options : TfLiteMM2IMDelegateOptionsDefault()));
//   return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
//       std::move(mm2im));
// }

// Destroys a delegate created with `TfLiteMM2IMDelegateCreate` call.
void TfLiteMM2IMDelegateDelete(TfLiteDelegate *delegate) {
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
    DTOG(std::cerr << "===========================" << std::endl);
    DTOG(std::cerr << "Unmapped DMA I/O Buffers" << std::endl);
    DTOG(std::cerr << "===========================" << std::endl);
    dparams.unmap = true;
  }
#endif
  p_t.print();
  p_t.save_prf();
  DTOG(std::cerr << "===========================" << std::endl);
  DTOG(std::cerr << "Deleted" << std::endl);
  DTOG(std::cerr << "===========================" << std::endl);
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
