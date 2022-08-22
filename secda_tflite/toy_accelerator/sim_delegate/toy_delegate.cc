/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define SYSC

#include <fstream>
#include <iostream>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_tflite/toy_accelerator/sim_delegate/toy_delegate.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/toy_accelerator/sim_delegate/util.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include "accelerator/driver/add_driver.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

// #include <cstring>
// #include <strstream>
// #include <chrono>
// #include <typeinfo>
// #include <cmath>

// #include <string>
// #include <vector>
// #include <sys/time.h>
// #include <cstdlib>

using namespace std;

unsigned int delegated_nodes = 0;
unsigned int layer = 0;
bool sysc_init = false;

ACCNAME* acc;
struct stream_dma* sdma;
struct systemC_sigs* scs;
struct Profile* profile;

namespace tflite {
namespace toy_test {

int rounddown(int x, int r) { return x - x % r; }

int roundup(int x, int r) { return rounddown(x + (r - 1), r); }

inline int32_t RoundingDivideByPOT(int32_t x, int exponent) {
  std::int32_t msk = (1 << exponent) - 1;
  std::int32_t sm = msk >> 1;
  std::int32_t val_3 = x >> exponent;

  std::int32_t temp_2 = x & msk;
  std::int32_t temp_3 = (x < 0) & 1;
  std::int32_t temp_4 = sm + temp_3;
  std::int32_t temp_5 = ((temp_2 > temp_4) & 1);
  std::int32_t result_32 = val_3 + temp_5;
}

inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                      std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(
    int32_t x, int32_t quantized_multiplier, int left_shift) {
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}

// Toy delegate kernel.
class ToyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit ToyDelegateKernel(const ToyDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Init SystemC

    if (!sysc_init) {
      sysC_init();
      static ACCNAME accelerator("accelerator");
      static struct stream_dma _sdma(0, 0, 8096, 0, 8096);
      // static AXIS_ENGINE _dmad("dmad");
      static struct systemC_sigs _scs(1);
      static struct Profile _profile;
      // _profile.addMetric(ClockCycles("per_batch_cycles"));
      // _profile.addMetric(ClockCycles("active_cycles"));
      systemC_binder(&accelerator, &_sdma, &_scs);
      acc = &accelerator;
      sdma = &_sdma;
      scs = &_scs;
      profile = &_profile;
      sysc_init = true;
    }

    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);

    opdatas.resize(params->nodes_to_replace->size);
    cparams.resize(params->nodes_to_replace->size);

    // wgt_sum.resize(params->nodes_to_replace->size);
    // biases.resize(params->nodes_to_replace->size);
    // crf.resize(params->nodes_to_replace->size);
    // crx.resize(params->nodes_to_replace->size);
    // weight_offsets.resize(params->nodes_to_replace->size);
    // del_weights.resize(params->nodes_to_replace->size);

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
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);

      TfLiteAddParams* cparam =
          reinterpret_cast<TfLiteAddParams*>(delegated_node->builtin_data);
      OpData* opdata = reinterpret_cast<OpData*>(delegated_node->user_data);

      cparams[i] = cparam;
      opdatas[i] = opdata;
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    // KernelType kernel_type = kCblasOptimized;

    int node_count = inputs_.size();
    int out_tid = 0;

    for (int i = 0; i < node_count; i++) {
      TfLiteAddParams* params = cparams[i];
      OpData* data = opdatas[i];

      const TfLiteTensor* input1;
      const TfLiteTensor* input2;
      TfLiteTensor* output;

      GetInputSafe(context, inputs_[i][0], &input1);
      GetInputSafe(context, inputs_[i][1], &input2);
      GetOutputSafe(context, outputs_[i][0], &output);
      output->type = input2->type;

      TfLiteIntArray* output_size = TfLiteIntArrayCopy(input1->dims);
      const bool requires_broadcast = false;

      bool general_scale_int16 = false;
      bool input1_scale_is_pot = false;
      bool input2_scale_is_pot = false;
      bool output_scale_is_pot = false;
      int input1_scale_log2_rounded{0};
      int input2_scale_log2_rounded{0};
      int output_scale_log2_rounded{0};

      data->pot_scale_int16 = !general_scale_int16;
      // 8bit -> 8bit general quantized path, with general rescalings
      // as well as, 16bit -> 16bit with general rescalings
      data->input1_offset = -input1->params.zero_point;
      data->input2_offset = -input2->params.zero_point;
      data->output_offset = output->params.zero_point;

      // The shift is set to 15 for 16-bit and 20 in case of 8-bit, accordingly.
      // In case of 16-bit we have 65535 << 15 which is less than 1 << 31,
      // therefore the addition will still fit in a 32 bit accumulator.
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

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    int node_count = inputs_.size();

    for (int i = 0; i < node_count; i++) {
      cout << "===========================" << endl;
      cout << "ADD Layer: " << layer << "      Node: " << associated_nodes[i]
           << endl;
      cout << "===========================" << endl;

      auto* params = cparams[i];
      OpData* data = opdatas[i];

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

      // auto dims = tflite::GetTensorShape(input1);

      int size = 1;
      for (int i = 0; i < input1->dims->size; ++i) {
        size *= input1->dims->data[i];
      }

      const int8* input1_data = input1->data.int8;
      const int8* input2_data = input2->data.int8;
      int8* output_data = output->data.int8;

      struct acc_container drv;
      drv.scs = scs;
      drv.profile = profile;
      drv.acc = acc;
      drv.sdma = sdma;

      drv.input_A = input1_data;
      drv.input_B = input2_data;
      drv.output_C = output_data;

      drv.length = rounddown(size, 4);
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

      for (int j = 0; j < size; j++) {
        const int32 input1_val = op_params.input1_offset + input1_data[j];
        const int32 input2_val = op_params.input2_offset + input2_data[j];
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
                raw_sum, op_params.output_multiplier, op_params.output_shift) +
            op_params.output_offset;
        const int32 clamped_output =
            std::min(op_params.quantized_activation_max,
                     std::max(op_params.quantized_activation_min, raw_output));
        output_data[j] = static_cast<int8>(clamped_output);
      }

      tflite_secda::Entry(drv);

      layer++;
      delegated_nodes--;
    }

    if (delegated_nodes == 0) {
      profile->saveCSVRecords("z_TOY_Sim");
    }

    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;

  std::vector<std::vector<int>> wgt_sum;
  std::vector<int> weight_offsets;

  std::vector<uint32_t*> del_weights;

  std::vector<int*> biases;
  std::vector<int*> crf;
  std::vector<int8_t*> crx;

  std::vector<OpData*> opdatas;
  std::vector<TfLiteAddParams*> cparams;

 private:
  const ToyDelegateOptions options_;
};

// ToyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class ToyDelegate : public SimpleDelegateInterface {
 public:
  explicit ToyDelegate(const ToyDelegateOptions& options) : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports FC ops
    if (kTfLiteBuiltinAdd != registration->builtin_code) return false;

    if (node->inputs->size != 2) return false;

    // This delegate only supports int8 types.
    for (int i = 0; i < 2; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    TfLiteTensor input1 = context->tensors[node->inputs->data[0]];
    TfLiteTensor input2 = context->tensors[node->inputs->data[1]];

    if (!TfLiteIntArrayEqual(input1.dims, input2.dims)) return false;

    // ADD
    // cout << delegated_nodes << endl;
    delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "ToyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<ToyDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const ToyDelegateOptions options_;
};

}  // namespace toy_test
}  // namespace tflite

ToyDelegateOptions TfLiteToyDelegateOptionsDefault() {
  ToyDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this toy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteToyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteToyDelegateCreate(const ToyDelegateOptions* options) {
  std::unique_ptr<tflite::toy_test::ToyDelegate> toy(
      new tflite::toy_test::ToyDelegate(
          options ? *options : TfLiteToyDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(toy));
}

// Destroys a delegate created with `TfLiteToyDelegateCreate` call.
void TfLiteToyDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
