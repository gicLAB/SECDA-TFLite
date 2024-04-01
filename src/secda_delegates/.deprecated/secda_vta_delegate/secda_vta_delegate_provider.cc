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
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_delegates/rvta_delegate/rvta_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class RVtaDelegateProvider : public DelegateProvider {
 public:
  RVtaDelegateProvider() {
    default_params_.AddParam("use_rvta_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "RVtaDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(RVtaDelegateProvider);

std::vector<Flag> RVtaDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_rvta_delegate", params,
                                              "use the rvta delegate.")};
  return flags;
}

void RVtaDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_rvta_delegate", "Use rvta test delegate",
                 verbose);
}

TfLiteDelegatePtr RVtaDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_rvta_delegate")) {
    auto default_options = TfLiteRVtaDelegateOptionsDefault();
    return TfLiteRVtaDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
RVtaDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_rvta_delegate"));
}
}  // namespace tools
}  // namespace tflite
