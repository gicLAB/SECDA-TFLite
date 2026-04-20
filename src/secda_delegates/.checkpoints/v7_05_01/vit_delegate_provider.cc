#include <string>
#include <utility>

#include "vit_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class VitDelegateProvider : public DelegateProvider {
public:
  VitDelegateProvider() {
    default_params_.AddParam("use_vit_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "VitDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(VitDelegateProvider);

std::vector<Flag> VitDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_vit_delegate", params, "use the vit delegate.")};
  return flags;
}

void VitDelegateProvider::LogParams(const ToolParams &params,
                                     bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_vit_delegate", "Use vit test delegate",
                 verbose);
}

TfLiteDelegatePtr
VitDelegateProvider::CreateTfLiteDelegate(const ToolParams &params) const {
  if (params.Get<bool>("use_vit_delegate")) {
    auto default_options = TfLiteVitDelegateOptionsDefault();
    return TfLiteVitDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
VitDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_vit_delegate"));
}
} // namespace tools
} // namespace tflite