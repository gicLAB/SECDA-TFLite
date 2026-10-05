#include <string>
#include <utility>

#include "omni_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class OmniDelegateProvider : public DelegateProvider {
public:
  OmniDelegateProvider() {
    default_params_.AddParam("use_omni_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "OmniDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(OmniDelegateProvider);

std::vector<Flag> OmniDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_omni_delegate", params, "use the omni delegate.")};
  return flags;
}

void OmniDelegateProvider::LogParams(const ToolParams &params,
                                     bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_omni_delegate", "Use omni test delegate",
                 verbose);
}

TfLiteDelegatePtr
OmniDelegateProvider::CreateTfLiteDelegate(const ToolParams &params) const {
  if (params.Get<bool>("use_omni_delegate")) {
    auto default_options = TfLiteOmniDelegateOptionsDefault();
    return TfLiteOmniDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
OmniDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_omni_delegate"));
}
} // namespace tools
} // namespace tflite