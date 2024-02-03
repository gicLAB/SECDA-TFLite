#include <string>
#include <utility>

#include "temp_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class TempDelegateProvider : public DelegateProvider {
public:
  TempDelegateProvider() {
    default_params_.AddParam("use_temp_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "TempDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(TempDelegateProvider);

std::vector<Flag> TempDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_temp_delegate", params, "use the temp delegate.")};
  return flags;
}

void TempDelegateProvider::LogParams(const ToolParams &params,
                                     bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_temp_delegate", "Use temp test delegate",
                 verbose);
}

TfLiteDelegatePtr
TempDelegateProvider::CreateTfLiteDelegate(const ToolParams &params) const {
  if (params.Get<bool>("use_temp_delegate")) {
    auto default_options = TfLiteTempDelegateOptionsDefault();
    return TfLiteTempDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
TempDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_temp_delegate"));
}
} // namespace tools
} // namespace tflite