#include <string>
#include <utility>

#include "tempdel_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class TempdelDelegateProvider : public DelegateProvider {
public:
  TempdelDelegateProvider() {
    default_params_.AddParam("use_tempdel_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "TempdelDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(TempdelDelegateProvider);

std::vector<Flag> TempdelDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_tempdel_delegate", params, "use the tempdel delegate.")};
  return flags;
}

void TempdelDelegateProvider::LogParams(const ToolParams &params,
                                     bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_tempdel_delegate", "Use tempdel test delegate",
                 verbose);
}

TfLiteDelegatePtr
TempdelDelegateProvider::CreateTfLiteDelegate(const ToolParams &params) const {
  if (params.Get<bool>("use_tempdel_delegate")) {
    auto default_options = TfLiteTempdelDelegateOptionsDefault();
    return TfLiteTempdelDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
TempdelDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_tempdel_delegate"));
}
} // namespace tools
} // namespace tflite