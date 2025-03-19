#include <string>
#include <utility>

#include "add_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class AddDelegateProvider : public DelegateProvider {
public:
  AddDelegateProvider() {
    default_params_.AddParam("use_add_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "AddDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(AddDelegateProvider);

std::vector<Flag> AddDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_add_delegate", params, "use the add delegate.")};
  return flags;
}

void AddDelegateProvider::LogParams(const ToolParams &params,
                                     bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_add_delegate", "Use add test delegate",
                 verbose);
}

TfLiteDelegatePtr
AddDelegateProvider::CreateTfLiteDelegate(const ToolParams &params) const {
  if (params.Get<bool>("use_add_delegate")) {
    auto default_options = TfLiteAddDelegateOptionsDefault();
    return TfLiteAddDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
AddDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_add_delegate"));
}
} // namespace tools
} // namespace tflite