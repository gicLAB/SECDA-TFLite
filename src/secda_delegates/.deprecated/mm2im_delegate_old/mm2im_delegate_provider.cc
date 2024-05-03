#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_delegates/mm2im_delegate/mm2im_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class MM2IMDelegateProvider : public DelegateProvider {
 public:
  MM2IMDelegateProvider() {
    default_params_.AddParam("use_mm2im_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "MM2IMDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(MM2IMDelegateProvider);

std::vector<Flag> MM2IMDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_mm2im_delegate", params,
                                              "use the mm2im delegate.")};
  return flags;
}

void MM2IMDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_mm2im_delegate", "Use mm2im test delegate",
                 verbose);
}

TfLiteDelegatePtr MM2IMDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_mm2im_delegate")) {
    auto default_options = TfLiteMM2IMDelegateOptionsDefault();
    return TfLiteMM2IMDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
MM2IMDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_mm2im_delegate"));
}
}  // namespace tools
}  // namespace tflite
