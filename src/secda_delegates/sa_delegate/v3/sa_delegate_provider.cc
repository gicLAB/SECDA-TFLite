#include <string>
#include <utility>

#include "sa_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class SADelegateProvider : public DelegateProvider {
 public:
  SADelegateProvider() {
    default_params_.AddParam("use_sa_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "SADelegate"; }
};
REGISTER_DELEGATE_PROVIDER(SADelegateProvider);

std::vector<Flag> SADelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_sa_delegate", params,
                                              "use the sa delegate.")};
  return flags;
}

void SADelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_sa_delegate", "Use sa test delegate",
                 verbose);
}

TfLiteDelegatePtr SADelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_sa_delegate")) {
    auto default_options = TfLiteSADelegateOptionsDefault();
    return TfLiteSADelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
SADelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_sa_delegate"));
}
}  // namespace tools
}  // namespace tflite
