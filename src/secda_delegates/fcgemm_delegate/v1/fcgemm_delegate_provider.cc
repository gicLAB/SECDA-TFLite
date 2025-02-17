#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_delegates/fcgemm_delegate/v1/fcgemm_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class FCGEMMDelegateProvider : public DelegateProvider {
 public:
  FCGEMMDelegateProvider() {
    default_params_.AddParam("use_fcgemm_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "FCGEMMDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(FCGEMMDelegateProvider);

std::vector<Flag> FCGEMMDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_fcgemm_delegate", params,
                                              "use the fcgemm delegate.")};
  return flags;
}

void FCGEMMDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_fcgemm_delegate", "Use fcgemm test delegate",
                 verbose);
}

TfLiteDelegatePtr FCGEMMDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_fcgemm_delegate")) {
    auto default_options = TfLiteFCGEMMDelegateOptionsDefault();
    return TfLiteFCGEMMDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
FCGEMMDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_fcgemm_delegate"));
}
}  // namespace tools
}  // namespace tflite
