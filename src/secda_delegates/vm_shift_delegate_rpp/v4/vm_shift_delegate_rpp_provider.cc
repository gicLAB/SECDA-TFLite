#include <string>
#include <utility>

#include "vm_shift_delegate_rpp.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class VMDelegateProvider : public DelegateProvider {
public:
  VMDelegateProvider() {
    default_params_.AddParam("use_vm_shift_delegate_rpp",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("use_simmode", ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "VMDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(VMDelegateProvider);

std::vector<Flag> VMDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_vm_shift_delegate_rpp", params,
                       "use the vm delegate."),
      CreateFlag<bool>("use_simmode", params, "use_simmode for delegate.")};
  return flags;
}

void VMDelegateProvider::LogParams(const ToolParams &params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_vm_shift_delegate_rpp", "Use vm test delegate",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "use_simmode", "Use use_simmode for delegate",
                 verbose);
}

TfLiteDelegatePtr
VMDelegateProvider::CreateTfLiteDelegate(const ToolParams &params) const {
  if (params.Get<bool>("use_vm_shift_delegate_rpp")) {
    auto default_options = TfLiteVMDelegateOptionsDefault();
    default_options.use_simmode = params.Get<bool>("use_simmode");
    return TfLiteVMDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
VMDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_vm_shift_delegate_rpp"));
}
} // namespace tools
} // namespace tflite
