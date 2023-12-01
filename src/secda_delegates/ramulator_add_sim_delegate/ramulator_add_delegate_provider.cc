#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_delegates/ramulator_add_sim_delegate/ramulator_add_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class Ramulator_addDelegateProvider : public DelegateProvider {
public:
  Ramulator_addDelegateProvider() {
    default_params_.AddParam("use_ramulator_add_sim_delegate",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("use_simmode", ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams *params) const final;

  void LogParams(const ToolParams &params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams &params) const final;
  std::pair<TfLiteDelegatePtr, int>
  CreateRankedTfLiteDelegate(const ToolParams &params) const final;

  std::string GetName() const final { return "Ramulator_addDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(Ramulator_addDelegateProvider);

std::vector<Flag>
Ramulator_addDelegateProvider::CreateFlags(ToolParams *params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_ramulator_add_sim_delegate", params,
                       "use the ramulator_add delegate."),
      CreateFlag<bool>("use_simmode", params, "use_simmode for delegate.")};
  return flags;
}

void Ramulator_addDelegateProvider::LogParams(const ToolParams &params,
                                              bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_ramulator_add_sim_delegate",
                 "Use ramulator_add test delegate", verbose);
  LOG_TOOL_PARAM(params, bool, "use_simmode",
                 "Use use_simmode for delegate", verbose);
}

TfLiteDelegatePtr Ramulator_addDelegateProvider::CreateTfLiteDelegate(
    const ToolParams &params) const {
  if (params.Get<bool>("use_ramulator_add_sim_delegate")) {
    auto default_options = TfLiteRamulator_addDelegateOptionsDefault();
    default_options.use_simmode = params.Get<bool>("use_simmode");


    return TfLiteRamulator_addDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate *) {});
}

std::pair<TfLiteDelegatePtr, int>
Ramulator_addDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams &params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<bool>(
                                            "use_ramulator_add_sim_delegate"));
}
} // namespace tools
} // namespace tflite