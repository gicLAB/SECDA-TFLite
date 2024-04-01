#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_RAMULATOR_ADD_DELEGATE_RAMULATOR_ADD_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_RAMULATOR_ADD_DELEGATE_RAMULATOR_ADD_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
  bool use_simmode;
} Ramulator_addDelegateOptions;

// Returns a structure with the default delegate options.
Ramulator_addDelegateOptions TfLiteRamulator_addDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteRamulator_addDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteRamulator_addDelegateCreate(const Ramulator_addDelegateOptions* options);

// Destroys a delegate created with `TfLiteRamulator_addDelegateCreate` call.
void TfLiteRamulator_addDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteRamulator_addDelegateCreateUnique(const Ramulator_addDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteRamulator_addDelegateCreate(options), TfLiteRamulator_addDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_RAMULATOR_ADD_DELEGATE_RAMULATOR_ADD_DELEGATE_H_