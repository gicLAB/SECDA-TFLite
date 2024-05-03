#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_MM2IM_DELEGATE_MM2IM_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_MM2IM_DELEGATE_MM2IM_DELEGATE_H_

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
} MM2IMDelegateOptions;

// Returns a structure with the default delegate options.
MM2IMDelegateOptions TfLiteMM2IMDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteMM2IMDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteMM2IMDelegateCreate(const MM2IMDelegateOptions* options);

// Destroys a delegate created with `TfLiteMM2IMDelegateCreate` call.
void TfLiteMM2IMDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteMM2IMDelegateCreateUnique(const MM2IMDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteMM2IMDelegateCreate(options), TfLiteMM2IMDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_MM2IM_DELEGATE_MM2IM_DELEGATE_H_
