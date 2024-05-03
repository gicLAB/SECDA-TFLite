#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SA_DELEGATE_SA_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SA_DELEGATE_SA_DELEGATE_H_

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
} SADelegateOptions;

// Returns a structure with the default delegate options.
SADelegateOptions TfLiteSADelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteSADelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteSADelegateCreate(const SADelegateOptions* options);

// Destroys a delegate created with `TfLiteSADelegateCreate` call.
void TfLiteSADelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteSADelegateCreateUnique(const SADelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteSADelegateCreate(options), TfLiteSADelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SA_DELEGATE_SA_DELEGATE_H_
