#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_TEMPDEL_DELEGATE_TEMPDEL_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_TEMPDEL_DELEGATE_TEMPDEL_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct {
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} TempdelDelegateOptions;

// Returns a structure with the default delegate options.
TempdelDelegateOptions TfLiteTempdelDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteTempdelDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteTempdelDelegateCreate(const TempdelDelegateOptions *options);

// Destroys a delegate created with `TfLiteTempdelDelegateCreate` call.
void TfLiteTempdelDelegateDelete(TfLiteDelegate *delegate);
#ifdef __cplusplus
}
#endif // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate *)>
TfLiteTempdelDelegateCreateUnique(const TempdelDelegateOptions *options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate *)>(
      TfLiteTempdelDelegateCreate(options), TfLiteTempdelDelegateDelete);
}

#endif // TENSORFLOW_LITE_DELEGATES_UTILS_TEMPDEL_DELEGATE_TEMPDEL_DELEGATE_H_