#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#ifdef SYSC
#include "../acc.sc.h"
#include "systemc_binding.h"
#include "sysc_profiler/profiler.h"
#endif

#include "axi_support/axi_api_v2.h"

struct acc_container {

#ifdef SYSC
  // Hardware
  ACCNAME *acc;
  struct sysC_sigs *scs;
  Profile *profile;
#else
  int *acc;
#endif
  struct stream_dma *sdma;
  
  // Data
  int length;
  int32_t *input_A;
  int32_t *input_B;
  int32_t *output_C;
};

#endif // ACC_CONTAINER