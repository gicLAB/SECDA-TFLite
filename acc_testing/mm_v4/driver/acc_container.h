#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#ifdef SYSC
#include "../acc.sc.h"
#include "sysc_profiler/profiler.h"
#include "systemc_binding.h"
#endif

#include "axi_support/axi_api_v2.h"

struct acc_container {

  // Hardware
#ifdef SYSC
  ACCNAME *acc;
  struct sysC_sigs *scs;
  Profile *profile;
#else
  int *acc;
#endif
  struct stream_dma *sdma;

  // Data
  int32_t *A;
  int32_t *B;
  int32_t *C;
  int M_size;
  int N_size;
  int K_size;
};

#endif // ACC_CONTAINER