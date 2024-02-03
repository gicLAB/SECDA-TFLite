#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <cassert>
#include <iomanip>
#include <vector>

#ifdef SYSC
#include "../acc.sc.h"
#include "systemc_binding.h"
#else
#endif

#include "../acc_config.sc.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;
#define TSCALE microseconds
#define TSCAST duration_cast<nanoseconds>

struct TOYADD_ACC_times {
  duration_ns driver_total;
  duration_ns delegate_total;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    prf_out(TSCALE, driver_total);
    prf_out(TSCALE, delegate_total);
    cout << "================================================" << endl;
#endif
  }
  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out(TSCALE, driver_total, file);
    prf_file_out(TSCALE, delegate_total, file);
    file.close();
#endif
  }
};

struct layer_details {
  int layer = 0;
  bool profile = false;
};

struct acc_container {
// Hardware
#ifdef SYSC
  ACCNAME *acc;
  struct sysC_sigs *scs;
#else
  int *acc;
#endif

  struct multi_dma *mdma;
  Profile *profile;
  MultiThreadContext *mt_context;
  int thread_count;

  // Accelerator Specific Parameters
  // Data
  int length;
  const int8_t *input_A;
  const int8_t *input_B;
  int8_t *output_C;

  // PPU
  int lshift;
  int in1_off;
  int in1_sv;
  int in1_mul;
  int in2_off;
  int in2_sv;
  int in2_mul;
  int out1_off;
  int out1_sv;
  int out1_mul;
  int qa_max;
  int qa_min;

  // Debugging
  struct layer_details t;
  struct TOYADD_ACC_times p_t;
};

#endif // ACC_CONTAINER