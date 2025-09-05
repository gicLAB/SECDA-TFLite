
#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#ifdef SYSC
#include "../acc.sc.h"
#include "systemc_binding.h"
#else
#endif


// #include "../acc_config.sc.h"
// #include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
// #include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
// #include <vector>
// #include "tensorflow/lite/util.h"

#include "../acc_config.sc.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <typeinfo>
#include <unistd.h>
#include <vector>


using namespace std;
using namespace std::chrono;
// TODO: Already defined in axi_api_v2.h, should it be?
// #define TSCALE microseconds; // Timescale
// #define TSCAST duration_cast<nanoseconds>

#ifdef ACC_NEON
#include "arm_neon.h"
#else
#include "NEON_2_SSE.h"
#endif

struct acc_times {
  // timescale name file
  duration_ns fpga_total; // driver
  duration_ns cpu_total; // delegate
  duration_ns driver; // delegate
  duration_ns prep; // delegate

  void print() {
    // #ifdef ACC_PROFILE
    cout << "====================" << endl;
    prf_out(TSCALE, fpga_total);
    prf_out(TSCALE, cpu_total);
    prf_out(TSCALE, driver);
    prf_out(TSCALE, prep);
    cout << "====================" << endl;
    // #endif
  }

  void save_prf() {
    // #ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out(TSCALE, fpga_total, file);
    prf_file_out(TSCALE, cpu_total, file);
    prf_file_out(TSCALE, driver, file);
    prf_file_out(TSCALE, prep, file);
    file.close();
    // #endif
  }
};



struct acc_container {
    // Harware signals
    #ifdef SYSC
      ACCNAME *acc;
      struct sysC_sigs *scs;
    #else
      int *acc;
    #endif

    Profile *profile;
    struct multi_dma *mdma;

    // dimensions - Input(N,K) * Weights(K,M) (+ bias[N])
    int N;
    int K;
    int M;

    // Paded dimensions i.e. M,N and K rounded up to be a factor of 16
    int pN;
    int pM;
    int pK;

    // Data
    // Changed from int8_t *
    int8_t *padded_input;
    int8_t *padded_weights;
    int8_t *padded_output;

    // Summing accross columns or smth? can't remember
    int *bias;
    int *wt_sum;
    int *in_sum;

    // Not a clue
    int crf;
    int crx;
    int ra;
    int rhs_offset;
    int lhs_offset;

    // Running variable
    int start_count;

    // Debugging
    int layer = 0;
    // struct acc_times a_t; // TODO: This is different
    struct acc_times *a_t;


};

void store_unpad(int8_t *data, int width, int depth, int8_t *shape_data,
                 int w_pad, int d_pad) {
  // Jude's code to store unpadded
  int w = roundUp(width, w_pad);
  int d = roundUp(depth, d_pad);
  int i_c = 0;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < depth; j++) {
      int8_t val = data[(i * d) + j];
      shape_data[i_c++] = val;
    }
  }
}

void precal_sum_load_pad(int8_t *data, int width, int depth, int8_t *shape_data,
                         vector<int> &sums) {
  int w = ((width + 16 - 1) - ((width + 16 - 1) % 16));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  int dm = roundDown(depth, 16);
  int i_c = 0;
  for (int i = 0; i < w; i++) {
    int s0 = 0;
    if (i < width) {
      for (int j = 0; j < d; j++) {
        if (j < depth) {
          int8_t val = data[(i * depth) + j];
          s0 += val;
          shape_data[i_c++] = val;
        } else {
          shape_data[i_c++] = 0;
        }
      }

    } else {
      for (int j = 0; j < d; j++)
        shape_data[i_c++] = 0;
    }
    sums.push_back(s0);
  }
}

void precal_sum_load_padv3_vectorized(int8_t *data, int width, int depth,
                                      int8_t *shape_data,
                                      std::vector<int> &sums) {
  constexpr int w_pad = 16;
  constexpr int d_pad = 64;
  int w = roundUp(width, w_pad);
  int d = roundUp(depth, d_pad);
  int max = width * depth;
  int i_c = 0;
  for (int i = 0; i < w; i++) {
    int s0 = 0;
    if (i < width) {
      for (int j = 0; j < d; j += 16) {
        if (j < depth) {
          int8x16_t val;
          int8_t val_arr[16];
          val = vld1q_s8(data + (i * depth) + j);
          vst1q_s8(val_arr, val);
          for (int k = 0; k < 16; k++) {
            s0 += val_arr[k];
            shape_data[i_c++] = val_arr[k];
          }
        } else {
          for (int k = 0; k < 16; k++) {
            shape_data[i_c++] = 0;
          }
        }
      }
    } else {
      for (int j = 0; j < d; j++) {
        shape_data[i_c++] = 0;
      }
    }
    sums.push_back(s0);
  }
}


#endif