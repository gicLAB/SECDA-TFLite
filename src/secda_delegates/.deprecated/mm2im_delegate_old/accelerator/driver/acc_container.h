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

int nofSteps(int length, int stride, int ks) {
  return int(((length - ks) / stride) + 1);
}

int ceiling(int a, int b) { return (a + b - 1) / b; }

// Used for storing current GEMM info
struct gemm_details {
  int layer = 0;
  bool profile = false;
};

struct mm2im_times {
  duration_ns driver_total;
  duration_ns acc_total;
  duration_ns tconv_total;
  duration_ns store;
  duration_ns load_wgt;
  duration_ns load_inp;
  duration_ns load_rowmap;
  duration_ns load_colmap;
  duration_ns start_sched;
  duration_ns load_config;
  duration_ns handle_rest;
  duration_ns ipack;

  int inp_data_sent = 0;
  int wgt_data_sent = 0;
  int colmap_data_sent = 0;
  int inp_load_calls = 0;
  int wgt_load_calls = 0;
  int colmap_load_calls = 0;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    prf_out(TSCALE, driver_total);
    prf_out(TSCALE, acc_total);
    prf_out(TSCALE, tconv_total);
    prf_out(TSCALE, store);
    prf_out(TSCALE, load_wgt);
    prf_out(TSCALE, load_inp);
    prf_out(TSCALE, load_colmap);
    prf_out(TSCALE, start_sched);
    prf_out(TSCALE, load_config);
    prf_out(TSCALE, handle_rest);
    prf_out(TSCALE, ipack);
    cout << "================================================" << endl;
#endif
  }

  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out(TSCALE, driver_total, file);
    prf_file_out(TSCALE, acc_total, file);
    prf_file_out(TSCALE, tconv_total, file);
    prf_file_out(TSCALE, store, file);
    prf_file_out(TSCALE, load_wgt, file);
    prf_file_out(TSCALE, load_inp, file);
    prf_file_out(TSCALE, load_colmap, file);
    prf_file_out(TSCALE, start_sched, file);
    prf_file_out(TSCALE, load_config, file);
    prf_file_out(TSCALE, handle_rest, file);
    prf_file_out(TSCALE, ipack, file);
    file.close();
#endif
  }
};

struct acc_container {
#ifdef SYSC
  ACCNAME *acc;
#else
  int *acc;
#endif

  struct multi_dma *mdma;
  Profile *profile;
  MultiThreadContext *mt_context;
  int thread_count;

  // Padded Buffers
  int *loaded_weights;
  int *loaded_inputs;
  int8_t *output_data;

  int8_t *weights;
  const int8_t *inputs;

  // mm2im map
  // vector<vector<int>> *mm2im_map;
  int *oh_ends;


  // // mm2im map
  // vector<vector<int>> *mm2im_map;
  // vector<vector<vector<int>>> *oh_map;
  // int *oh_lengths;
  // int *oh_starts;
  // int *oh_ends;
  // // vector<vector<int>> col_dexs;
  // // vector<vector<int>> out_dexs;
  // vector<vector<int>> *col_dexs;
  // vector<vector<int>> *out_dexs;

  // Output Pipeline Metadata
  int32_t *acc_wt_sum;
  int *crf;
  int8_t *crx;
  int *bias;

  // External Params
  int ra;
  int rhs_offset = 0;
  int lhs_offset = 0;
  int ih = 0;
  int iw = 0;
  int ic = 0;
  int f = 0;
  int ks = 0;
  int oh = 0;
  int ow = 0;
  int oc = 0;
  int sx = 0;
  int sy = 0;
  int pt = 0;
  int pl = 0;
  int pb = 0;
  int pr = 0;
  int width_col = 0;
  int rows = 0;
  int cols = 0;
  int depth = 0;

  // GEMM Info variable
  struct gemm_details t;
  struct mm2im_times p_t;
  acc_container() {}

  void validate() {
    int padded_depth = roundUp(depth, 16);
    int padded_out_width = ow + pl + pr;
    int noOfStepsX = nofSteps(padded_out_width, sx, ks);
    int max_input_rows_per_oh = noOfStepsX * ceiling(ks, sy);

    int PE_COUNT_val = oc;
    int PE_WGTCOLBUF_SIZE_val = ks * ks * padded_depth / UF;
    int PE_WGTCOLSUMBUF_SIZE_val = ks * ks;
    int PE_INPROWBUF_SIZE_val = padded_depth / UF;
    int PE_OUTBUF_SIZE_val = ks * ks * max_input_rows_per_oh;
    int PE_POUTDEXBUF_SIZE_val = ks * ks;
    int PE_ACC_BUF_SIZE_val = oh * ow;
    assert(PE_COUNT_val >= PE_COUNT);
    assert(PE_WGTCOLBUF_SIZE_val <= PE_WGTCOLBUF_SIZE);
    assert(PE_WGTCOLSUMBUF_SIZE_val <= PE_WGTCOLSUMBUF_SIZE);
    assert(PE_INPROWBUF_SIZE_val <= PE_INPROWBUF_SIZE);
    assert(PE_OUTBUF_SIZE_val <= PE_OUTBUF_SIZE);
    assert(PE_POUTDEXBUF_SIZE_val <= PE_POUTDEXBUF_SIZE);
    assert(PE_ACC_BUF_SIZE_val <= PE_ACC_BUF_SIZE);
    // cerr << "=====================" << endl;
    // cerr << "PE_COUNT_val: " << PE_COUNT_val << endl;
    // cerr << "PE_WGTCOLBUF_SIZE_val: " << PE_WGTCOLBUF_SIZE_val << endl;
    // cerr << "PE_WGTCOLSUMBUF_SIZE_val: " << PE_WGTCOLSUMBUF_SIZE_val << endl;
    // cerr << "PE_INPROWBUF_SIZE_val: " << PE_INPROWBUF_SIZE_val << endl;
    // cerr << "PE_OUTBUF_SIZE_val: " << PE_OUTBUF_SIZE_val << endl;
    // cerr << "PE_POUTDEXBUF_SIZE_val: " << PE_POUTDEXBUF_SIZE_val << endl;
    // cerr << "PE_ACC_BUF_SIZE_val: " << PE_ACC_BUF_SIZE_val << endl;
    // cerr << "=====================" << endl;
  }
};

void preload_weights(int8_t *wgt, int depth, int rows, int *wt_sum,
                     int8_t *loaded_weights) {
  int d = roundUp(depth, 16);
  int max = rows * depth;
  int wt_sum_dex = 0;
  for (int i = 0; i < rows; i++) {
    int s0 = 0;
    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 = wgt[i * depth + j];
        loaded_weights[i * d + j] = w0;
        s0 += w0;
      } else loaded_weights[i * d + j] = 0;
    }
    wt_sum[wt_sum_dex++] = s0;
  }
}

void preload_inputs(const int8_t *inp, int depth, int rows,
                    int8_t *loaded_inputs) {
  int d = roundUp(depth, 16);
  int max = rows * depth;
  int wt_sum_dex = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 = inp[i * depth + j];
        loaded_inputs[i * d + j] = w0;
      } else loaded_inputs[i * d + j] = 0;
    }
  }
}
void swap_weights(const int8_t *wgt, int8_t *new_wgt, int filters, int ks,
                  int ic) {
  for (int k = 0; k < ks * ks; k++) {
    for (int j = 0; j < filters; j++) {
      for (int i = 0; i < ic; i++) {
        new_wgt[j * ks * ks * ic + k * ic + i] =
            wgt[k * filters * ic + j * ic + i];
      }
    }
  }
}

#endif // ACC_CONTAINER