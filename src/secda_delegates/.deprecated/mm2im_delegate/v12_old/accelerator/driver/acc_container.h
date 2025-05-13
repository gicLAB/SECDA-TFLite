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
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "secda_tools/secda_utils/acc_helpers.h"
#include "secda_tools/secda_utils/multi_threading.h"
#include "secda_tools/secda_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

#define TOG(X)

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

struct preloader {
  int *preload_buf;
  unsigned int phys_offsets[500];
  int preload_buf_sizes[500];
  int cur_idx = 0;

  int layer = 0;
  int alloced_size = 0;
  int loaded_tiles = 0;
  int current_tile = 0;
  int current_offset = 0;
  bool allocation_done = false;

  void init(int *preload_buf) { this->preload_buf = preload_buf; }

  void dma_preload(int *loaded_weights, int ks, int oc, int padded_depth,
                   int rhs_offset, int *acc_wt_sum, int *crf, double *crx_scale,
                   int8_t *crx, int *bias) {
    int data_per_filter = ks * ks * padded_depth;
    int cols_per_filter = ks * ks;
    int acc_weight_cols_sup = PE_WGTCOLBUF_SIZE * UF * PE_COUNT;
    int filter_step = min(acc_weight_cols_sup / data_per_filter, PE_COUNT);
    assert(filter_step == PE_COUNT);
    int size_per_input_row = padded_depth;
    assert((size_per_input_row / UF) <= PE_INPROWBUF_SIZE);
    int remaining_filters = oc % filter_step;
    int acc_filters = oc - remaining_filters;
    int o_3 = 0;
    int wgt_size = 0;
    for (; o_3 < oc; o_3 += filter_step) {
      wgt_size += 4;
      int padded_depth_4 = padded_depth / 4;
      int fs_rem = min(filter_step, oc - o_3);
      int number_of_rows = fs_rem * cols_per_filter;
      for (int i = 0; i < number_of_rows; i++) {
        wgt_size += padded_depth_4 + 1;
      }
      wgt_size += filter_step;
      wgt_size += filter_step;
      wgt_size += roundUp(filter_step, 4) / 4;
    }
    if (alloced_size + wgt_size > DMA_WGT_SIZE) {
      TOG(cerr << "Preload buffer size exceeded" << endl;);
      allocation_done = true;
      return;
    }

    o_3 = 0;
    for (; o_3 < oc; o_3 += filter_step) {
      int fs_rem = min(filter_step, oc - o_3);
      int starting_row = o_3 * cols_per_filter;
      int number_of_rows = fs_rem * cols_per_filter;
      int filter_step = fs_rem;
      int starting_filter = o_3;

      int *in0 = &preload_buf[current_offset + (DMA_WGT_OFFSET / 4)];
      int inl0 = 0;
      int padded_depth_4 = padded_depth / 4;
      int opcode = 2;
      int wgt_packet_a = number_of_rows;
      int wgt_packet_b = padded_depth_4 / (UF / 4);

      in0[inl0++] = opcode;
      in0[inl0++] = wgt_packet_a;
      in0[inl0++] = wgt_packet_b;
      in0[inl0++] = filter_step;
      for (int i = 0; i < number_of_rows; i++) {
        int src_addr = (starting_row + i) * padded_depth_4;
        memcpy(&in0[inl0], &loaded_weights[src_addr], padded_depth_4 * 4);
        inl0 += padded_depth_4;
        in0[inl0++] = acc_wt_sum[starting_row + i] * rhs_offset;
      }
      memcpy(&in0[inl0], &bias[starting_filter], filter_step * 4);
      inl0 += filter_step;
      memcpy(&in0[inl0], &crf[starting_filter], filter_step * 4);
      inl0 += filter_step;

      memcpy(&in0[inl0], &crx_scale[starting_filter], filter_step * 8);
      int* test = reinterpret_cast<int*>(&crx_scale[starting_filter]);
      inl0 += filter_step*2;

      memcpy(&in0[inl0], &crx[starting_filter], filter_step * 4);
      inl0 += roundUp(filter_step, 4) / 4;
      preload_buf_sizes[cur_idx] = inl0;
      phys_offsets[cur_idx] = (current_offset * 4) + (DMA_WGT_OFFSET);
      cur_idx++;
      loaded_tiles++;
      current_offset += inl0;
      alloced_size += inl0;
    }
    cerr << "Allocated tiles: " << loaded_tiles << " for layer " << layer
         << endl;
    layer++;
  }

  bool load_weights(s_mdma *mdma) {
    if (allocation_done && current_tile < loaded_tiles) {
      int pyhsical_offset = phys_offsets[current_tile];
      int inl0 = preload_buf_sizes[current_tile];
      mdma->dmas[0].dma_change_start(pyhsical_offset);
      mdma->dmas[0].dma_start_send(inl0);
      mdma->multi_dma_wait_send();
      TOG(cerr << "Loaded tile: " << current_tile << endl;);
      current_tile++;
      mdma->dmas[0].dma_change_start(0);
      return true;
    }
    return false;
  }

  void reset_loading() { current_tile = 0; }
};

struct mm2im_times {
  duration_ns t_tconv;
  duration_ns p_store;
  duration_ns p_load_wgt;
  duration_ns p_load_inp;
  duration_ns p_start_sched;
  duration_ns p_load_config;
  duration_ns p_ipack;

  int call_count = 0;

  int inp_data_sent = 0;
  int wgt_data_sent = 0;
  int colmap_data_sent = 0;
  int inp_load_calls = 0;
  int wgt_load_calls = 0;
  int colmap_load_calls = 0;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    prf_out_n(TSCALE, t_tconv, call_count);
    prf_out_n(TSCALE, p_store, call_count);
    prf_out_n(TSCALE, p_load_wgt, call_count);
    prf_out_n(TSCALE, p_load_inp, call_count);
    prf_out_n(TSCALE, p_start_sched, call_count);
    prf_out_n(TSCALE, p_load_config, call_count);
    prf_out_n(TSCALE, p_ipack, call_count);
    cout << "================================================" << endl;
#endif
  }

  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out_n(TSCALE, p_ipack, file, call_count);
    prf_file_out_n(TSCALE, p_load_config, file, call_count);
    prf_file_out_n(TSCALE, p_load_wgt, file, call_count);
    prf_file_out_n(TSCALE, p_load_inp, file, call_count);
    prf_file_out_n(TSCALE, p_start_sched, file, call_count);
    prf_file_out_n(TSCALE, p_store, file, call_count);
    prf_file_out_n(TSCALE, t_tconv, file, call_count);
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

  struct s_mdma *mdma;
  Profile *profile;
  MultiThreadContext *mt_context;
  int thread_count;
  struct preloader *preloader;

  // Padded Buffers
  int *loaded_weights;
  int *loaded_inputs;
  int8_t *output_data;

  int8_t *weights;
  const int8_t *inputs;
  int *oh_ends;

  // Output Pipeline Metadata
  int32_t *acc_wt_sum;
  int *crf;
  int8_t *crx;
  double *crx_scale;
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

  bool input_preloaded = false;
  bool weight_preloaded = false;

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
  int padded_depth = roundUp(depth, 16);
  int max = rows * depth;
  for (int i = 0; i < rows; i++) {
    memcpy(&loaded_inputs[i * padded_depth], &inp[i * depth], depth);
    memset(&loaded_inputs[i * padded_depth + depth], 0, padded_depth - depth);
  }
}

bool preload_inputs_dma(const int8_t *inp, int depth, int rows,
                        int32_t *dma_buf) {
  int padded_depth = roundUp(depth, 16);
  int max = rows * depth;
  int *dma_inp_buf = &dma_buf[DMA_INS_SIZE / 4];
  if (DMA_INP_SIZE < rows * padded_depth) {
    cerr << "Input buffer size exceeded" << endl;
    return false;
  }
  int8_t *inp_buf = reinterpret_cast<int8_t *>(dma_inp_buf);
  for (int i = 0; i < rows; i++) {
    memcpy(&inp_buf[i * padded_depth], &inp[i * depth], depth);
    memset(&inp_buf[i * padded_depth + depth], 0, padded_depth - depth);
  }
  return true;
}

void swap_weights(const int8_t *wgt, int8_t *new_wgt, int filters, int ks,
                  int ic) {
  for (int k = 0; k < ks * ks; k++) {
    for (int j = 0; j < filters; j++) {
      for (int i = 0; i < ic; i++) {
        new_wgt[(j * ks * ks * ic) + (k * ic) + (i)] =
            wgt[(k * filters * ic) + (j * ic) + (i)];
      }
    }
  }
}

#endif // ACC_CONTAINER