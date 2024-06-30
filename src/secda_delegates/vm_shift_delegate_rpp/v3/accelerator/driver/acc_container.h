#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#ifdef SYSC
#include "../acc.sc.h"
#include "systemc_binding.h"
#else
#endif

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

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;
#define TSCALE microseconds

struct vm_times {
  duration_ns load_send_inputs;
  duration_ns load_weights;
  duration_ns send_weights;
  duration_ns set_results;
  duration_ns start_compute;
  duration_ns receive_results;
  duration_ns vm_acc;
  duration_ns store;
  duration_ns ipack;
  duration_ns conv_total;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    prf_out(TSCALE, load_send_inputs);
    prf_out(TSCALE, load_weights);
    prf_out(TSCALE, send_weights);
    prf_out(TSCALE, set_results);
    prf_out(TSCALE, start_compute);
    prf_out(TSCALE, receive_results);    
    prf_out(TSCALE, store);
    prf_out(TSCALE, vm_acc);
    prf_out(TSCALE, ipack);
    prf_out(TSCALE, conv_total);
    cout << "================================================" << endl;
#endif
  }

  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out(TSCALE, load_send_inputs, file);
    prf_file_out(TSCALE, load_weights, file);
    prf_file_out(TSCALE, send_weights, file);
    prf_file_out(TSCALE, set_results, file);
    prf_file_out(TSCALE, start_compute, file);
    prf_file_out(TSCALE, receive_results, file);
    prf_file_out(TSCALE, store, file);
    prf_file_out(TSCALE, vm_acc, file);
    prf_file_out(TSCALE, ipack, file);
    prf_file_out(TSCALE, conv_total, file);
    file.close();
#endif
  }
};

// Used for profiling
struct layer_details {
  int layer = 0;
  int layer_weight_tile = 0;
  int layer_input_tile = 0;
  int layer_print = -1;
  int layer_ww = -1;
  int layer_iw = -1;
  bool profile = false;
};

// Used for tracking output locations
struct store_params {
  int *dst;
  int dcs;
  int rows;
  int cols;
  int rcols;
  int rrows;
};

struct acc_container {
#ifdef SYSC
  // Gives SystemC accelerator access
  ACCNAME *acc;
#else
  // Gives accelerator access
  int *acc;
#endif

  Profile *profile;
  // DMAs Pointer
  struct multi_dma *mdma;

  // Temporary Weight non-MMapped Padded Buffers
  int *wb_0;
  int *wb_1;
  int *wb_2;
  int *wb_3;

  // Temporary Input non-MMapped Padded Buffers
  int *inb_0;
  int *inb_1;
  int *inb_2;
  int *inb_3;
  int in_id = 0;

  // Driver variables
  struct store_params *st_params;
  MultiThreadContext *mt_context;
  int thread_count;
  int w_c = 0;

  // Output Pipeline Metadata
  vector<int> wt_sum1;
  vector<int> wt_sum2;
  vector<int> wt_sum3;
  vector<int> wt_sum4;
  int *in_sum1;
  int *in_sum2;
  int *in_sum3;
  int *in_sum4;
  int *bias;
  vector<int> crf;
  vector<int8_t> crx;
  int ra;
  int inp_offset = 0;
  int wgt_offset = 0;

  int rows = 0;
  int cols = 0;
  int depth = 0;
  int8_t *dst;

  // Pipeline vars
  struct dma_buffer_set *dfs;
  struct DSR *dsr;
  bool wgt_start = false;
  int recv_len;

  // GEMM Info variable
  struct layer_details t;
  struct vm_times t2;
  bool use_sim = false;

  // void clear_traces() {
  //   if (!use_sim) {
  //     ofstream file;
  //     std::string filename =
  //         ".data/secda_pim/traces/" + std::to_string(t.layer) + ".trace";
  //     file.open(filename, std::ios_base::trunc);
  //     file.close();
  //   }
  // }

  // // TODO generate read per byte
  // template <typename T>
  // void massign(T *dst, T *src, int d_dex, int s_dex, T value) {
  //   if (!use_sim) {
  //     cout << "dst_addr: " << (void *)&dst[d_dex] << endl;
  //     auto dst_addr = (void *)(&dst[d_dex]);
  //     auto src_addr = (void *)(&src[s_dex]);
  //     // truncate hex address to 32 bits
  //     dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
  //     src_addr = (void *)((uint64_t)src_addr & 0xffffffff);
  //     // create memory trace for ramulator and write to file
  //     ofstream file;
  //     std::string filename =
  //         ".data/secda_pim/traces/" + std::to_string(t.layer) + ".trace";
  //     file.open(filename, std::ios_base::app);
  //     file << std::hex << (src_addr++) << " R" << endl;
  //     file << std::hex << (dst_addr++) << " W" << endl;
  //     file.close();
  //   }
  //   // dst[d_dex] = value;
  // }

  // template <typename D, typename S>
  // void massign(D *dst, S *src, int d_dex, int s_dex) {
  //   auto dst_addr = (void *)(&dst[d_dex]);
  //   auto src_addr = (void *)(&src[s_dex]);
  //   // truncate hex address to 32 bits
  //   dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
  //   src_addr = (void *)((uint64_t)src_addr & 0xffffffff);
  //   // create memory trace for ramulator and write to file
  //   ofstream file;
  //   std::string filename =
  //       ".data/secda_pim/traces/" + std::to_string(t.layer) + ".trace";
  //   file.open(filename, std::ios_base::app);
  //   // file << (uint64_t) (src_addr++) << " R" << endl;
  //   // file << (uint64_t) (dst_addr++) << " W" << endl;

  //   file << " 3 " << (uint64_t)(src_addr++) << " " << (uint64_t)(dst_addr++)
  //        << endl;

  //   // file << std::hex << (src_addr++) << " R" << endl;
  //   // file << std::hex << (dst_addr++) << " W" << endl;
  //   file.close();
  // }

  // template <typename D, typename S>
  // void massign(D *dst, S *src, int d_dex, int s_dex, int cpu_ins) {
  //   auto dst_addr = (void *)(&dst[d_dex]);
  //   auto src_addr = (void *)(&src[s_dex]);
  //   // truncate hex address to 32 bits
  //   dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
  //   src_addr = (void *)((uint64_t)src_addr & 0xffffffff);
  //   // create memory trace for ramulator and write to file
  //   ofstream file;
  //   std::string filename =
  //       ".data/secda_pim/traces/" + std::to_string(t.layer) + ".trace";
  //   file.open(filename, std::ios_base::app);
  //   // file << (uint64_t) (src_addr++) << " R" << endl;
  //   // file << (uint64_t) (dst_addr++) << " W" << endl;
  //   uint64_t csrc = (uint64_t)(src_addr++);
  //   uint64_t cdst = (uint64_t)(dst_addr++);
  //   // modulo the address to 512 megabytes
  //   csrc = csrc % (512 * 1024 * 1024);
  //   cdst = cdst % (512 * 1024 * 1024);

  //   file << cpu_ins << " " << (uint64_t)(src_addr++) << " "
  //        << (uint64_t)(dst_addr++) << endl;

  //   // file << std::hex << (src_addr++) << " R" << endl;
  //   // file << std::hex << (dst_addr++) << " W" << endl;
  //   file.close();
  // }

  // template <typename T>
  // void massign(T *dst, int d_dex) {
  //   auto dst_addr = (void *)(&dst[d_dex]);
  //   // truncate hex address to 32 bits
  //   dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
  //   // create memory trace for ramulator and write to file
  //   ofstream file;
  //   std::string filename =
  //       ".data/secda_pim/traces/" + std::to_string(t.layer) + ".trace";
  //   file.open(filename, std::ios_base::app);
  //   file << std::hex << (dst_addr++) << " W" << endl;
  //   file.close();
  // }

  // void load_inject_dram_cycles() {
  //   if (use_sim) {
  //     // load latency from ramulator
  //     fstream file;
  //     std::string filename =
  //         ".data/secda_pim/layers/" + std::to_string(t.layer) + ".csv";
  //     file.open(filename, ios::in);

  //     // read header
  //     vector<string> row;
  //     std::string line, word, temp;
  //     int dram_cycles = 0;
  //     int mhz = 0;
  //     int cpu_cycles = 0;
  //     getline(file, line);
  //     istringstream s(line);
  //     char delim = ',';
  //     while (getline(s, word, delim)) {
  //       row.push_back(word);
  //     }
  //     dram_cycles = std::stoi(row[0]);
  //     mhz = std::stoi(row[1]);
  //     cpu_cycles = std::stoi(row[2]);

  //     // sc_start(dram_cycles * 1.85, SC_NS);
  //     sc_start(cpu_cycles * 1.54, SC_NS);
  //   }
  // }

  acc_container(int *_wb_0, int *_wb_1, int *_wb_2, int *_wb_3,
                std::vector<int> _wt_sum1, std::vector<int> _wt_sum2,
                std::vector<int> _wt_sum3, std::vector<int> _wt_sum4,
                std::vector<int> _crf, std::vector<int8_t> _crx) {
    wb_0 = _wb_0;
    wb_1 = _wb_1;
    wb_2 = _wb_2;
    wb_3 = _wb_3;
    wt_sum1 = _wt_sum1;
    wt_sum2 = _wt_sum2;
    wt_sum3 = _wt_sum3;
    wt_sum4 = _wt_sum4;
    crf = _crf;
    crx = _crx;
  }

  bool Check_Done() { return (mdma->multi_dma_check_recv() == 0); }

  void End_Transfer() { mdma->multi_dma_wait_send(); }

  bool Start_Transfer() {
    // if (!(dsr.sID == dsr.cID && dsr.dID > dsr.sID)) return false;
    // int s_buf = find_dbuf(dfs[0], dsr.sID);
    int s_buf = wait_for_dbuf(dfs[0], dsr->sID);
    mdma->multi_dma_change_start_4(dfs[0].dbuf_set[s_buf].offset);
    mdma->dmas[0].dma_start_send(dfs[0].dbuf_set[s_buf].len);
    mdma->dmas[1].dma_start_send(dfs[1].dbuf_set[s_buf].len);
    mdma->dmas[2].dma_start_send(dfs[2].dbuf_set[s_buf].len);
    mdma->dmas[3].dma_start_send(dfs[3].dbuf_set[s_buf].len);
    End_Transfer();
    // dsr.sID++;
    dsr->sID++;
    return true;
  }

  void Set_Results() {
    // int s_buf = find_dbuf(dfs[0], dsr.cID);
    int s_buf = wait_for_dbuf(dfs[0], dsr->cID);
    mdma->multi_dma_change_end(dfs[0].dbuf_set[s_buf].offset);
    mdma->multi_dma_start_recv(recv_len);
    // dsr.cID++;
    dsr->cID++;
  }

  void Recieve_Results() { mdma->multi_dma_wait_recv_4(); }
};

//========================//========================//========================//

void preload_weights(int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4, int inpZeroPoint, int* bias) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        int8_t weights[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
      } else {
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
      }
    }

    wt_sum1.push_back((s0*inpZeroPoint)+bias[(i*4)+0]);
    wt_sum2.push_back((s1*inpZeroPoint)+bias[(i*4)+1]);
    wt_sum3.push_back((s2*inpZeroPoint)+bias[(i*4)+2]);
    wt_sum4.push_back((s3*inpZeroPoint)+bias[(i*4)+3]);

    // wt_sum1.push_back(s0);
    // wt_sum2.push_back(s1);
    // wt_sum3.push_back(s2);
    // wt_sum4.push_back(s3);
  }
}


void precal_sum_load_pad(const int8_t *data, int width, int depth, int8_t *inb0,
                         int8_t *inb1, int8_t *inb2, int8_t *inb3, int *in_sum1,
                         int *in_sum2, int *in_sum3, int *in_sum4) {
  int w = ((width + 3) - ((width + 3) % 4));
  int d = ((depth + 15) - ((depth + 15) % 16));
  int d2 = depth * 2;
  int d3 = depth * 3;
  int d4 = depth * 4;
  int i_c = 0;
  int sums_curr = 0;

  const int8_t *inp_d = reinterpret_cast<const int8_t *>(data);
  int dm = 0;
  for (int i = 0; i < w / 4; i++) {
    int id = i * d4;
    int i0 = id;
    int i1 = id + depth;
    int i2 = id + d2;
    int i3 = id + d3;
    int ss0 = 0;
    int ss1 = 0;
    int ss2 = 0;
    int ss3 = 0;

#ifdef ACC_NEON
    dm = d - 16;
    int8x16_t tmp0;
    int8x16_t tmp1;
    int8x16_t tmp2;
    int8x16_t tmp3;

    int32x4_t tmp0_2;
    int32x4_t tmp1_2;
    int32x4_t tmp2_2;
    int32x4_t tmp3_2;

    int32x2_t tmp0_3;
    int32x2_t tmp1_3;
    int32x2_t tmp2_3;
    int32x2_t tmp3_3;
    int32x2_t tmp0_4 = vdup_n_s32(0);
    int32x2_t tmp1_4 = vdup_n_s32(0);
    int32x2_t tmp2_4 = vdup_n_s32(0);
    int32x2_t tmp3_4 = vdup_n_s32(0);

    for (int j = 0; j < dm; j += 16) {
      tmp0 = vld1q_s8(inp_d + i0 + j);
      tmp1 = vld1q_s8(inp_d + i1 + j);
      tmp2 = vld1q_s8(inp_d + i2 + j);
      tmp3 = vld1q_s8(inp_d + i3 + j);
#if 0
      tmp0_2 = vpaddlq_s16(vpaddlq_s8(tmp0));
      tmp1_2 = vpaddlq_s16(vpaddlq_s8(tmp1));
      tmp2_2 = vpaddlq_s16(vpaddlq_s8(tmp2));
      tmp3_2 = vpaddlq_s16(vpaddlq_s8(tmp3));

      tmp0_3 = vadd_s32(vget_high_s32(tmp0_2), vget_low_s32(tmp0_2));
      tmp1_3 = vadd_s32(vget_high_s32(tmp1_2), vget_low_s32(tmp1_2));
      tmp2_3 = vadd_s32(vget_high_s32(tmp2_2), vget_low_s32(tmp2_2));
      tmp3_3 = vadd_s32(vget_high_s32(tmp3_2), vget_low_s32(tmp3_2));
      tmp0_4 = vadd_s32(tmp0_4, tmp0_3);
      tmp1_4 = vadd_s32(tmp1_4, tmp1_3);
      tmp2_4 = vadd_s32(tmp2_4, tmp2_3);
      tmp3_4 = vadd_s32(tmp3_4, tmp3_3);
#endif
      vst1q_s8(inb0 + i_c, tmp0);
      vst1q_s8(inb1 + i_c, tmp1);
      vst1q_s8(inb2 + i_c, tmp2);
      vst1q_s8(inb3 + i_c, tmp3);
      i_c += 16;
    }
    int32_t tmp0_s[2];
    int32_t tmp1_s[2];
    int32_t tmp2_s[2];
    int32_t tmp3_s[2];
    vst1_s32(tmp0_s, tmp0_4);
    vst1_s32(tmp1_s, tmp1_4);
    vst1_s32(tmp2_s, tmp2_4);
    vst1_s32(tmp3_s, tmp3_4);
    ss0 += tmp0_s[0] + tmp0_s[1];
    ss1 += tmp1_s[0] + tmp1_s[1];
    ss2 += tmp2_s[0] + tmp2_s[1];
    ss3 += tmp3_s[0] + tmp3_s[1];
#endif
    for (int j = dm; j < d; j++) {
      if (j < depth) {
        unsigned char w0 = data[i0 + j];
        unsigned char w1 = data[i1 + j];
        unsigned char w2 = data[i2 + j];
        unsigned char w3 = data[i3 + j];
        // ss0 += w0;
        // ss1 += w1;
        // ss2 += w2;
        // ss3 += w3;
        inb0[i_c] = w0;
        inb1[i_c] = w1;
        inb2[i_c] = w2;
        inb3[i_c++] = w3;
      } else {
        inb0[i_c] = 0;
        inb1[i_c] = 0;
        inb2[i_c] = 0;
        inb3[i_c++] = 0;
      }
    }
    // in_sum1[sums_curr] = (ss0);
    // in_sum2[sums_curr] = (ss1);
    // in_sum3[sums_curr] = (ss2);
    // in_sum4[sums_curr++] = (ss3);
  }
}

#if 0
void preload_weights(const int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;

  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        int8_t weights[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
      } else {
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
      }
    }
    wt_sum1.push_back(s0);
    wt_sum2.push_back(s1);
    wt_sum3.push_back(s2);
    wt_sum4.push_back(s3);
  }
}

#endif


#endif // ACC_CONTAINER