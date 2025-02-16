#ifndef GEMM_MT
#define GEMM_MT

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include <cstring>
#include <mutex>

#define TOG(X)
// #define TOG(X) X

// #define TOG2(X) threadsafe_cout(X)
#define TOG2(X)

#define STR(X) std::to_string(X)
#define HEX(X) std::hex << X << std::dec

namespace tflite_vm {
using namespace std;

void threadsafe_cout(std::string log_msg) {
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  cout << std::move(log_msg);
}

void Load_Weight_Data(acc_container &drv, int free_buf, int8_t *results,
                      int output_stride, int c, int rcols_step, int r,
                      int rrows_step, int rdepth_step, int rows_step,
                      int cols_step) {
  TOG2("Load_Weight_Data" << endl);
  prf_start(1);
  int offset = drv.dfs[0].dbuf_set[free_buf].offset;
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer() + (offset / 4);
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer() + (offset / 4);
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer() + (offset / 4);
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer() + (offset / 4);

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int w_dex = (drv.w_c / 4);
  int data_length = rdepth_step * rcols_step;
  int wt_sums_len = rcols_step / 4;

  in0[inl0++] = OPCODE_LOAD_WGT;
  in0[inl0++] = (rcols_step * rdepth_step / 16); // wgt_size
  in0[inl0++] = wt_sums_len;                     // wgt_sum_size

#ifndef ACC_NEON
  for (int i = 0; i < data_length / 16; i++) {
    in0[inl0++] = drv.wb_0[w_dex + i];
    in1[inl1++] = drv.wb_1[w_dex + i];
    in2[inl2++] = drv.wb_2[w_dex + i];
    in3[inl3++] = drv.wb_3[w_dex + i];
  }
#else
  for (int i = 0; i < data_length / 16; i += 4) {
    vst1q_s32(in0 + inl0, vld1q_s32(drv.wb_0 + w_dex + i));
    vst1q_s32(in1 + inl1, vld1q_s32(drv.wb_1 + w_dex + i));
    vst1q_s32(in2 + inl2, vld1q_s32(drv.wb_2 + w_dex + i));
    vst1q_s32(in3 + inl3, vld1q_s32(drv.wb_3 + w_dex + i));
    inl0 += 4;
    inl1 += 4;
    inl2 += 4;
    inl3 += 4;
  }
#endif

  int b_c = c;
  int crf_c = c;
  int crx_c = c;
  int start_dex = (c / 4);
  int *wsums1 = reinterpret_cast<int *>(&drv.wt_sum1[start_dex]);
  int *wsums2 = reinterpret_cast<int *>(&drv.wt_sum2[start_dex]);
  int *wsums3 = reinterpret_cast<int *>(&drv.wt_sum3[start_dex]);
  int *wsums4 = reinterpret_cast<int *>(&drv.wt_sum4[start_dex]);

  for (int i = 0; i < wt_sums_len; i++) {
    in0[inl0++] = (wsums1[i] * drv.inp_offset) + drv.bias[b_c++];
    in1[inl1++] = (wsums2[i] * drv.inp_offset) + drv.bias[b_c++];
    in2[inl2++] = (wsums3[i] * drv.inp_offset) + drv.bias[b_c++];
    in3[inl3++] = (wsums4[i] * drv.inp_offset) + drv.bias[b_c++];
    in0[inl0++] = drv.crf[crf_c++];
    in1[inl1++] = drv.crf[crf_c++];
    in2[inl2++] = drv.crf[crf_c++];
    in3[inl3++] = drv.crf[crf_c++];
    int8_t w0 = drv.crx[crx_c++];
    int8_t w1 = drv.crx[crx_c++];
    int8_t w2 = drv.crx[crx_c++];
    int8_t w3 = drv.crx[crx_c++];
    int8_t ex[] = {w0, w1, w2, w3};
    in0[inl0++] = *(int *)(ex);
  }
  drv.w_c += data_length / 4;

  int8_t *res_pointer = results + c + r * output_stride;
  drv.st_params[free_buf].dst = reinterpret_cast<int *>(res_pointer);
  drv.st_params[free_buf].dcs = output_stride;
  drv.st_params[free_buf].cols = rcols_step;
  drv.st_params[free_buf].rows = rrows_step;
  drv.st_params[free_buf].rrows = rows_step;
  drv.st_params[free_buf].rcols = cols_step;
  // TOG2("offset: " + STR((unsigned long long)drv.st_params[free_buf].dst) +
  //      " output_stride: " + STR(output_stride) +
  //      " rcols_step: " + STR(rcols_step) + " rows_step: " + STR(rows_step) +
  //      " cols_step: " + STR(cols_step) + "\n");
  alloc_dbuf(drv.dfs[0], free_buf, drv.dsr->dID, inl0);
  alloc_dbuf(drv.dfs[1], free_buf, drv.dsr->dID, inl1);
  alloc_dbuf(drv.dfs[2], free_buf, drv.dsr->dID, inl2);
  alloc_dbuf(drv.dfs[3], free_buf, drv.dsr->dID, inl3);
  TOG2("Alloc: " + STR(free_buf) + "\n");
  drv.dsr->dID++;

  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.t2.p_load_weights);
}

void Start_Compute(acc_container &drv, int inp_block, int wgt_block) {
  TOG2("Start_Compute" << endl);
  drv.mdma->multi_dma_change_start_4(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  in0[inl0++] = OPCODE_COMPUTE;
  in0[inl0++] = inp_block;
  in0[inl0++] = wgt_block;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[0].dma_wait_send();
}

void Store_Results(acc_container &drv) {
  prf_start(1);
  int r_buf = find_dbuf(drv.dfs[0], drv.dsr->rID);
  int offset = drv.dfs[0].dbuf_set[r_buf].offset;
  dealloc_dbuf(drv.dfs[0], r_buf);
  dealloc_dbuf(drv.dfs[1], r_buf);
  dealloc_dbuf(drv.dfs[2], r_buf);
  dealloc_dbuf(drv.dfs[3], r_buf);
  TOG2("Dealloc: " + STR(r_buf) + "\n");
  drv.dsr->rID++;

  struct store_params sp = drv.st_params[r_buf];

  int output_stride = sp.dcs;
  int rcols_step = sp.cols;
  int rows_step = sp.rrows;
  int cols_step = sp.rcols;
  int8_t *base = reinterpret_cast<int8_t *>(drv.st_params[r_buf].dst);
  // TOG2("offset: " + STR((unsigned long long)drv.st_params[r_buf].dst) +
  //      " output_stride: " + STR(output_stride) +
  //      " rcols_step: " + STR(rcols_step) + " rows_step: " + STR(rows_step) +
  //      " cols_step: " + STR(cols_step) + "\n");
  int *o0 = drv.mdma->dmas[0].dma_get_outbuffer() + (offset / 4);
  int *o1 = drv.mdma->dmas[1].dma_get_outbuffer() + (offset / 4);
  int *o2 = drv.mdma->dmas[2].dma_get_outbuffer() + (offset / 4);
  int *o3 = drv.mdma->dmas[3].dma_get_outbuffer() + (offset / 4);
  int8_t *bo0 = reinterpret_cast<int8_t *>(o0);
  int8_t *bo1 = reinterpret_cast<int8_t *>(o1);
  int8_t *bo2 = reinterpret_cast<int8_t *>(o2);
  int8_t *bo3 = reinterpret_cast<int8_t *>(o3);
  int out0 = 0;
  int out1 = 0;
  int out2 = 0;
  int out3 = 0;
  int drows = rows_step - (rows_step % 4);
  int colsr = rcols_step - cols_step;
  int unrolled_cols = cols_step - cols_step % 16;

#ifndef ACC_NEON
  for (int i = 0; i < drows; i += 4) {
    for (int j = 0; j < cols_step; j++) {
      base[(i + 0) * output_stride + j] = bo0[out0++];
      base[(i + 1) * output_stride + j] = bo1[out1++];
      base[(i + 2) * output_stride + j] = bo2[out2++];
      base[(i + 3) * output_stride + j] = bo3[out3++];
    }
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
    out3 += colsr;
  }
#else
  for (int i = 0; i < drows; i += 4) {
    int8x16_t tmp0;
    int8x16_t tmp1;
    int8x16_t tmp2;
    int8x16_t tmp3;
    int di0 = i * output_stride;
    int di1 = (i + 1) * output_stride;
    int di2 = (i + 2) * output_stride;
    int di3 = (i + 3) * output_stride;
    for (int j = 0; j < unrolled_cols; j += 16) {
      tmp0 = vld1q_s8(bo0 + out0);
      tmp1 = vld1q_s8(bo1 + out1);
      tmp2 = vld1q_s8(bo2 + out2);
      tmp3 = vld1q_s8(bo3 + out3);
      vst1q_s8(base + di0 + j, tmp0);
      vst1q_s8(base + di1 + j, tmp1);
      vst1q_s8(base + di2 + j, tmp2);
      vst1q_s8(base + di3 + j, tmp3);
      out0 += 16;
      out1 += 16;
      out2 += 16;
      out3 += 16;
    }
    for (int j = unrolled_cols; j < cols_step; j++) {
      base[di0 + j] = bo0[out0++];
      base[di1 + j] = bo1[out1++];
      base[di2 + j] = bo2[out2++];
      base[di3 + j] = bo3[out3++];
    }
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
    out3 += colsr;
  }
#endif

  if ((rows_step % 4) == 3) {
    for (int j = 0; j < cols_step; j++) {
      base[(drows + 0) * output_stride + j] = bo0[out0++];
      base[(drows + 1) * output_stride + j] = bo1[out1++];
      base[(drows + 2) * output_stride + j] = bo2[out2++];
    }
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
  } else if ((rows_step % 4) == 2) {
    for (int j = 0; j < cols_step; j++) {
      base[(drows + 0) * output_stride + j] = bo0[out0++];
      base[(drows + 1) * output_stride + j] = bo1[out1++];
    }
    out0 += colsr;
    out1 += colsr;
  } else if ((rows_step % 4) == 1) {
    for (int j = 0; j < cols_step; j++) {
      base[(drows + 0) * output_stride + j] = bo0[out0++];
    }
    out0 += colsr;
  }
  prf_end(1, drv.t2.p_store);
}

struct Load_Send_Acc : Task {

  Load_Send_Acc(acc_container &drv_, int8_t *results_, int rcols_, int col_inc_,
                int cols_, int output_stride_, int rdepth_)
      : drv(drv_), results(results_), rcols(rcols_), col_inc(col_inc_),
        cols(cols_), output_stride(output_stride_), rdepth(rdepth_) {}

  void Run() override {
    for (int c = 0; c < rcols; c += col_inc) {
      int free_buf = wait_for_free_dbuf(drv.dfs[0]);
      TOG2("Free buffer found: " + STR(free_buf) + "\n");
      int rcols_step = std::min(col_inc, rcols - c);
      int cols_step = std::min(col_inc, cols - c);
      TOG2("Loading weights: " + STR(c) + " to " + STR(c + rcols_step) +
           " for " + STR(r) + "\n");
      TOG2("LDSR: " + drv.dsr->str() + "\n");
      Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step, r,
                       rrows_step, rdepth, rows_step, cols_step);
      TOG2("LDSR: " + drv.dsr->str() + "\n");
      TOG2("Starting Transfer: " + STR(c) + " to " + STR(c + rcols_step) +
           "\n");
      drv.Start_Transfer();
      TOG2("LDSR: " + drv.dsr->str() + "\n");
      TOG2("Starting Compute : " + STR(c) + " to " + STR(c + rcols_step) +
           "\n");

      Start_Compute(drv, rrows_step, rcols_step);
    }
  }

  acc_container &drv;
  int8_t *results;
  int rcols;
  int col_inc;
  int cols;
  int output_stride;
  int rdepth;

  int r;
  int rrows_step;
  int rows_step;
};

struct Store_Results_Acc : Task {

  Store_Results_Acc(acc_container &drv_, int rcols_, int col_inc_)
      : drv(drv_), rcols(rcols_), col_inc(col_inc_) {}

  void Run() override {
    for (int c = 0; c < rcols; c += col_inc) {
      int rcols_step = std::min(col_inc, rcols - c);
      TOG2("SDSR: " + drv.dsr->str() + "\n");
      TOG2("Set_Results: " + STR(c) + " to " + STR(c + rcols_step) + " for " +
           STR(r) + "\n");
      drv.Set_Results();
      TOG2("SDSR: " + drv.dsr->str() + "\n");

      TOG2("Recieve_Results: " + STR(c) + " to " + STR(c + rcols_step) + "\n");
      drv.Recieve_Results();

      TOG2("Storing Results: " + STR(c) + " to " + STR(c + rcols_step) + "\n");
      Store_Results(drv);
      TOG2("SDSR: " + drv.dsr->str() + "\n");
    }
  }

  acc_container &drv;
  int rcols;
  int col_inc;
  int r;
};

} // namespace tflite_vm

#endif // GEMM_MT