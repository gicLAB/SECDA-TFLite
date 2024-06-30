#ifndef GEMM_DRIVER
#define GEMM_DRIVER

#include "acc_container.h"
#include "gemm_mt.h"

#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <sys/stat.h>
#include <typeinfo>

// #define TOG(X)                                                                 \
//   if (drv.verb) X;

// #define TOG(X) X

// GEMM_Driver for simulated VM acccelerator
namespace tflite_vm {

tflite_vm::Load_Send_Acc *LSA;
tflite_vm::Store_Results_Acc *SRA;

void Config_Acc(acc_container &drv) {
  drv.mdma->multi_dma_change_start_4(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  in0[inl0++] = OPCODE_CONFIG;
  in0[inl0++] = roundUp(drv.depth, 16);
  in0[inl0++] = drv.ra;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[0].dma_wait_send();
}

// Previously called Load_inp_Data
void Load_Input_Data(acc_container &drv, int start_row, int rows_step,
                     int depth, int rdepth) {
  prf_start(1);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer();

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int offdepth = depth * drv.inp_offset;
  int start_dex = (start_row / 4);
  int *p_inp_sums1 = reinterpret_cast<int *>(&drv.in_sum1[start_dex]);
  int *p_inp_sums2 = reinterpret_cast<int *>(&drv.in_sum2[start_dex]);
  int *p_inp_sums3 = reinterpret_cast<int *>(&drv.in_sum3[start_dex]);
  int *p_inp_sums4 = reinterpret_cast<int *>(&drv.in_sum4[start_dex]);

  int rrow_steps = ((rows_step + 3) - ((rows_step + 3) % 4));
  int in_sum_length = rrow_steps / 4;
  in0[inl0++] = OPCODE_LOAD_INP;
  in0[inl0++] = (rrow_steps * rdepth / 16); // inp_size
  in0[inl0++] = in_sum_length;              // inp_sum_size

#ifndef ACC_NEON
  for (int c = 0; c < rows_step; c += 4) {
    for (int i = 0; i < rdepth / 4; i++) {
      in0[inl0++] = drv.inb_0[i + drv.in_id];
      in1[inl1++] = drv.inb_1[i + drv.in_id];
      in2[inl2++] = drv.inb_2[i + drv.in_id];
      in3[inl3++] = drv.inb_3[i + drv.in_id];
    }
    drv.in_id += rdepth / 4;
  }
  for (int i = 0; i < in_sum_length; i++) {
    in0[inl0++] = (p_inp_sums1[i] + offdepth) * drv.wgt_offset;
    in1[inl1++] = (p_inp_sums2[i] + offdepth) * drv.wgt_offset;
    in2[inl2++] = (p_inp_sums3[i] + offdepth) * drv.wgt_offset;
    in3[inl3++] = (p_inp_sums4[i] + offdepth) * drv.wgt_offset;
  }
#else
  int32x4_t tmp0;
  int32x4_t tmp1;
  int32x4_t tmp2;
  int32x4_t tmp3;
  for (int r = 0; r < rows_step; r += 4) {
    int *inb0 = drv.inb_0;
    int *inb1 = drv.inb_1;
    int *inb2 = drv.inb_2;
    int *inb3 = drv.inb_3;
    for (int i = 0; i < rdepth / 4; i += 4) {
      tmp0 = vld1q_s32(inb0 + i + drv.in_id);
      tmp1 = vld1q_s32(inb1 + i + drv.in_id);
      tmp2 = vld1q_s32(inb2 + i + drv.in_id);
      tmp3 = vld1q_s32(inb3 + i + drv.in_id);
      vst1q_s32(in0 + inl0, tmp0);
      vst1q_s32(in1 + inl1, tmp1);
      vst1q_s32(in2 + inl2, tmp2);
      vst1q_s32(in3 + inl3, tmp3);
      inl0 += 4;
      inl1 += 4;
      inl2 += 4;
      inl3 += 4;
    }
    drv.in_id += rdepth / 4;
  }
  int vin_sum_len = roundDown(in_sum_length, 4);
  const int32_t *tmp_wgt_off =
      reinterpret_cast<const int32_t *>(&drv.wgt_offset);
  const int32_t *tmp_offdepth = reinterpret_cast<const int32_t *>(&offdepth);
  int32x4_t vwgt_wgtoffset = vld1q_dup_s32(tmp_wgt_off);
  int32x4_t vwgt_offdepth = vld1q_dup_s32(tmp_offdepth);
  for (int i = 0; i < vin_sum_len; i += 4) {
    vst1q_s32(in0 + inl0,
              vmulq_s32(vaddq_s32(vld1q_s32(p_inp_sums1 + i), vwgt_offdepth),
                        vwgt_wgtoffset));
    vst1q_s32(in1 + inl1,
              vmulq_s32(vaddq_s32(vld1q_s32(p_inp_sums2 + i), vwgt_offdepth),
                        vwgt_wgtoffset));
    vst1q_s32(in2 + inl2,
              vmulq_s32(vaddq_s32(vld1q_s32(p_inp_sums3 + i), vwgt_offdepth),
                        vwgt_wgtoffset));
    vst1q_s32(in3 + inl3,
              vmulq_s32(vaddq_s32(vld1q_s32(p_inp_sums4 + i), vwgt_offdepth),
                        vwgt_wgtoffset));
    inl0 += 4;
    inl1 += 4;
    inl2 += 4;
    inl3 += 4;
  }
  for (int i = vin_sum_len; i < in_sum_length; i++) {
    in0[inl0++] = (p_inp_sums1[i] + offdepth) * drv.wgt_offset;
    in1[inl1++] = (p_inp_sums2[i] + offdepth) * drv.wgt_offset;
    in2[inl2++] = (p_inp_sums3[i] + offdepth) * drv.wgt_offset;
    in3[inl3++] = (p_inp_sums4[i] + offdepth) * drv.wgt_offset;
  }
#endif
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.wgt_start = true;

  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.t2.p_load_inputs);
}

void Load_Weight_Compute_Store(acc_container &drv, int8_t *results,
                               int output_stride, int c, int rcols_step, int r,
                               int rrows_step, int rdepth_step, int rows_step,
                               int cols_step) {
  int free_buf = check_for_free_dbuf(drv.dfs[0]);
  Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step, r,
                   rrows_step, rdepth_step, rows_step, cols_step);
  prf_start(1);
  drv.Start_Transfer();
  drv.Set_Results();
  Start_Compute(drv, rrows_step, rcols_step);

  drv.Recieve_Results();
  prf_end(1, drv.t2.p_vm_acc);

  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  Store_Results(drv);
}

void VM_Inner_Threaded(acc_container &drv, int r, int rrows_step,
                       int rows_step) {
  int c = 0;
  TOG(cerr << "Starting Threaded Load and Store" << endl;);
  std::vector<Task *> tasks;
  auto *workers_pool = drv.mt_context->workers_pool();
  LSA->r = r;
  LSA->rrows_step = rrows_step;
  LSA->rows_step = rows_step;
  SRA->r = r;
  tasks.push_back(LSA);
  tasks.push_back(SRA);
  workers_pool->Execute(tasks);
  TOG(cerr << "Finished Threaded Load and Store" << endl;);
}

void TileGEMM(acc_container &drv, int output_stride, int depth, int rdepth,
              int rows, int rrows, int cols, int rcols, int8_t *results) {
  // prf_start(1);
  drv.t.layer_weight_tile = 0;
  drv.t.layer_input_tile = 0;
  int acc_weight_buffer_size = WGT_BUF_LEN * 16;
  int acc_input_buffer_size = GINP_BUF_LEN * 16;
  int max_cols = acc_weight_buffer_size / rdepth;
  max_cols = max_cols - (max_cols % 4);
  int col_inc = std::min(std::min(rcols, max_cols), WSUMS_BUF_LEN);
  int max_rows = acc_input_buffer_size / rdepth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(std::min(rrows, max_rows), ISUMS_BUF_LEN);
  assert(col_inc > 0 && "col_inc must be greater than 0");
  assert(row_inc > 0 && "row_inc must be greater than 0");
  drv.dsr = new struct DSR();

  if (drv.thread_count > 1) {
    TOG(cout << "Threaded GEMM" << endl;);
    LSA = new struct Load_Send_Acc(drv, results, rcols, col_inc, cols,
                                   output_stride, rdepth);
    SRA = new struct Store_Results_Acc(drv, rcols, col_inc);
  }
  Config_Acc(drv);
  for (int r = 0; r < rrows; r += row_inc) {
    int rrows_step = std::min(row_inc, rrows - r);
    int rows_step = std::min(row_inc, rows - r);
    drv.w_c = 0;
    // Load Inputs into the accelerator
    Load_Input_Data(drv, r, rrows_step, depth, rdepth);

    if (drv.thread_count > 3) {
      VM_Inner_Threaded(drv, r, rrows_step, rows_step);
    } else {
      for (int c = 0; c < rcols; c += col_inc) {
        int rcols_step = std::min(col_inc, rcols - c);
        int cols_step = std::min(col_inc, cols - c);
        Load_Weight_Compute_Store(drv, results, output_stride, c, rcols_step, r,
                                  rrows_step, rdepth, rows_step, cols_step);
        drv.t.layer_weight_tile++;
      }
    }
    drv.dsr->reset();
    drv.mdma->multi_dma_change_start_4(0);
    drv.t.layer_input_tile++;
  }
  // cin.ignore();
  if (drv.thread_count > 1) {
    delete LSA;
    delete SRA;
  }
  delete drv.dsr;

  // prf_end(1, drv.t2.p_vm_acc);
}

void Entry(acc_container &drv, int8_t *dst) {
  int rows = drv.rows;
  int cols = drv.cols;
  int depth = drv.depth;
  int rrows = roundUp(drv.rows, 2);
  int rcols = roundUp(drv.cols, 4);
  int rdepth = roundUp(drv.depth, 16);
  int output_stride = drv.cols;

#if defined(SYSC) || defined(DELEGATE_VERBOSE)
  cerr << "VM" << endl;
  cerr << "===========================" << endl;
  cerr << "Pre-ACC Info: " << drv.t.layer << endl;
  cerr << "rdepth: " << rdepth << " depth: " << depth << endl;
  cerr << "rcols: " << rcols << " cols: " << cols << endl;
  cerr << "rrows: " << rrows << " rows: " << rows << endl;
  cerr << "output_stride: " << output_stride << endl;
  cerr << "===========================" << endl;
#endif
  // if (drv.t.layer >= 14) {
  //   drv.thread_count = 1;
  // }

  TileGEMM(drv, output_stride, depth, rdepth, rows, rrows, cols, rcols, dst);
  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/out_vm_" + std::to_string(drv.t.layer) + "_1.csv");
  int8_t *res_pointer = dst;
  int index = 0;
  for (int r = 0; r < rows; r++) {
    myfile << endl;
    for (int c = 0; c < cols; c++) {
      myfile << (int)res_pointer[index] << ",";
      index++;
    }
  }
  myfile.close();
#endif
}

} // namespace tflite_vm
#endif // GEMM_DRIVER
