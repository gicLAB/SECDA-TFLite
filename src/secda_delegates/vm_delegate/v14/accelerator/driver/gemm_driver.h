#ifndef GEMM_DRIVER
#define GEMM_DRIVER

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <sys/stat.h>
#include <typeinfo>

// GEMM_Driver for simulated VM acccelerator
namespace tflite_vm {

void Config_Acc(acc_container &drv) {
  drv.mdma->multi_dma_change_start_4(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  in0[inl0++] = OPCODE_CONFIG;
  in0[inl0++] = roundUp(drv.depth, 16);
  // in0[inl0++] = roundUp(drv.depth, 32);
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
#endif
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.wgt_start = true;

  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.t2.load_send_inputs);
}

void Load_Weight_Data(acc_container &drv, int free_buf, int8_t *results,
                      int output_stride, int c, int rcols_step, int r,
                      int rrows_step, int rdepth_step, int rows_step,
                      int cols_step, bool tileOpt) {
  prf_start(1);

  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer();

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;
  if (tileOpt) {
    in0[inl0++] = OPCODE_NOP;
    // cout << "weight tileOpt" << endl;
  } else {

    int w_dex = (drv.w_c / 4);
    int data_length = rdepth_step * rcols_step;
    int rcolnoDiv4 = rcols_step / 4;
    int rdepthDiv4 = rdepth_step / 4;
    int minWgtBlockNo =
        rcolnoDiv4 / VMM_COUNT; // minimum weight block number that will be
                                // distributed in each VMM/GEMM.
    int getExtraWgtBlock = (rcolnoDiv4 % VMM_COUNT); // who will get extra block

    in0[inl0++] = OPCODE_LOAD_WGT;
    in0[inl0++] = rcolnoDiv4;               // rcolnoDiv4
    in0[inl0++] = rdepthDiv4;               // rdepthDiv4
    in0[inl0++] = minWgtBlockNo;            // minWgtBlockNo
    in0[inl0++] = getExtraWgtBlock;         // getExtraWgtBlock
    in0[inl0++] = (rdepthDiv4) / DEPTHTILE; // depthSwitch

#ifndef DMA_WGT_PRELOAD
    // int wsend_full = data_length / (32); // Bit shift specific code
    int wsend_full = data_length / (16);
    int wsend_neon_len = roundDown(wsend_full, 4);

    // cout << "Testpoint 1" << endl;
#ifndef ACC_NEON
    for (int i = 0; i < wsend_full; i++) {
      in0[inl0++] = drv.wb_0[w_dex + i];
      in1[inl1++] = drv.wb_1[w_dex + i];
      in2[inl2++] = drv.wb_2[w_dex + i];
      in3[inl3++] = drv.wb_3[w_dex + i];
    }
#else
    for (int i = 0; i < wsend_neon_len; i += 4) {
      vst1q_s32(in0 + inl0, vld1q_s32(drv.wb_0 + w_dex + i));
      vst1q_s32(in1 + inl1, vld1q_s32(drv.wb_1 + w_dex + i));
      vst1q_s32(in2 + inl2, vld1q_s32(drv.wb_2 + w_dex + i));
      vst1q_s32(in3 + inl3, vld1q_s32(drv.wb_3 + w_dex + i));
      inl0 += 4;
      inl1 += 4;
      inl2 += 4;
      inl3 += 4;
    }
    for (int i = wsend_neon_len; i < wsend_full;
         i++) { // Bit shift specific code
      in0[inl0++] = drv.wb_0[w_dex + i];
      in1[inl1++] = drv.wb_1[w_dex + i];
      in2[inl2++] = drv.wb_2[w_dex + i];
      in3[inl3++] = drv.wb_3[w_dex + i];
    }
#endif

#else

    prf_start(3);
    drv.mdma->dmas[0].dma_start_send(inl0);
    drv.mdma->dmas[1].dma_start_send(inl1);
    drv.mdma->dmas[2].dma_start_send(inl2);
    drv.mdma->dmas[3].dma_start_send(inl3);
    drv.mdma->multi_dma_wait_send();
    // write the function to send the preloaded DMA weights to accelerator
    int offset =
        drv.t->layer_wgt_offsets[drv.t->conv_layer_no] + drv.t->wgt_tile_offset;
    // DLOG("Load_Weight_Data: conv_layer_no: "
    //      << drv.t->conv_layer_no << " layer_wgt_offsets: "
    //      << drv.t->layer_wgt_offsets[drv.t->conv_layer_no]);
    // DLOG("Load_Weight_Data: offset: " << offset);
    drv.mdma->multi_dma_change_start_4((DMA_SCRATCH_SIZE_4 + offset) * 4);
    drv.mdma->dmas[0].dma_start_send(data_length / 16);
    drv.mdma->dmas[1].dma_start_send(data_length / 16);
    drv.mdma->dmas[2].dma_start_send(data_length / 16);
    drv.mdma->dmas[3].dma_start_send(data_length / 16);
    drv.mdma->multi_dma_wait_send();
    drv.t->wgt_tile_offset += data_length / 16;
    prf_end(3, drv.t2.send_weights);

    // change the start address of the multi DMA to 0 for other transfers
    drv.mdma->multi_dma_change_start_4(0);
    inl0 = 0;
    inl1 = 0;
    inl2 = 0;
    inl3 = 0;
#endif

    int b_c = c;
    int crf_c = c;
    int crx_c = c;
    int start_dex = (c / 4);
    int *wsums1 = reinterpret_cast<int *>(&drv.wt_sum1[start_dex]);
    int *wsums2 = reinterpret_cast<int *>(&drv.wt_sum2[start_dex]);
    int *wsums3 = reinterpret_cast<int *>(&drv.wt_sum3[start_dex]);
    int *wsums4 = reinterpret_cast<int *>(&drv.wt_sum4[start_dex]);

    for (int i = 0; i < rcolnoDiv4; i++) {
      in0[inl0++] = wsums1[i];
      in1[inl1++] = wsums2[i];
      in2[inl2++] = wsums3[i];
      in3[inl3++] = wsums4[i];

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
    // drv.w_c += data_length / 8; // Bit shift specific code
    drv.w_c += data_length / 4;
  }

  int8_t *res_pointer = results + c + r * output_stride;
  drv.st_params[free_buf].dst = reinterpret_cast<int *>(res_pointer);
  drv.st_params[free_buf].dcs = output_stride;
  drv.st_params[free_buf].cols = rcols_step;
  drv.st_params[free_buf].rows = rrows_step;
  drv.st_params[free_buf].rrows = rows_step;
  drv.st_params[free_buf].rcols = cols_step;

  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));

  prf_start(2);
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  prf_end(2, drv.t2.send_weights);
  prf_end(1, drv.t2.load_send_weights);
}

void Start_Compute(acc_container &drv, int inp_block, int wgt_block) {
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
  int r_buf = 0;

  struct store_params sp = drv.st_params[r_buf];
  int output_stride = sp.dcs;
  int rcols_step = sp.cols;
  int rows_step = sp.rrows;
  int cols_step = sp.rcols;
  int8_t *base = reinterpret_cast<int8_t *>(sp.dst);

  int *o0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int *o1 = drv.mdma->dmas[1].dma_get_outbuffer();
  int *o2 = drv.mdma->dmas[2].dma_get_outbuffer();
  int *o3 = drv.mdma->dmas[3].dma_get_outbuffer();
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
#if 1
    for (int j = unrolled_cols; j < cols_step; j++) {
      base[di0 + j] = bo0[out0++];
      base[di1 + j] = bo1[out1++];
      base[di2 + j] = bo2[out2++];
      base[di3 + j] = bo3[out3++];
    }
#endif
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
    out3 += colsr;
  }
#endif

#if 1
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
#endif
}

void Load_Weight_Compute_Store(acc_container &drv, int8_t *results,
                               int output_stride, int c, int rcols_step, int r,
                               int rrows_step, int rdepth_step, int rows_step,
                               int cols_step, bool tileOpt) {
  int free_buf = check_for_free_dbuf(drv.dfs[0]);

  // cout << "Load_Weight_data" << endl;
  Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step, r,
                   rrows_step, rdepth_step, rows_step, cols_step, tileOpt);

  prf_start(1);
  drv.Set_Results();
  prf_end(1, drv.t2.set_results);

  prf_start(2);
  Start_Compute(drv, rrows_step, rcols_step);
  prf_end(2, drv.t2.start_compute);

  prf_start(3);
  drv.Recieve_Results();
  prf_end(3, drv.t2.receive_results);

  prf_start(4);
  Store_Results(drv);
  prf_end(4, drv.t2.store);
}

void find_critical_rcol_step(int *crsArr) {
  for (int i = 0; i < VMM_COUNT - 2; i++) {
    crsArr[i] = 4 * (VMM_COUNT - (i + 1));
  }
}

void find_critical_depth(int *cdArr, int *crsArr) {
  for (int i = 0; i < VMM_COUNT - 2; i++) {
    cdArr[i] = (4 * 4 * WGT_BUF_LEN) / crsArr[i];
  }
}

int check_for_crs_cd(int *crsArr, int *cdArr, int *rcols_step, int rdepth,
                     int *col_inc) {
  for (int i = 0; i < VMM_COUNT - 2; i++) {
    if (*rcols_step == crsArr[i] && rdepth > cdArr[i]) {
      // check if we decrease the rcols_step by 4  until it 4
      // check each time (rcols_step-4)*rdepth is less than
      // 4*4*WGT_BUF_LEN
      // if yes then return 1
      for (int j = *rcols_step; j > 4; j -= 4) {
        if ((j - 4) * rdepth < (4 * 4 * WGT_BUF_LEN)) {
          // cout << "critical rcols_step and rdepth found" << endl;
          // cout << "rcols_step: " << *rcols_step << " rdepth: " << rdepth
          //      << " col_inc: " << *col_inc << endl;
          *col_inc = j - 4;
          *rcols_step = j - 4;
          // cout << "rcols_step updated: " << *rcols_step
          //      << " col_inc updated: " << *col_inc << endl;
          return 1;
        }
      }
    }
  }
  return 0;
}

void TileGEMM(acc_container &drv, int output_stride, int depth, int rdepth,
              int rows, int rrows, int cols, int rcols, int8_t *results) {
  prf_start(1);
  drv.t->layer_weight_tile = 0;
  drv.t->layer_input_tile = 0;

  //// need to review the following code

  int *crsArr = new int[VMM_COUNT - 2];
  int *cdArr = new int[VMM_COUNT - 2];

  // here 4 indicates no of data per 32-bit(BRAM_Data_width or
  // AXI_DATA_WIDTH) Each weight data is 8-bit for this case
  if (rdepth > (4 * WGT_BUF_LEN)) {
    cout << "rdepth is greater than 4*WG_BUF_LEN= " << 4 * WGT_BUF_LEN
         << " which cannot be handled by single VMM local weight buffer "
            "single "
            "channel that is 4*WGT_BUF_LEN"
         << endl;
    exit(0);
  }

  find_critical_rcol_step(crsArr);
  find_critical_depth(cdArr, crsArr);

  //// END: need to review the above code

  int acc_weight_buffer_size = WGT_BUF_LEN * 16 * VMM_COUNT;
  int acc_input_buffer_size = GINP_BUF_LEN * 16;
  int max_cols_beforeRounding = acc_weight_buffer_size / rdepth;
  int max_cols = max_cols_beforeRounding - (max_cols_beforeRounding % 4);

  int col_inc =
      std::min(std::min(rcols, max_cols), (WSUMS_BUF_LEN * VMM_COUNT * 4));
  if (std::min(rcols, max_cols) >= (WSUMS_BUF_LEN * VMM_COUNT * 4)) {
    cout << "col_inc was limited by WSUMS_BUF_LEN" << endl;
  }

  // No of wgtBlocks from col_inc
  int noOfWgtBlocks = col_inc / 4;
  int noOfWgtBlocksPerVMM = noOfWgtBlocks / VMM_COUNT;
  // check if noOfWgtBlocks is divisible by VMM_COUNT
  int noOfWgtBlocksRem = noOfWgtBlocks % VMM_COUNT;
  // rest of the wgtBlocks will be distributed to the first noOfWgtBlocksRem
  // VMMs therefore, each of the first noOfWgtBlocksRem VMMs will have one extra
  // wgtBlock
  // if (noOfWgtBlocksRem > 0) noOfWgtBlocksRem = 1; // not tested
  // calculate maximum no of wgtBlocks one VMM can handle
  // int maxWgtBlocks = noOfWgtBlocks + noOfWgtBlocksRem;
  int maxWgtBlocks = noOfWgtBlocksPerVMM + noOfWgtBlocksRem;
  // check if maxWgtBlocks size is less than individual weght buffer size
  int acc_individual_weight_buffer_size = WGT_BUF_LEN * 16; // 8*2*WGT_BUF_LEN
  if ((maxWgtBlocks * 4 * rdepth) >
      acc_individual_weight_buffer_size) { // 4*4*WGT_BUF_LEN
    col_inc = col_inc - (noOfWgtBlocksRem * 4);
  }

  int max_rows = acc_input_buffer_size / rdepth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(rrows, max_rows);
  // row_inc = 4;
  Config_Acc(drv);
  for (int r = 0; r < rrows; r += row_inc) {
    int rrows_step = std::min(row_inc, rrows - r);
    int rows_step = std::min(row_inc, rows - r);
    drv.w_c = 0;
    // Load Inputs into the accelerator
#if 1
    Load_Input_Data(drv, r, rrows_step, depth, rdepth);
#endif
    // drv.t->layer_weight_tile = 0;
    bool tileOpt = false;

    // function of tileOpt: when there is single weight tile but multiple input
    // tiles, then we can set the tileOpt to true for weight tiles to avoid
    // same weight tile being loaded multiple times for each input tile
    if (rcols == col_inc && rrows > row_inc) {
      if (r > 0) {
        tileOpt = true;
        // cout << "tileOpt is set to true for layer: " << drv.t->layer
        //      << " node: " << drv.t->node << endl;
      }
    }
    for (int c = 0; c < rcols; c += col_inc) {
      // cout << "weight tile: " << drv.t->layer_weight_tile << endl;
      int rcols_step = std::min(col_inc, rcols - c);
      int cols_step = std::min(col_inc, cols - c);

      // check for critical rcols_step and rdepth
      if (rcols_step < (4 * VMM_COUNT)) {
        check_for_crs_cd(crsArr, cdArr, &rcols_step, rdepth, &col_inc);
      }
#if 1
      Load_Weight_Compute_Store(drv, results, output_stride, c, rcols_step, r,
                                rrows_step, rdepth, rows_step, cols_step,
                                tileOpt);
#endif
      if (!tileOpt) {
        drv.t->layer_weight_tile++;
      }
    }
    drv.t->wgt_tile_offset = 0;
    drv.mdma->multi_dma_change_start_4(0);
    drv.t->layer_input_tile++;
  }
  // cout << "layer: " << drv.t->layer << " node: " << drv.t->node
  //      << " layer_weight_tile: " << drv.t->layer_weight_tile
  //      << " layer_input_tile: " << drv.t->layer_input_tile << endl;
  prf_end(1, drv.t2.vm_acc);
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
  cout << "VM" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info: Layer: " << drv.t->layer << " Node: " << drv.t->node
       << endl;
  cout << "rdepth: " << rdepth << " depth: " << depth << endl;
  cout << "rcols: " << rcols << " cols: " << cols << endl;
  cout << "rrows: " << rrows << " rows: " << rows << endl;
  cout << "output_stride: " << output_stride << endl;
  cout << "===========================" << endl;
#endif

  TileGEMM(drv, output_stride, depth, rdepth, rows, rrows, cols, rcols, dst);

  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));

#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/out_vm_" + std::to_string(drv.t->layer) + ".csv");
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