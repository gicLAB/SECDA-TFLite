#ifndef VMM_COMPUTE_H
#define VMM_COMPUTE_H

#include "acc_config.sc.h"

sc_int<64> VMM_UNIT::mul_s64(int a, sc_int<64> b) {
  sc_int<64> c;
  // #pragma HLS RESOURCE variable = c core = MulnS
  c = a * b;
  return c;
}

int VMM_UNIT::Quantised_Multiplier_gemmlowp(int x, int qm, sc_int<64> pl,
                                            sc_int<32> pr, sc_int<32> msk,
                                            sc_int<32> sm) {
  sc_int<64> val = mul_s64(x, pl);
  if (val > MAX) val = MAX; // ALU MIN
  if (val < MIN) val = MIN; // ALU MAX
  sc_int<64> val_2 = mul_s64(qm, val);
  sc_int<32> temp_1;
  temp_1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
  sc_int<32> val_3 = temp_1;
  val_3 = val_3 >> pr;
  sc_int<32> temp_2 = temp_1 & msk;
  sc_int<32> temp_3 = (temp_1 < 0) & 1;
  sc_int<32> temp_4 = sm + temp_3;
  sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
  sc_int<32> result_32 = val_3 + temp_5;
  int res = result_32;
  return result_32;
}

// int VMM_UNIT::Quantised_Multiplier_ruy_reference(int x, int qm, sc_int<8>
// shift) {
//   int nshift = shift;
//   int total_shift = 31 - shift;
//   sc_int<64> x_64 = x;
//   sc_int<64> quantized_multiplier_64(qm);
//   sc_int<64> one = 1;
//   sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
//   sc_int<64> result =
//       x_64 * quantized_multiplier_64 + round; // ALU ADD + ALU MUL
//   result = result >> total_shift;             // ALU SHRI
//   int nresult = result;
//   if (result > MAX) result = MAX; // ALU MIN
//   if (result < MIN) result = MIN; // ALU MAX
//   sc_int<32> result_32 = result;
//   return result_32;
// }
//  sc_int<32> one = 1;
//  sc_uint<5> roundSh = total_shift.range(4,0) - 1;
//  sc_int<32> round = one << roundSh;

int VMM_UNIT::Quantised_Multiplier_ruy_reference(int x, int qm, sc_int<8> shift,
                                                 sc_int<64> round) {
  sc_int<7> total_shift = shift;
  sc_int<32> x_64 = x; // accumulator
  sc_int<32> quantized_multiplier_64(qm);
  sc_int<64> result = x_64 * quantized_multiplier_64; // ALU ADD + ALU MUL
  result = result + round;
  result = result >> (total_shift.range(5, 0)); // ALU SHRI
  if (result > MAX) result = MAX;               // ALU MIN
  if (result < MIN) result = MIN;               // ALU MAX
  sc_int<32> result_32 = result;
  return result_32;
}

void VMM_UNIT::LoadInputs() {
  wait();
  while (1) {
    while (!load_inp.read()) wait();
    int len = inp_len.read();
    int len_switch = len / DEPTHTILE;
    // if (len_switch < 1) {
    //   len_switch = 1;
    // }

    for (int i = 0; i < len; i++) {
#pragma HLS pipeline II = 1
      bUF data = inp_fifo.read();
      // switch when we reach len_switch

      if (i < len_switch) {
        // data.unpack(inp_1a_1, inp_1b_1, inp_1c_1, inp_1d_1,
        //             i - (len_switch * 0));
        data.unpack_URAM(inp_1a_1, inp_1b_1, (i - (len_switch * 0))); // URAM
      }
#ifdef EN_DEPTHTILE
      else if (i >= len_switch && i < (len_switch * 2)) {
        // data.unpack(inp_1a_2, inp_1b_2, inp_1c_2, inp_1d_2,
        //             (i - (len_switch * 1)));
        data.unpack_URAM(inp_1a_2, inp_1b_2, (i - (len_switch * 1))); // URAM
      }
#if (DEPTHTILE == 4) || (DEPTHTILE == 6) || (DEPTHTILE == 8)
      else if (i >= (len_switch * 2) && i < (len_switch * 3)) {
        data.unpack(inp_1a_3, inp_1b_3, inp_1c_3, inp_1d_3,
                    (i - (len_switch * 2)));
      } else if (i >= (len_switch * 3) && i < (len_switch * 4)) {
        data.unpack(inp_1a_4, inp_1b_4, inp_1c_4, inp_1d_4,
                    (i - (len_switch * 3)));
      }
#if (DEPTHTILE == 6) || (DEPTHTILE == 8)
      else if (i >= (len_switch * 4) && i < (len_switch * 5)) {
        data.unpack(inp_1a_5, inp_1b_5, inp_1c_5, inp_1d_5,
                    (i - (len_switch * 4)));
      } else if (i >= (len_switch * 5) && i < (len_switch * 6)) {
        data.unpack(inp_1a_6, inp_1b_6, inp_1c_6, inp_1d_6,
                    (i - (len_switch * 5)));
      }
#if (DEPTHTILE == 8)
      else if (i >= (len_switch * 6) && i < (len_switch * 7)) {
        data.unpack(inp_1a_7, inp_1b_7, inp_1c_7, inp_1d_7,
                    (i - (len_switch * 6)));
      } else if (i >= (len_switch * 7) && i < (len_switch * 8)) {
        data.unpack(inp_1a_8, inp_1b_8, inp_1c_8, inp_1d_8,
                    (i - (len_switch * 7)));
      }
#endif
#endif
#endif
#endif
    }
  }
}

void VMM_UNIT::LoadWeights() {
  wait();
  while (1) {
    while (!load_wgt.read()) wait();

    int depth_switch = wgt_len.read();
    int wgtBlockNo = wgt_colno.read();
    int depthDiv4 = depthLoadWgt.read();

    // cout << "wgtBlockNo: " << wgtBlockNo << endl;
    // Expect depth should be equally divided between two VM PE

#if 1
    int lenSize = wgtBlockNo * depthDiv4;
    for (int i = 0; i < lenSize; i++) {
#pragma HLS pipeline II = 1
      bUF data = wgt_fifo.read();
      data.unpack_URAM(wgt_data1a, wgt_data2a, i);
    }
#endif

#if 0
    for (int i = 0; i < wgtBlockNo; i++) {
      for (int j = 0; j < depthDiv4; j++) {
#pragma HLS pipeline II = 1
        bUF data = wgt_fifo.read();
        // switch when we reach depth_switch
        if (j < depth_switch) {
          // data.unpack(wgt_data1a, wgt_data2a, wgt_data3a, wgt_data4a,
          //             (i * depth_switch) + (j - (depth_switch * 0)));
          data.unpack_URAM(wgt_data1a, wgt_data2a,
                           (i * depth_switch) + (j - (depth_switch * 0)));
        }
#ifdef EN_DEPTHTILE
        else if (j >= depth_switch && j < (depth_switch * 2)) {
          // data.unpack(wgt_data1a_2, wgt_data2a_2, wgt_data3a_2, wgt_data4a_2,
          //             ((i * depth_switch) + (j - (depth_switch * 1))));
          data.unpack_URAM(wgt_data1a_2, wgt_data2a_2,
                           ((i * depth_switch) + (j - (depth_switch * 1))));
        }
#if (DEPTHTILE == 4) || (DEPTHTILE == 6) || (DEPTHTILE == 8)

        else if (j >= (depth_switch * 2) && j < (depth_switch * 3)) {
          data.unpack(wgt_data1a_3, wgt_data2a_3, wgt_data3a_3, wgt_data4a_3,
                      ((i * depth_switch) + (j - (depth_switch * 2))));
        } else if (j >= (depth_switch * 3) && j < (depth_switch * 4)) {
          data.unpack(wgt_data1a_4, wgt_data2a_4, wgt_data3a_4, wgt_data4a_4,
                      ((i * depth_switch) + (j - (depth_switch * 3))));
        }
#if (DEPTHTILE == 6) || (DEPTHTILE == 8)

        else if (j >= (depth_switch * 4) && j < (depth_switch * 5)) {
          data.unpack(wgt_data1a_5, wgt_data2a_5, wgt_data3a_5, wgt_data4a_5,
                      ((i * depth_switch) + (j - (depth_switch * 4))));
        } else if (j >= (depth_switch * 5) && j < (depth_switch * 6)) {
          data.unpack(wgt_data1a_6, wgt_data2a_6, wgt_data3a_6, wgt_data4a_6,
                      ((i * depth_switch) + (j - (depth_switch * 5))));
        }
#if (DEPTHTILE == 8)
        else if (j >= (depth_switch * 6) && j < (depth_switch * 7)) {
          data.unpack(wgt_data1a_7, wgt_data2a_7, wgt_data3a_7, wgt_data4a_7,
                      ((i * depth_switch) + (j - (depth_switch * 6))));
        } else if (j >= (depth_switch * 7) && j < (depth_switch * 8)) {
          data.unpack(wgt_data1a_8, wgt_data2a_8, wgt_data3a_8, wgt_data4a_8,
                      ((i * depth_switch) + (j - (depth_switch * 7))));
        }
#endif
#endif
#endif
#endif
        DWAIT(1);
      }
    }
#endif
  }
}

void VMM_UNIT::LoadWSumsCrfCrx() {
  wait();
  while (1) {
    while (!load_wsum.read()) wait();
    int len = wsum_len.read();
    for (int i = 0; i < len; i++) {
#pragma HLS pipeline II = 1
      bUF wsumData = wsum_fifo.read();
      bUF crfData = crf_fifo.read();
      sc_int<32> crxData = crx_fifo.read();
      wsumData.unpack(wgt_sum1, wgt_sum2, wgt_sum3, wgt_sum4, i);
      crfData.unpack(crf1, crf2, crf3, crf4, i);
      // crx data unpacking
      crx1[i] = crxData.range(7, 0);
      crx2[i] = crxData.range(15, 8);
      crx3[i] = crxData.range(23, 16);
      crx4[i] = crxData.range(31, 24);
      DWAIT(1);
    }
  }
}

void VMM_UNIT::Compute() {
  computeS.write(0);
  wait();
  while (1) {
    ready.write(true);
    vmm_ready.write(true);
    computeS.write(1);
    while (!compute.read()) wait();
    ready.write(false);
    vmm_ready.write(false);
    computeS.write(2);
    wait();
    while (compute.read()) wait();

    computeS.write(3);
    // Expecting depth should be equally divided between two VM PE
    int d = (depth.read() / (4 * DEPTHTILE));
    // if (d < 1) {
    //   d = 1;
    // }
    // cout << "d: " << d << endl;
    // cout << "VM PE: 1" << endl;

    VM_PE_URAM(wgt_data1a, wgt_data2a, inp_1a_1, inp_1b_1, wgt_sum1, wgt_sum2,
               wgt_sum3, wgt_sum4, out, d, w_idx, wsum_idx, 1);

    // VM_PE(wgt_data1a, wgt_data2a, wgt_data3a, wgt_data4a, inp_1a_1,
    // inp_1b_1,
    //       inp_1c_1, inp_1d_1, wgt_sum1, wgt_sum2, wgt_sum3, wgt_sum4, out,
    //       d, w_idx, wsum_idx, 1);

#ifdef EN_DEPTHTILE
    // tiling depthwise

    VM_PE_DSP_URAM(wgt_data1a_2, wgt_data2a_2, inp_1a_2, inp_1b_2, out_2, d,
                   w_idx);

    // VM_PE_2(wgt_data1a_2, wgt_data2a_2, wgt_data3a_2, wgt_data4a_2,
    // inp_1a_2,
    //         inp_1b_2, inp_1c_2, inp_1d_2, out_2, d, w_idx);
    // VM_PE_DSP(wgt_data1a_2, wgt_data2a_2, wgt_data3a_2, wgt_data4a_2,
    // inp_1a_2,
    //           inp_1b_2, inp_1c_2, inp_1d_2, out_2, d, w_idx);

#if (DEPTHTILE == 4) || (DEPTHTILE == 6) || (DEPTHTILE == 8)
    // cout << "VM PE: 3" << endl;
    VM_PE_DSP(wgt_data1a_3, wgt_data2a_3, wgt_data3a_3, wgt_data4a_3, inp_1a_3,
              inp_1b_3, inp_1c_3, inp_1d_3, out_3, d, w_idx);
    // cout << "VM PE: 4" << endl;
    VM_PE_DSP(wgt_data1a_4, wgt_data2a_4, wgt_data3a_4, wgt_data4a_4, inp_1a_4,
              inp_1b_4, inp_1c_4, inp_1d_4, out_4, d, w_idx);
#if (DEPTHTILE == 6) || (DEPTHTILE == 8)
    // cout << "VM PE: 5" << endl;
    VM_PE_DSP(wgt_data1a_5, wgt_data2a_5, wgt_data3a_5, wgt_data4a_5, inp_1a_5,
              inp_1b_5, inp_1c_5, inp_1d_5, out_5, d, w_idx);
    // cout << "VM PE: 6" << endl;
    VM_PE_DSP(wgt_data1a_6, wgt_data2a_6, wgt_data3a_6, wgt_data4a_6, inp_1a_6,
              inp_1b_6, inp_1c_6, inp_1d_6, out_6, d, w_idx);
#if (DEPTHTILE == 8)
    // cout << "VM PE: 7" << endl;
    VM_PE_DSP(wgt_data1a_7, wgt_data2a_7, wgt_data3a_7, wgt_data4a_7, inp_1a_7,
              inp_1b_7, inp_1c_7, inp_1d_7, out_7, d, w_idx);
    // cout << "VM PE: 8" << endl;
    VM_PE_DSP(wgt_data1a_8, wgt_data2a_8, wgt_data3a_8, wgt_data4a_8, inp_1a_8,
              inp_1b_8, inp_1c_8, inp_1d_8, out_8, d, w_idx);
#endif
#endif
#endif
#endif

    vmm_ready.write(false);
    computeS.write(4);
    wait();
    while (!post_ready1) {
      computeS.write(5);
      wait();
    }

    for (int i = 0; i < 16; i++) {
#pragma HLS unroll

#if (DEPTHTILE == 8)
      g[i] = out[i][0] + out_2[i][0] + out_3[i][0] + out_4[i][0] + out_5[i][0] +
             out_6[i][0] + out_7[i][0] + out_8[i][0];

#elif (DEPTHTILE == 6)
      g[i] = out[i][0] + out_2[i][0] + out_3[i][0] + out_4[i][0] + out_5[i][0] +
             out_6[i][0];
#elif (DEPTHTILE == 4)
      g[i] = out[i][0] + out_2[i][0] + out_3[i][0] + out_4[i][0];
#elif (DEPTHTILE == 2)
      g[i] = out[i][0] + out_2[i][0];
#else
      g[i] = out[i][0];
#endif
    }

    computeS.write(6);
    post_ready1.write(false);

    // wait();
    DWAIT(1);
  }
}

// optimizing PPU will be crtical when we do tiling accross depth
// currentl single VM will do all the computation accross depth
// therefor VM time will be always higher than Post
// therefore when we do tiling accross depth we need to optimize PPU
// to reduce the time
// currently Post take ~29 cycle and VM take on average ~256 cycle
// by doing tiling accross depth we can reduce the VM time to less than 29
// cycle then we need to optimize PPU to reduce the time
// introducing intermedite buffer between VM and Post will help to reduce the
// time
// see the code of "accelerator_backup_ppu_outbuf" folder
void VMM_UNIT::Post() {
  DATA last = {5000, 1};

  post_ready1.write(true);
  ppu_done.write(false);
  postS.write(0);
  wait();
  while (true) {

    postS.write(1);
    while (post_ready1 && !send_done) wait();
    if (!post_ready1) {
      postS.write(2);

      int ppu_wsum_Idx = post_fifo.read();
      postS.write(3);

      PPU(crf1, crf2, crf3, crf4, crx1, crx2, crx3, crx4, g, r, ppu_wsum_Idx);

      postS.write(4);

      DATA data1;
      DATA data2;
      DATA data3;
      DATA data4;
      data1.pack(r[0], r[4], r[8], r[12]);
      data2.pack(r[1], r[5], r[9], r[13]);
      data3.pack(r[2], r[6], r[10], r[14]);
      data4.pack(r[3], r[7], r[11], r[15]);
      dout1.write(data1);
      dout2.write(data2);
      dout3.write(data3);
      dout4.write(data4);
      postS.write(5);
      // wait();
      DWAIT(1);
      post_ready1.write(true);
      DWAIT(2);
    }

    if (send_done) {
      dout1.write(last);
      dout2.write(last);
      dout3.write(last);
      dout4.write(last);
      ppu_done.write(true);
      while (send_done) wait();
      ppu_done.write(false);
      DWAIT(1);
    }

    // wait();
  }
}

void VMM_UNIT::PPU(ACC_DTYPE<32> *crf1, ACC_DTYPE<32> *crf2,
                   ACC_DTYPE<32> *crf3, ACC_DTYPE<32> *crf4, sc_int<8> *crx1,
                   sc_int<8> *crx2, sc_int<8> *crx3, sc_int<8> *crx4,
                   sc_int<32> *g, ACC_DTYPE<8> *r, int wsum_idx) {

  sc_int<64> round[4];
  ACC_DTYPE<32> crf_read[4];
  sc_int<8> crx_read[4];
  sc_int<8> crx_read_temp[4];

#pragma HLS array_partition variable = round complete dim = 0
#pragma HLS array_partition variable = crf_read complete dim = 0
#pragma HLS array_partition variable = crx_read complete dim = 0

  wait();
  crf_read[0] = crf1[wsum_idx];
  crf_read[1] = crf2[wsum_idx];
  crf_read[2] = crf3[wsum_idx];
  crf_read[3] = crf4[wsum_idx];

  crx_read[0] = crx1[wsum_idx];
  crx_read[1] = crx2[wsum_idx];
  crx_read[2] = crx3[wsum_idx];
  crx_read[3] = crx4[wsum_idx];

  crx_read_temp[0] = crx_read[0] - 1;
  crx_read_temp[1] = crx_read[1] - 1;
  crx_read_temp[2] = crx_read[2] - 1;
  crx_read_temp[3] = crx_read[3] - 1;

  sc_int<64> one[4] = {1, 1, 1, 1};
  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    // sc_int<64> one = 1;
    // round[i] = one[i] << (crx_read[i].range(5, 0) - 1);
    round[i] = one[i] << (crx_read_temp[i].range(5, 0));
  }
  DWAIT(9);

#if 0
  // test for simplified PPU
  for (int i = 0; i < 16; i++) {
#pragma HLS unroll
    r[i] = g[i].range(7, 0);
  }
#endif
#if 1
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
      // int accum1 = accum[j * 4 + i];
      int accum1 = g[j * 4 + i];

#ifndef __SYNTHESIS__
      int ret_accum1 = Quantised_Multiplier_ruy_reference(
          accum1, crf_read[j], crx_read[j], round[j]);
#else
      int ret_accum1 = Quantised_Multiplier_ruy_reference(
          accum1, crf_read[j], crx_read[j], round[j]);
#endif

      sc_int<32> f1_a1 = ret_accum1 + ra;
      int res = f1_a1;
      if (f1_a1 > MAX8) f1_a1 = MAX8;
      else if (f1_a1 < MIN8) f1_a1 = MIN8;
      r[j * 4 + i] = f1_a1.range(7, 0);
    }
  }
#endif
  DWAIT(44);
}

#endif // VMM_COMPUTE_H
