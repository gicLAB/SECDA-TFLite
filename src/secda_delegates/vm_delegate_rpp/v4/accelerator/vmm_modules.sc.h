#ifndef VMM_COMPUTE_H
#define VMM_COMPUTE_H

#include "acc_config.sc.h"

sc_int<64> VMM_UNIT::mul_s64(int a, sc_int<64> b) {
  sc_int<64> c;
  // #pragma HLS RESOURCE variable = c core = MulnS
  c = a * b;
  return c;
}

int VMM_UNIT::Quantised_Multiplier_ruy_reference(int x, int qm, sc_int<8> shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
  sc_int<64> result =
      x_64 * quantized_multiplier_64 + round; // ALU ADD + ALU MUL
  result = result >> total_shift;             // ALU SHRI
  int nresult = result;
  if (result > MAX) result = MAX; // ALU MIN
  if (result < MIN) result = MIN; // ALU MAX
  sc_int<32> result_32 = result;
  return result_32;
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

void VMM_UNIT::PPU(int *x, int *y, int *pcrf, sc_int<8> *pex, sc_int<32> *g,
                   sc_int<8> *r) {
  int accum[16];
  ACC_DTYPE<64> pls[4];
  ACC_DTYPE<32> prs[4];
  ACC_DTYPE<32> msks[4];
  ACC_DTYPE<32> sms[4];

#pragma HLS array_partition variable = accum complete dim = 0
#pragma HLS array_partition variable = pls complete dim = 0
#pragma HLS array_partition variable = prs complete dim = 0
#pragma HLS array_partition variable = msks complete dim = 0
#pragma HLS array_partition variable = sms complete dim = 0
  wait();
  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    for (int j = 0; j < 4; j++) {
#pragma HLS unroll
      // accum[j * 4 + i] = g[j * 4 + i] + y[i] + x[j];
      accum[j * 4 + i] = g[j * 4 + i] + x[j];
    }
  }

  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    if (pex[i] > 0) {
      pls[i] = pex[i];
      prs[i] = 0;
      msks[i] = 0;
      sms[i] = 0;
    } else {
      pls[i] = 1;
      prs[i] = -pex[i];
      msks[i] = (1 << -pex[i]) - 1;
      sms[i] = ((1 << -pex[i]) - 1) >> 1;
    }
  }

  DWAIT(9);
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 3
      int accum1 = accum[j * 4 + i];
#ifndef __SYNTHESIS__
      int ret_accum1 = Quantised_Multiplier_ruy_reference(accum1, pcrf[j], pex[j]);
#else
      int ret_accum1 = Quantised_Multiplier_gemmlowp(accum1, pcrf[j], pls[j], prs[j],
                                               msks[j], sms[j]);
#endif
      sc_int<32> f1_a1 = ret_accum1 + ra;
      int res = f1_a1;
      if (f1_a1 > MAX8) f1_a1 = MAX8;
      else if (f1_a1 < MIN8) f1_a1 = MIN8;
      r[j * 4 + i] = f1_a1.range(7, 0);
    }
  }
  DWAIT(44);
}

void VMM_UNIT::LoadWeights() {
  wait();
  wait();
  while (1) {
    while (!load_wgt.read()) wait();
    int len = wgt_len.read();
    for (int i = 0; i < wgt_len.read(); i++) {
#pragma HLS pipeline II = 1
      bUF data = wgt_fifo.read();
      data.unpack(wgt_data1a, wgt_data2a, wgt_data3a, wgt_data4a, i);
      DWAIT(1);
    }
  }
}

void VMM_UNIT::LoadInputs() {
  wait();
  while (1) {
    while (!load_inp.read()) wait();
    int len = inp_len.read();
    for (int i = 0; i < inp_len.read(); i++) {
#pragma HLS pipeline II = 1
      bUF data = inp_fifo.read();
      data.unpack(inp_1a_1, inp_1b_1, inp_1c_1, inp_1d_1, i);
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
    int d = (depth.read() / 4);
    VM_PE(wgt_data1a, wgt_data2a, wgt_data3a, wgt_data4a, inp_1a_1, inp_1b_1,
          inp_1c_1, inp_1d_1, out, d, w_idx, 0);
    vmm_ready.write(false);
    computeS.write(4);
    wait();
    while (!post_ready) {
      computeS.write(5);
      wait();
    }
    for (int i = 0; i < 16; i++) {
#pragma HLS unroll
      g[i] = out[i][0];
    }
    computeS.write(6);
    post_ready.write(false);
    wait();
  }
}

void VMM_UNIT::Post() {
  int y[4];
  int x[4];
  int pcrf[4];
  ACC_DTYPE<8> pex[4];
  DATA last1 = {5000, 0};
  DATA last2 = {5000, 0};
  DATA last3 = {5000, 0};
  DATA last = {5000, 1};

#pragma HLS array_partition variable = y complete dim = 0
#pragma HLS array_partition variable = x complete dim = 0
#pragma HLS array_partition variable = pcrf complete dim = 0
#pragma HLS array_partition variable = pex complete dim = 0
  post_ready.write(true);
  ppu_done.write(false);
  postS.write(0);
  wait();
  while (true) {

    postS.write(1);
    while (post_ready && !send_done) wait();
    if (!post_ready) {
      postS.write(2);
      for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        x[i] = post_fifo.read();
      }
      for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        pcrf[i] = post_fifo.read();
      }
      ACC_DTYPE<32> ex = post_fifo.read();
      pex[0] = ex.range(7, 0);
      pex[1] = ex.range(15, 8);
      pex[2] = ex.range(23, 16);
      pex[3] = ex.range(31, 24);
      postS.write(3);
      PPU(x, y, pcrf, pex, g, r);
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
      wait();
      DWAIT(1);

      post_ready.write(true);
      DWAIT(2);
    }
    if (send_done) {
      dout1.write(last1);
      dout2.write(last2);
      dout3.write(last3);
      dout4.write(last);
      ppu_done.write(true);
      while (send_done) wait();
      ppu_done.write(false);
      DWAIT(1);
    }
    wait();
  }
}

#endif // VMM_COMPUTE_H
