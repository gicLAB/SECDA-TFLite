#if defined(QKERAS)

// 8x4 bit non - uniform multiplication - QKeras or
//     META 15 - bit intermediate result

sc_int<PROD_DATA_WIDTH> VMM_UNIT::mul_s8(sc_int<4> wgt, sc_int<8> inp) {
  // #pragma HLS inline off
  sc_int<PROD_DATA_WIDTH> c15 = 0;

  c15 = inp;
  c15 = c15 << wgt.range(2, 0);

  // cout << "inp: " << (int)inp << " wgt: " << (int)wgt.range(2,0) << " prod="
  // << (int)c15 << endl;

  sc_int<PROD_DATA_WIDTH> result = c15;
  return result;
}

// 8x4 bit non-uniform multiplication - MSQ
// 12-bit intermediate result

#elif defined(MSQ)

sc_int<PROD_DATA_WIDTH> VMM_UNIT::mul_s8(sc_int<4> wgt, sc_int<8> inp) {
  sc_int<11> c11 = 0;
  // dealing with first PoT term
  if (wgt.range(2, 1) == 0) {
    c11 = 0;
  } else // shift by 1 or 2 or 3
  {
    c11 = inp;
    c11 = c11 << wgt.range(2, 1);
  }

  sc_int<9> c9 = 0;
  // dealing with second PoT term
  if (wgt.range(0, 0) == 0) {
    c9 = 0;
  } else {
    c9 = inp;
    c9 = c9 << 1;
    // c9 = c9 << wgt.range(0, 0);
  }

  sc_int<PROD_DATA_WIDTH> result = c11 + c9;
  return result;
}

#elif defined(APOT)

// // 8x4 bit non-uniform multiplication - APoT
// // 14-bit intermediate result
sc_int<PROD_DATA_WIDTH> VMM_UNIT::mul_s8(sc_int<4> wgt, sc_int<8> inp) {

  sc_int<12> c12 = 0;
  // dealing with first PoT term
  if (wgt.range(2, 1) == 0) {
    c12 = 0;
  } else if (wgt.range(2, 1) == 3) {
    c12 = inp;
    c12 = c12 << 4;
  } else // shift by 1 or 2
  {
    c12 = inp;
    c12 = c12 << wgt.range(2, 1);
  }

  sc_int<11> c11 = 0;
  // dealing with second PoT term
  if (wgt.range(0, 0) == 0) {
    c11 = 0;
  } else {
    c11 = inp;
    c11 = c11 << 3;
  }

  sc_int<PROD_DATA_WIDTH> result = c12 + c11;
  return result;
}

#else
// 8x4 bit uniform multiplication
// 12-bit intermediate result
sc_int<PROD_DATA_WIDTH> VMM_UNIT::mul_s8(sc_int<4> wgt, sc_int<8> inp) {
  sc_int<PROD_DATA_WIDTH> c = wgt * inp;
#pragma HLS RESOURCE variable = c core = Mul_LUT

  return c;
}

#endif

void VMM_UNIT::VM_PE(ACC_DTYPE<WGT_BRAM_DATAWIDTH> *l1,
                     ACC_DTYPE<WGT_BRAM_DATAWIDTH> *l2,
                     ACC_DTYPE<WGT_BRAM_DATAWIDTH> *l3,
                     ACC_DTYPE<WGT_BRAM_DATAWIDTH> *l4, ACC_DTYPE<32> *r1,
                     ACC_DTYPE<32> *r2, ACC_DTYPE<32> *r3, ACC_DTYPE<32> *r4,
                     ACC_DTYPE<32> out[][4], int d, int w_idx, int wID) {
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_read[4];
  ACC_DTYPE<32> inp_read[4];
  ACC_DTYPE<8> in_a[8];
  ACC_DTYPE<8> in_b[8];
  ACC_DTYPE<8> we_a[8];
  ACC_DTYPE<8> we_b[8];
  ACC_DTYPE<PROD_DATA_WIDTH> prod[16][4];

#pragma HLS array_partition variable = wgt_read complete dim = 0
#pragma HLS array_partition variable = inp_read complete dim = 0
#pragma HLS array_partition variable = in_a complete dim = 0
#pragma HLS array_partition variable = in_b complete dim = 0
#pragma HLS array_partition variable = we_a complete dim = 0
#pragma HLS array_partition variable = we_b complete dim = 0
#pragma HLS array_partition variable = prod complete dim = 0

  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    for (int j = 0; j < 16; j++) {
#pragma HLS unroll
      out[j][i] = 0;
    }
  }

  for (int rin = 0; rin < d; rin++) {
#pragma HLS loop_tripcount min = 64 max = 64 avg = 64
#pragma HLS pipeline II = 1
    wgt_read[0] = l1[rin + w_idx];
    wgt_read[1] = l2[rin + w_idx];
    wgt_read[2] = l3[rin + w_idx];
    wgt_read[3] = l4[rin + w_idx];
    inp_read[0] = r1[rin];
    inp_read[1] = r2[rin];
    inp_read[2] = r3[rin];
    inp_read[3] = r4[rin];
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      we_a[i + 0] = wgt_read[i].range(((WGT_BRAM_DATAWIDTH / 4) * 1) - 1,
                                      (WGT_BRAM_DATAWIDTH / 4) * 0);
      we_a[i + 4] = wgt_read[i].range(((WGT_BRAM_DATAWIDTH / 4) * 2) - 1,
                                      (WGT_BRAM_DATAWIDTH / 4) * 1);
      we_b[i + 0] = wgt_read[i].range(((WGT_BRAM_DATAWIDTH / 4) * 3) - 1,
                                      (WGT_BRAM_DATAWIDTH / 4) * 2);
      we_b[i + 4] = wgt_read[i].range(((WGT_BRAM_DATAWIDTH / 4) * 4) - 1,
                                      (WGT_BRAM_DATAWIDTH / 4) * 3);

      in_a[i + 0] = inp_read[i].range(7, 0);
      in_a[i + 4] = inp_read[i].range(15, 8);
      in_b[i + 0] = inp_read[i].range(23, 16);
      in_b[i + 4] = inp_read[i].range(31, 24);
    }
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll

      // assignin products into the LUT only
      prod[i * 4 + 0][0] = mul_s8(we_a[0 * 4 + i], in_a[0 * 4 + 0]);
      prod[i * 4 + 1][0] = mul_s8(we_a[0 * 4 + i], in_a[0 * 4 + 1]);
      prod[i * 4 + 2][0] = mul_s8(we_a[0 * 4 + i], in_a[0 * 4 + 2]);
      prod[i * 4 + 3][0] = mul_s8(we_a[0 * 4 + i], in_a[0 * 4 + 3]);
      prod[i * 4 + 0][1] = mul_s8(we_a[1 * 4 + i], in_a[1 * 4 + 0]);
      prod[i * 4 + 1][1] = mul_s8(we_a[1 * 4 + i], in_a[1 * 4 + 1]);
      prod[i * 4 + 2][1] = mul_s8(we_a[1 * 4 + i], in_a[1 * 4 + 2]);
      prod[i * 4 + 3][1] = mul_s8(we_a[1 * 4 + i], in_a[1 * 4 + 3]);
      prod[i * 4 + 0][2] = mul_s8(we_b[0 * 4 + i], in_b[0 * 4 + 0]);
      prod[i * 4 + 1][2] = mul_s8(we_b[0 * 4 + i], in_b[0 * 4 + 1]);
      prod[i * 4 + 2][2] = mul_s8(we_b[0 * 4 + i], in_b[0 * 4 + 2]);
      prod[i * 4 + 3][2] = mul_s8(we_b[0 * 4 + i], in_b[0 * 4 + 3]);
      prod[i * 4 + 0][3] = mul_s8(we_b[1 * 4 + i], in_b[1 * 4 + 0]);
      prod[i * 4 + 1][3] = mul_s8(we_b[1 * 4 + i], in_b[1 * 4 + 1]);
      prod[i * 4 + 2][3] = mul_s8(we_b[1 * 4 + i], in_b[1 * 4 + 2]);
      prod[i * 4 + 3][3] = mul_s8(we_b[1 * 4 + i], in_b[1 * 4 + 3]);
    }

    // cout << endl << endl;
    // exit(0);
    // int dd = 0;
    for (int i = 0; i < 16; i++) {
#pragma HLS unroll
      if (we_a[0 * 4 + (i >> 2)].range(3, 3)) out[i][0] -= prod[i][0];
      else out[i][0] += prod[i][0];
      if (we_a[1 * 4 + (i >> 2)].range(3, 3)) out[i][1] -= prod[i][1];
      else out[i][1] += prod[i][1];
      if (we_b[0 * 4 + (i >> 2)].range(3, 3)) out[i][2] -= prod[i][2];
      else out[i][2] += prod[i][2];
      if (we_b[1 * 4 + (i >> 2)].range(3, 3)) out[i][3] -= prod[i][3];
      else out[i][3] += prod[i][3];
      // out[i][0] += prod[i][0];
      // out[i][1] += prod[i][1];
      // out[i][2] += prod[i][2];
      // out[i][3] += prod[i][3];
    }
  }
  DWAIT(5 + d);
  for (int i = 0; i < 16; i++) {
#pragma HLS unroll
    out[i][0] += out[i][1] + out[i][2] + out[i][3];
    // out[i][0] += l1[i] * r1[i];
  }
  DWAIT(2);
}
