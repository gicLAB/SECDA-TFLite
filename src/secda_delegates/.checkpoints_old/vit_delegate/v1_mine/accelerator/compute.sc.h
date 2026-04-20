/*
CORRECT ONE
Pragmas:
Unrolls at each loop
Paritioning arrays accross dimension 0 for all arrays
  > res
  > rows
  > cols
  > cur_outs
Should I pipeline? What?
  > first k loop
*/

// #include "acc.sc.h"

#ifndef __SYNTHESIS__
int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1);
  sc_int<64> result = x_64 * quantized_multiplier_64 + round;
  result = result >> total_shift;
  int nresult = result;
  if (result > MAX32) result = MAX32;
  if (result < MIN32) result = MIN32;
  sc_int<32> result_32 = result;
  return result_32;
}
#else
int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
  sc_int<64> val = x * pl;
  if (val > MAX32) val = MAX32;
  if (val < MIN32) val = MIN32;
  sc_int<64> val_2 = val * qm;
  sc_int<32> temp_1;
  temp_1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
  sc_int<32> val_3 = temp_1;
  val_3 = val_3 >> pr;
  sc_int<32> temp_2 = temp_1 & msk;
  sc_int<32> temp_3 = (temp_1 < 0) & 1;
  sc_int<32> temp_4 = sm + temp_3;
  sc_int<32> temp_5 = ((temp_2 >= temp_4) & 1);
  sc_int<32> result_32 = val_3 + temp_5;
  return result_32;
}
#endif

sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) {
  sc_int<32> c;
#pragma HLS RESOURCE variable = c core = Mul
  c = a * b;
  return c;
}

ACC_DTYPE<32> ACCNAME::Clamp_Combine(int i1, int i2, int i3, int i4, int qa_max,
                                     int qa_min) {
  if (i1 < qa_min) i1 = qa_min;
  if (i1 > qa_max) i1 = qa_max;
  if (i2 < qa_min) i2 = qa_min;
  if (i2 > qa_max) i2 = qa_max;
  if (i3 < qa_min) i3 = qa_min;
  if (i3 > qa_max) i3 = qa_max;
  if (i4 < qa_min) i4 = qa_min;
  if (i4 > qa_max) i4 = qa_max;
  ACC_DTYPE<32> d;
  d.range(7, 0) = i1;
  d.range(15, 8) = i2;
  d.range(23, 16) = i3;
  d.range(31, 24) = i4;
  return d;
}

void ACCNAME::PE(int N_rem, int M_rem) {
  // cout << "pnrem = " << N_rem << " pmrem = " << M_rem << endl;
  for (int i = 0; i < N_rem; i++) { // Loop 1.1.2.3

#pragma HLS pipeline II = 1
    // should this be at this loop or the next loop?

    for (int j = 0; j < M_rem; j += 4) { // Loop 1.1.2.3.1
      // #pragma HLS unroll factor = 4
      int temp[pkF][4];
#pragma HLS array_partition variable = temp complete dim = 0
      for (int k = 0; k < pkF; k++) {
#pragma HLS unroll
        temp[k][0] = 0;
        temp[k][1] = 0;
        temp[k][2] = 0;
        temp[k][3] = 0;
      }

      for (int k = 0; k < pK; k += pkF) { // Loop 1.1.2.3.1.1
        // TODO: This is where I kinda get lost with the dwaits, the graph is
        // confusing and moving around all over the place.
#pragma loop_tripcount min = pKT max = pKT
#pragma HLS pipeline II = 1
        for (int l = 0; l < (pkF / 2); l++) {
          temp[l][0] += rows[i][k + l] * cols[j + 0][k + l];
          temp[l][1] += rows[i][k + l] * cols[j + 1][k + l];
          temp[l][2] += rows[i][k + l] * cols[j + 2][k + l];
          temp[l][3] += rows[i][k + l] * cols[j + 3][k + l];
        }
        for (int l = (pkF / 2); l < pkF; l++) {
          temp[l][0] += mul_s8(rows[i][k + l], cols[j + 0][k + l]);
          temp[l][1] += mul_s8(rows[i][k + l], cols[j + 1][k + l]);
          temp[l][2] += mul_s8(rows[i][k + l], cols[j + 2][k + l]);
          temp[l][3] += mul_s8(rows[i][k + l], cols[j + 3][k + l]);
        }
        wait();
      }
      for (int l = 1; l < pkF; l++) {
        temp[0][0] += temp[l][0];
        temp[0][1] += temp[l][1];
        temp[0][2] += temp[l][2];
        temp[0][3] += temp[l][3];
      }
      res[i][j + 0] += temp[0][0];
      res[i][j + 1] += temp[0][1];
      res[i][j + 2] += temp[0][2];
      res[i][j + 3] += temp[0][3];
      wait();

      int value1 = Quantised_Multiplier(res[i][j + 0], crf, crx);
      int value2 = Quantised_Multiplier(res[i][j + 1], crf, crx);
      int value3 = Quantised_Multiplier(res[i][j + 2], crf, crx);
      int value4 = Quantised_Multiplier(res[i][j + 3], crf, crx);
      sc_int<64> svalue1 = value1 + ra;
      sc_int<64> svalue2 = value2 + ra;
      sc_int<64> svalue3 = value3 + ra;
      sc_int<64> svalue4 = value4 + ra;
      if (svalue1 > MAX8) svalue1 = MAX8;
      if (svalue2 > MAX8) svalue2 = MAX8;
      if (svalue3 > MAX8) svalue3 = MAX8;
      if (svalue4 > MAX8) svalue4 = MAX8;
      if (svalue1 < MIN8) svalue1 = MIN8;
      if (svalue2 < MIN8) svalue2 = MIN8;
      if (svalue3 < MIN8) svalue3 = MIN8;
      if (svalue4 < MIN8) svalue4 = MIN8;
      cur_outs[0] = svalue1.range(7, 0);
      cur_outs[1] = svalue2.range(7, 0);
      cur_outs[2] = svalue3.range(7, 0);
      cur_outs[3] = svalue4.range(7, 0);
      // cout << "(v1) cur_outs: " << cur_outs[0] << " " << cur_outs[1] << " "
      //      << cur_outs[2] << " " << cur_outs[3] << endl;
      d_array[j / 4].data = Clamp_Combine(cur_outs[0], cur_outs[1], cur_outs[2],
                                          cur_outs[3], MAX8, MIN8);
    }
    for (int j = 0; j < (M_rem / 4); j++) {
      if (i == (N_rem - 1) && j == (M_rem / 4) - 1) {
        d_array[j].tlast = true;
      } else d_array[j].tlast = false;
      dout.write(d_array[j]);
    }
    wait();
    // DWAIT(128);
  }
  // DWAIT(131 + N_rem); // Where x is the number of cycles it takes and n_rem is
                      // the number of rows
  // x + n_rem
  // DWAIT(N_rem);
  wait();
}

void ACCNAME::Compute() {
#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = computeSS

  computeSS.write(0);
  cout << "pN, pM, pK, " << endl;

  wait();
  while (1) { // Loop 1
    readS.write(1);
    pN = din->read().data;
    pM = din->read().data;
    pK = din->read().data;
    // cout << pN << "," << pM << "," << pK << endl;

    crx = din->read().data;
    crf = din->read().data;
    ra = din->read().data;
    no_rows = din->read().data;
    no_cols = din->read().data;

    if (crx > 0) {
      pl = crx;
      pr = 0;
      msk = 0;
      sm = 0;
    } else {
      pl = 1;
      pr = -crx;
      msk = (1 << -crx) - 1;
      sm = msk >> 1;
    }
    // DWAIT(12); // 12 cycles before we get to the first reading loop - to my
               // understanding this is completely constant
    int tpk = pK;
    for (int n = 0; n < pN; n += no_rows) { // Loop 1.1
      int pN_rem = din->read().data;
      DWAIT(1);
      for (int i = 0; i < pN_rem; i++) { // Loop 1.1.1
        DWAIT(2);
        for (int k = 0; k < pK; k += 4) { // Loop 1.1.1.1
          DATA d = din->read();
          rows[i][k + 0] = d.data.range(7, 0);
          rows[i][k + 1] = d.data.range(15, 8);
          rows[i][k + 2] = d.data.range(23, 16);
          rows[i][k + 3] = d.data.range(31, 24);

          DWAIT(3);
        }
      }

      for (int m = 0; m < pM; m += no_cols) { // Loop 1.1.2
        int pM_rem = din->read().data;
        DWAIT(1);
        for (int i = 0; i < pM_rem; i++) { // Loop 1.1.2.1
          DWAIT(2);
          for (int k = 0; k < pK; k += 4) { // Loop 1.1.2.1.1
            DATA d = din->read();
            cols[i][k + 0] = d.data.range(7, 0);
            cols[i][k + 1] = d.data.range(15, 8);
            cols[i][k + 2] = d.data.range(23, 16);
            cols[i][k + 3] = d.data.range(31, 24);
            DWAIT(3); 
          }
        }

        int c = 0;
        // load bias
        for (int a = 0; a < pN_rem; a++) { // Loop 1.1.2.2
          DWAIT(1);
          for (int b = 0; b < pM_rem; b++) { // Loop 1.1.2.2.1
            res[a][b] = din->read().data;
            DWAIT(2); 
          }
        }

        wait();
        readS.write(0);
        PE(pN_rem, pM_rem);
        readS.write(1);
        wait();
      }
      wait();
    }
    wait();

    DWAIT();
  }
}