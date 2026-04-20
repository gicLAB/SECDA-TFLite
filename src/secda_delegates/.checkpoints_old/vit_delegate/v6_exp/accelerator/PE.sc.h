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
  sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
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
  ACC_DTYPE<32> d;
  d.range(7, 0) = i1;
  d.range(15, 8) = i2;
  d.range(23, 16) = i3;
  d.range(31, 24) = i4;
  return d;
}

void ACCNAME::PE() {
  peReadyS.write(0);
  peTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (peReadyS.read() == 0) wait();

    wait(); // a

    // cout << "PE START" << endl;

    int pe_pnr = pN_rem;
    int pe_pmr = pM_rem;
    int pe_pkr = pK_rem;
    wait(); // a

    peTotalS.write(1);
    DWAIT(1);
    for (int i = 0; i < pe_pnr; i++) {
      for (int j = 0; j < pe_pmr; j += 4) {
        for (int k = 0; k < pe_tile; k++) {
#pragma HLS unroll
          temp[k][0] = 0;
          temp[k][1] = 0;
          temp[k][2] = 0;
          temp[k][3] = 0;
        }
        DWAIT(1);
        wait(); // a

        for (int k = 0; k < pe_pkr; k += pe_tile) {
#pragma loop_tripcount min = (pe_tile / 2) max = (pe_tile / 2)
#pragma HLS pipeline II = 1
          for (int l = 0; l < (pe_tile / 2); l++) {
            int curRow = rows[i][k + l];
            temp[l][0] += curRow * cols[j + 0][k + l];
            temp[l][1] += curRow * cols[j + 1][k + l];
            temp[l][2] += curRow * cols[j + 2][k + l];
            temp[l][3] += curRow * cols[j + 3][k + l];
          }
          for (int l = (pe_tile / 2); l < pe_tile; l++) {
            int curRow = rows[i][k + l];
            temp[l][0] += mul_s8(curRow, cols[j + 0][k + l]);
            temp[l][1] += mul_s8(curRow, cols[j + 1][k + l]);
            temp[l][2] += mul_s8(curRow, cols[j + 2][k + l]);
            temp[l][3] += mul_s8(curRow, cols[j + 3][k + l]);
          }
          DWAIT(5);
        }
        wait(); // a

        for (int l = 1; l < pe_tile; l++) {
#pragma HLS unroll
          temp[0][0] += temp[l][0];
          temp[0][1] += temp[l][1];
          temp[0][2] += temp[l][2];
          temp[0][3] += temp[l][3];
          // DWAIT(4); // ! Not sure if this one should be here or the one below
          wait(); // a
        }
        DWAIT(4);
        wait(); // a
        res[i][j + 0] += temp[0][0];
        res[i][j + 1] += temp[0][1];
        res[i][j + 2] += temp[0][2];
        res[i][j + 3] += temp[0][3];
        DWAIT(1);
        wait(); // a
      }
    }
    wait(); // a
    DWAIT(1);
    // cout << "PE END" << endl;
    peReadyS.write(0);
    peTotalS.write(0);
    wait();
  }
}
