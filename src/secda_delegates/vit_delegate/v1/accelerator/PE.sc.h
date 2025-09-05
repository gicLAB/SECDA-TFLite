void ACCNAME::PE() {
  // peTotalS.write(1);
  for (int i = 0; i < pN_rem; i++) { // Loop 1.1.2.3
    for (int j = 0; j < pM_rem; j += 4) {
      DWAIT(2);
      for (int k = 0; k < pkF; k++) {
#pragma HLS unroll
        temp[k][0] = 0;
        temp[k][1] = 0;
        temp[k][2] = 0;
        temp[k][3] = 0;
      }
      DWAIT(1);

      // peMultiplyS.write(1);
      for (int k = 0; k < pK; k += pkF) { // Loop 1.1.2.3.1.1
#pragma loop_tripcount min = pKT max = pKT
#pragma HLS pipeline II = 1
        // for row i and cols j, j+1, j+2, j+3,
        for (int l = 0; l < (pkF / 2); l++) {
          int curRow = rows[i][k + l];
          temp[l][0] += curRow * cols[j + 0][k + l];
          temp[l][1] += curRow * cols[j + 1][k + l];
          temp[l][2] += curRow * cols[j + 2][k + l];
          temp[l][3] += curRow * cols[j + 3][k + l];
        }
        for (int l = (pkF / 2); l < pkF; l++) {
          int curRow = rows[i][k + l];
          temp[l][0] += mul_s8(curRow, cols[j + 0][k + l]);
          temp[l][1] += mul_s8(curRow, cols[j + 1][k + l]);
          temp[l][2] += mul_s8(curRow, cols[j + 2][k + l]);
          temp[l][3] += mul_s8(curRow, cols[j + 3][k + l]);
        }
        // END SECTION
        DWAIT(5);
      }
      // peMultiplyS.write(0);
      // peAccumulateS.write(1);
      for (int l = 1; l < pkF; l++) {
#pragma HLS unroll
        temp[0][0] += temp[l][0];
        temp[0][1] += temp[l][1];
        temp[0][2] += temp[l][2];
        temp[0][3] += temp[l][3];
        // DWAIT(4);
      }
      DWAIT(4);
      res[i][j + 0] += temp[0][0];
      res[i][j + 1] += temp[0][1];
      res[i][j + 2] += temp[0][2];
      res[i][j + 3] += temp[0][3];
      DWAIT(1);
      // peAccumulateS.write(0);
    }
    // wait();
    // DWAIT(128);
  }
  // peTotalS.write(0);
  // wait();
}
