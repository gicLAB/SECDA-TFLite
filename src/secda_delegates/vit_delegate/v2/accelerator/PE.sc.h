void ACCNAME::PE(int N_rem, int M_rem) {
  // The kinda PE_GEMM part
  for (int i = 0; i < N_rem; i++) { // Loop 1.1.2.3
  // cout << "N_rem = " << N_rem << endl;
// TODO: This pipeline does not work
#pragma loop_tripcount min = loopmax max = loopmax
#pragma HLS pipeline II = 1
    for (int j = 0; j < M_rem; j += 4) { // Loop 1.1.2.3.1
    // cout << "M_rem = " << M_rem << endl;
#pragma loop_tripcount min = loopqtmin max = loopqtmax
      int temp[pkF][4];
#pragma HLS array_partition variable = temp complete dim = 0
      for (int k = 0; k < pkF; k++) { /// Loop 1.1.2.3.1.1
#pragma loop_tripcount min = pkF max = pkF
#pragma HLS unroll
        temp[k][0] = 0;
        temp[k][1] = 0;
        temp[k][2] = 0;
        temp[k][3] = 0;
      }


      // SECTION: Effectively the entire performance is here
      for (int k = 0; k < pK; k += pkF) { // Loop 1.1.2.3.1.2
#pragma loop_tripcount min = pKT max = pKT
#pragma HLS pipeline II = 1
        // for row i and cols j, j+1, j+2, j+3, 
        for (int l = 0; l < (pkF / 2); l++) { // Loop 1.1.2.3.2.1.
          // int curRow = rows[i][k+l]; // TODO: This is different
          temp[l][0] += rows[i][k+l] * cols[j + 0][k + l];
          temp[l][1] += rows[i][k+l] * cols[j + 1][k + l];
          temp[l][2] += rows[i][k+l] * cols[j + 2][k + l];
          temp[l][3] += rows[i][k+l] * cols[j + 3][k + l];
        }
        for (int l = (pkF / 2); l < pkF; l++) {
          // int curRow = rows[i][k+l];
          temp[l][0] += mul_s8(rows[i][k+l], cols[j + 0][k + l]);
          temp[l][1] += mul_s8(rows[i][k+l], cols[j + 1][k + l]);
          temp[l][2] += mul_s8(rows[i][k+l], cols[j + 2][k + l]);
          temp[l][3] += mul_s8(rows[i][k+l], cols[j + 3][k + l]);
        }
      // END SECTION
      }
      for (int l = 1; l < pkF; l++) {
        #pragma loop_tripcount min = pkF-1 max = pkF-1
        temp[0][0] += temp[l][0];
        temp[0][1] += temp[l][1];
        temp[0][2] += temp[l][2];
        temp[0][3] += temp[l][3];
      }
      res[i][j + 0] += temp[0][0];
      res[i][j + 1] += temp[0][1];
      res[i][j + 2] += temp[0][2];
      res[i][j + 3] += temp[0][3];
      
      // PE_Post(i, j, crf, crx, ra);


    value1 = Quantised_Multiplier(res[i][j + 0], crf, crx);
    value2 = Quantised_Multiplier(res[i][j + 1], crf, crx);
    value3 = Quantised_Multiplier(res[i][j + 2], crf, crx);
    value4 = Quantised_Multiplier(res[i][j + 3], crf, crx);

    svalue1 = value1 + ra;
    svalue2 = value2 + ra;
    svalue3 = value3 + ra;
    svalue4 = value4 + ra;

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

    // cout << "cur outs = " << cur_outs[0] << " " << cur_outs[1] << " " << cur_outs[2] << " " << cur_outs[3] << endl;

    d_array[j / 4].data = Clamp_Combine(cur_outs[0], cur_outs[1], cur_outs[2],
                                                cur_outs[3], MAX8, MIN8);
    }
    for (int j = 0; j < (M_rem / 4); j++) {
      #pragma loop_tripcount min = loopqtmin max = loopqtmax
      if (i == (N_rem - 1) && j == (M_rem / 4) - 1) {
        d_array[j].tlast = true;
      } else d_array[j].tlast = false;
      dout1.write(d_array[j]);
    }

    // cout << "M_rem = " << M_rem << endl;
    // cout << "M_rem / 4 = " << M_rem / 4 << endl;

    // for (int j = 0; j < (M_rem / 4); j+=4) {
    //   // cout << "j = " << j << endl;
    //   d_array[j].tlast = false;
    //   d_array[j + 1].tlast = false;
    //   d_array[j + 2].tlast = false;
    //   dout1.write(d_array[j]);
    //   dout2.write(d_array[j + 1]);
    //   dout3.write(d_array[j + 2]);
    //   if (i == (N_rem - 1) && j == (M_rem / 4) - 4) {
    //     d_array[j+3].tlast = true;
        
    //   } else d_array[j+3].tlast = false;
    //   dout4.write(d_array[j + 3]);
    // }

    wait();
    // DWAIT(128); // Initiation interval
  }
  // DWAIT(N_rem);
  wait();
}
