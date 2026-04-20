void ACCNAME::PPU() {
  ppuReadyS.write(0);
  pePostTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {

    while (ppuReadyS.read() == 0) wait();

    pePostTotalS.write(1);

    int pnr = pN_rem;
    int pmr = pM_rem;


    int cur_ra = ra;

    DWAIT(3);

    for (int i = 0; i < pnr; i++) {
      DWAIT(1);
      for (int j = 0; j < pmr; j += 4) {
        // #pragma HLS pipeline II = 1
        // if (j < pmr) {

        res[i][j + 0] += prec[i][j + 0];
        res[i][j + 1] += prec[i][j + 1];
        res[i][j + 2] += prec[i][j + 2];
        res[i][j + 3] += prec[i][j + 3];

        wait();
        if (layer_t == 1) {
          wait();
          wait();
          // CONV layer - Use local variables to avoid array access conflicts
          int local_crf_0 = crf_v[j + 0];
          int local_crf_1 = crf_v[j + 1]; 
          int local_crf_2 = crf_v[j + 2];
          int local_crf_3 = crf_v[j + 3];
          sc_int<8> local_crx_0 = crx_v[j + 0];
          sc_int<8> local_crx_1 = crx_v[j + 1];
          sc_int<8> local_crx_2 = crx_v[j + 2]; 
          sc_int<8> local_crx_3 = crx_v[j + 3];
          wait();
          // CONV layer - Add extra synchronization between multiplier calls
          value1 = Quantised_Multiplier(
              res[i][j + 0], local_crf_0, local_crx_0);
          wait();
          wait();
          wait(); // Extra wait for stabilization
          value2 = Quantised_Multiplier(
              res[i][j + 1], local_crf_1, local_crx_1);
          wait();
          wait();
          wait(); // Extra wait for stabilization
          value3 = Quantised_Multiplier(
              res[i][j + 2], local_crf_2, local_crx_2);
          wait();
          wait();
          wait(); // Extra wait for stabilization
          value4 = Quantised_Multiplier(
              res[i][j + 3], local_crf_3, local_crx_3);
          wait();
          wait();
          wait(); // Extra wait for stabilization



        } else {
          // FC layer
          // value1 = Quantised_Multiplier(res[i][j + 0], crf, crx);
          // value2 = Quantised_Multiplier(res[i][j + 1], crf, crx);
          // value3 = Quantised_Multiplier(res[i][j + 2], crf, crx);
          // value4 = Quantised_Multiplier(res[i][j + 3], crf, crx);
          wait();
        }
        wait(); // a

        svalue1 = value1 + cur_ra;
        svalue2 = value2 + cur_ra;
        svalue3 = value3 + cur_ra;
        svalue4 = value4 + cur_ra;

        wait();

        if (svalue1 > MAX8) svalue1 = MAX8;
        else if (svalue1 < MIN8) svalue1 = MIN8;
        if (svalue2 > MAX8) svalue2 = MAX8;
        else if (svalue2 < MIN8) svalue2 = MIN8;
        if (svalue3 > MAX8) svalue3 = MAX8;
        else if (svalue3 < MIN8) svalue3 = MIN8;
        if (svalue4 > MAX8) svalue4 = MAX8;
        else if (svalue4 < MIN8) svalue4 = MIN8;

        wait(); // a

        dout_1 = svalue1.range(7, 0);
        dout_2 = svalue2.range(7, 0);
        dout_3 = svalue3.range(7, 0);
        dout_4 = svalue4.range(7, 0);

        wait(); // a;

        res[i][j + 0] = 0;
        res[i][j + 1] = 0;
        res[i][j + 2] = 0;
        res[i][j + 3] = 0;

        d_array[j / 4].data =
            Clamp_Combine(dout_1, dout_2, dout_3, dout_4, MAX8, MIN8);
        DWAIT(32);
        wait(); // a
        // }
      }
      wait(); // a

      for (int j = 0; j < pmQ; j++) {
#pragma HLS pipeline II = 1
        if (j < (pmr / 4)) {
          if (i == (pnr - 1) && j == (pmr / 4) - 1) {
            d_array[j].tlast = true;
          } else d_array[j].tlast = false;
          dout1.write(d_array[j]);
        }
      }
    }
    wait();
    ppuReadyS.write(0);
    pePostTotalS.write(0);
    wait(); // DO NOT REMOVE THIS IT BREAKS EVERYTHING
  }
}