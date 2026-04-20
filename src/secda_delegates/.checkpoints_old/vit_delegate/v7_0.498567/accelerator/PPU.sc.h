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
    if (layer_t == 0) {
      // sc_int<32> cur_crf = crf;
      // sc_int<8> cur_crx = crx;
    }

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

          int local_crf[4];
          sc_int<8> local_crx[4];
          int local_res[4];
          ACC_DTYPE<64> local_pl[4];
          ACC_DTYPE<32> local_pr[4];
          ACC_DTYPE<32> local_msk[4];
          ACC_DTYPE<32> local_sm[4];
#pragma HLS array_partition variable = local_crf complete dim = 0
#pragma HLS array_partition variable = local_crx complete dim = 0
#pragma HLS array_partition variable = local_res complete dim = 0
#pragma HLS array_partition variable = local_pl complete dim = 0
#pragma HLS array_partition variable = local_pr complete dim = 0
#pragma HLS array_partition variable = local_msk complete dim = 0
#pragma HLS array_partition variable = local_sm complete dim = 0

          for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
            local_crf[c] = crf_v[j + c];
            local_crx[c] = crx_v[j + c];
            local_res[c] = res[i][j + c];
          }

          for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
            if (local_crx[c] > 0) {
              local_pl[c] = local_crx[c];
              local_pr[c] = 0;
              local_msk[c] = 0;
              local_sm[c] = 0;
            } else {
              local_pl[c] = 1;
              local_pr[c] = -local_crx[c];
              local_msk[c] = (1 << -local_crx[c]) - 1;
              local_sm[c] = local_msk[c] >> 1;
            }
          }

          wait();



          value1 = Quantised_Multiplier(local_res[0], local_crf[0], local_pl[0],
                                        local_pr[0], local_msk[0], local_sm[0]);
          value2 = Quantised_Multiplier(local_res[1], local_crf[1], local_pl[1],
                                        local_pr[1], local_msk[1], local_sm[1]);
          value3 = Quantised_Multiplier(local_res[2], local_crf[2], local_pl[2],
                                        local_pr[2], local_msk[2], local_sm[2]);
          value4 = Quantised_Multiplier(local_res[3], local_crf[3], local_pl[3],
                                        local_pr[3], local_msk[3], local_sm[3]);
          wait();


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