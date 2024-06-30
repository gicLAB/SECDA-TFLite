void ACCNAME::Data_In() {
  init_wgts_VMM();
  wait();
  while (1) {
    while (!load_data.read()) wait();
    DWAIT(1);
    if (load_wgt) {
      wgt_packet wp = wgt_packet(&din1);
      wgt_len_VMM(wp.wgt_size);
      DWAIT(3);
      for (int i = 0; i < wp.wgt_size; i++) {
        ACC_DTYPE<32> data1 = din1.read().data.to_int();
        ACC_DTYPE<32> data2 = din2.read().data.to_int();
        ACC_DTYPE<32> data3 = din3.read().data.to_int();
        ACC_DTYPE<32> data4 = din4.read().data.to_int();
        sc_bigint<32 * 4> data;
        data.range(31, 0) = data1;
        data.range(63, 32) = data2;
        data.range(95, 64) = data3;
        data.range(127, 96) = data4;
        fill_wgts_VMM(data);
        DWAIT();
      }
      int ra = 0, rb = 0;
      for (int i = 0; i < wp.wgt_sum_size; i++) {
        ACC_DTYPE<32> wsums1 = din1.read().data.to_int();
        ACC_DTYPE<32> wsums2 = din2.read().data.to_int();
        ACC_DTYPE<32> wsums3 = din3.read().data.to_int();
        ACC_DTYPE<32> wsums4 = din4.read().data.to_int();
        // ACC_DTYPE<32> rfs1 = din1.read().data.to_int();
        // ACC_DTYPE<32> rfs2 = din2.read().data.to_int();
        // ACC_DTYPE<32> rfs3 = din3.read().data.to_int();
        // ACC_DTYPE<32> rfs4 = din4.read().data.to_int();

        ACC_DTYPE<32> rfs = din1.read().data.to_int();
        ACC_DTYPE<32> exs = din1.read().data.to_int();
        rb++;
        wgt_sum1[ra] = wsums1;
        wgt_sum2[ra] = wsums2;
        wgt_sum3[ra] = wsums3;
        wgt_sum4[ra] = wsums4;

        // crf1[ra] = rfs1;
        // crf2[ra] = rfs2;
        // crf3[ra] = rfs3;
        // crf4[ra] = rfs4;

        crf[ra] = rfs;
        crx[ra] = exs;
        ra = rb;
        DWAIT(3);
      }
    }

    DWAIT();
    if (load_inp) {
      inp_packet ip = inp_packet(&din1);
      int la = 0, lb = 0, ra = 0, rb = 0;
      for (int i = 0; i < ip.inp_size; i++) {
        ACC_DTYPE<32> data1 = din1.read().data.to_int();
        ACC_DTYPE<32> data2 = din2.read().data.to_int();
        ACC_DTYPE<32> data3 = din3.read().data.to_int();
        ACC_DTYPE<32> data4 = din4.read().data.to_int();
        rb++;
        inp_data1[ra] = data1;
        inp_data2[ra] = data2;
        inp_data3[ra] = data3;
        inp_data4[ra] = data4;
        ra = rb;
        DWAIT();
      }
      // for (int i = 0; i < ip.inp_sum_size; i++) {
      //   ACC_DTYPE<32> isums1 = din1.read().data.to_int();
      //   ACC_DTYPE<32> isums2 = din2.read().data.to_int();
      //   ACC_DTYPE<32> isums3 = din3.read().data.to_int();
      //   ACC_DTYPE<32> isums4 = din4.read().data.to_int();
      //   lb++;
      //   // inp_sum1[la] = isums1;
      //   // inp_sum2[la] = isums2;
      //   // inp_sum3[la] = isums3;
      //   // inp_sum4[la] = isums4;
      //   inp_sum1[la] = 0;
      //   inp_sum2[la] = 0;
      //   inp_sum3[la] = 0;
      //   inp_sum4[la] = 0;
      //   la = lb;
      //   DWAIT();
      // }
    }
    load_data.write(false);
    // wait();
    DWAIT();
  }
}
