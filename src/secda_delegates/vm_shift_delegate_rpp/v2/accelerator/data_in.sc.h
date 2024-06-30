void ACCNAME::Data_In() {
  init_wgts_VMM();
  w2SS.write(0);
  w3SS.write(0);

  wait();
  while (1) {
    w1SS.write(1);
    while (!load_data.read()) wait();
    DWAIT(1);
    if (load_wgt) {
      w1SS.write(2);
      wgt_packet wp = wgt_packet(&din1);
      wgt_len_VMM(wp.wgt_size);
      DWAIT(3);
      w1SS.write(21);
      w2SS.write(wp.wgt_size);
      w3SS.write(wp.wgt_sum_size);
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
        // if (i==16)
        // {
        //   cout << "special data_in" << endl;
        // }
        // cout << "data_in batch no: " << i << endl;
        // cout << (int)data1.range(2, 0) << " " << (int)data1.range (6, 4) << " " << (int)data1.range(10, 8) << " " << (int)data1.range(14, 12) << " " << (int)data1.range(18, 16) << " " << (int)data1.range(22, 20) << " " << (int)data1.range(26, 24) << " " << (int)data1.range(30, 28) << endl;

        // cout << (int)data2.range(2, 0) << " " << (int)data2.range (6, 4) << " " << (int)data2.range(10, 8) << " " << (int)data2.range(14, 12) << " " << (int)data2.range(18, 16) << " " << (int)data2.range(22, 20) << " " << (int)data2.range(26, 24) << " " << (int)data2.range(30, 28) << endl;

        // cout << (int)data3.range(2, 0) << " " << (int)data3.range (6, 4) << " " << (int)data3.range(10, 8) << " " << (int)data3.range(14, 12) << " " << (int)data3.range(18, 16) << " " << (int)data3.range(22, 20) << " " << (int)data3.range(26, 24) << " " << (int)data3.range(30, 28) << endl;

        // cout << (int)data4.range(2, 0) << " " << (int)data4.range (6, 4) << " " << (int)data4.range(10, 8) << " " << (int)data4.range(14, 12) << " " << (int)data4.range(18, 16) << " " << (int)data4.range(22, 20) << " " << (int)data4.range(26, 24) << " " << (int)data4.range(30, 28) << endl;

        
        // exit (0);

        fill_wgts_VMM(data);
        DWAIT();
      }
      int ra = 0, rb = 0;
      w1SS.write(22);
      for (int i = 0; i < wp.wgt_sum_size; i++) {
        ACC_DTYPE<32> wsums1 = din1.read().data.to_int();
        ACC_DTYPE<32> wsums2 = din2.read().data.to_int();
        ACC_DTYPE<32> wsums3 = din3.read().data.to_int();
        ACC_DTYPE<32> wsums4 = din4.read().data.to_int();
        ACC_DTYPE<32> rfs1 = din1.read().data.to_int();
        ACC_DTYPE<32> rfs2 = din2.read().data.to_int();
        ACC_DTYPE<32> rfs3 = din3.read().data.to_int();
        ACC_DTYPE<32> rfs4 = din4.read().data.to_int();
        ACC_DTYPE<32> exs = din1.read().data.to_int();
        rb++;
        wgt_sum1[ra] = wsums1;
        wgt_sum2[ra] = wsums2;
        wgt_sum3[ra] = wsums3;
        wgt_sum4[ra] = wsums4;
        crf1[ra] = rfs1;
        crf2[ra] = rfs2;
        crf3[ra] = rfs3;
        crf4[ra] = rfs4;
        crx[ra] = exs;
        ra = rb;
        DWAIT(3);
      }
    }

    DWAIT();
    if (load_inp) {
      w1SS.write(3);
      inp_packet ip = inp_packet(&din1);
      int la = 0, lb = 0, ra = 0, rb = 0;
      w2SS.write(ip.inp_size);
      w3SS.write(ip.inp_sum_size);
      w1SS.write(31);
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
