void ACCNAME::Data_In() {
  init_wgts_VMM();
  wait();
  while (1) {
    while (!load_data.read()) wait();
    DWAIT(1);
    if (load_wgt_drv) {
      wgt_packet wp = wgt_packet(&din1);

      // if we use if condition for this line it will not synthesize
      // cout << "wp.depth_switch: " << wp.depth_switch
      //      << " wp.colnoDiv4: " << wp.colnoDiv4
      //      << " wp.depthDiv4: " << wp.depthDiv4
      //      << " wp.minWgtBlockNo: " << wp.minWgtBlockNo
      //      << " wp.getExtraWgtBlock: " << wp.getExtraWgtBlock << endl;

      // for (int i = 0; i < VMM_COUNT; i++) {
      //   cout << "wp.wgtBlockArray[" << i << "]: " << wp.wgtBlockArray[i]
      //        << endl;
      // }

      // wgt_len_VMM(wp.depth_switch, wp.minWgtBlockNo, wp.getExtraWgtBlock,
      //             wp.depthDiv4);
      // wgt_len_VMM_arr(wp.depth_switch, wp.wgtBlockArray, wp.depthDiv4);
      wgt_len_VMM_arr2(wp.depth_switch, wp.wgtBlockArray, wp.loadWgtArr,
                       wp.depthDiv4);

      DWAIT(3);

      // var0, var1, var2, var3
      //  L1,   L5,   L9,   L13
      //  L2,   L6,   L10,  L14
      //  L3,   L7,   L11,  L15
      //  L4,   L8,   L12,  L16

      // broadcast mode
      // var0, var1, var2, var3
      //  L1,   L1,   L1,   L1
      //  L2,   L2,   L2,   L2
      //  L3,   L3,   L3,   L3
      //  L4,   L4,   L4,   L4

      for (int i = 0; i < wp.colnoDiv4; i++) {
        for (int j = 0; j < wp.depthDiv4; j++) {
          ACC_DTYPE<32> data1 = din1.read().data.to_int();
          ACC_DTYPE<32> data2 = din2.read().data.to_int();
          ACC_DTYPE<32> data3 = din3.read().data.to_int();
          ACC_DTYPE<32> data4 = din4.read().data.to_int();
          sc_bigint<32 * 4> data;
          data.range(31, 0) = data1;
          data.range(63, 32) = data2;
          data.range(95, 64) = data3;
          data.range(127, 96) = data4;

          // check depth is done for one line
          int var_counter = i % VMM_COUNT;
          var_counter = var_counter + unit_counter;
          if (var_counter >= VMM_COUNT) {
            var_counter = var_counter - VMM_COUNT;
          }
          // cout << "var_counter: " << var_counter
          //      << " unit_counter: " << unit_counter << endl;
          // DLOG << "wgtdata1: " << data1 << endl;
          // DLOG << "wgtdata2: " << data2 << endl;
          // DLOG << "wgtdata3: " << data3 << endl;
          // DLOG << "wgtdata4: " << data4 << endl;
          fill_wgts_VMM_individually(data, var_counter);

          DWAIT();
        }
      }
      // cout << "wgt input done" << endl;
      // wsum_len_VMM(wp.minWgtBlockNo, wp.getExtraWgtBlock);
      // wsum_len_VMM_arr(wp.wgtBlockArray);
      wsum_len_VMM_arr2(wp.wgtBlockArray, wp.loadWgtArr);

      DWAIT(3);
      int ra = 0, rb = 0;
      for (int i = 0; i < wp.colnoDiv4; i++) {
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

        sc_bigint<32 * 4> wsumdata;
        sc_bigint<32 * 4> crfdata;
        wsumdata.range(31, 0) = wsums1;
        wsumdata.range(63, 32) = wsums2;
        wsumdata.range(95, 64) = wsums3;
        wsumdata.range(127, 96) = wsums4;

        crfdata.range(31, 0) = rfs1;
        crfdata.range(63, 32) = rfs2;
        crfdata.range(95, 64) = rfs3;
        crfdata.range(127, 96) = rfs4;

        // ideally each data is equivalen to one weight line
        // check depth is done for one line
        int var_counter = i % VMM_COUNT;
        var_counter = var_counter + unit_counter;
        if (var_counter >= VMM_COUNT) {
          var_counter = var_counter - VMM_COUNT;
        }
        fill_crx_VMM_individually(exs, var_counter);
        fill_crf_VMM_individually(crfdata, var_counter);
        fill_wsums_VMM_individually(wsumdata, var_counter);
        ra = rb;
        DWAIT(3);
      }
    }

    DWAIT();
    if (load_inp_drv) {
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

        // cout << "inp_data1[" << ra << "], " << inp_data1[ra] << endl;
        // cout << "inp_data2[" << ra << "], " << inp_data2[ra] << endl;
        // cout << "inp_data3[" << ra << "], " << inp_data3[ra] << endl;
        // cout << "inp_data4[" << ra << "], " << inp_data4[ra] << endl;
        ra = rb;
        DWAIT();
      }
    }
    load_data.write(false);
    // wait();
    DWAIT();
  }
}
