void ACCNAME::load_inputs(int i_idx, int d) {
  inp_len_VMM(d);
  for (int i = 0; i < d; i++) {
#pragma HLS pipeline II = 1
    ACC_DTYPE<32> data1 = inp_data1[i_idx];
    ACC_DTYPE<32> data2 = inp_data2[i_idx];
    ACC_DTYPE<32> data3 = inp_data3[i_idx];
    ACC_DTYPE<32> data4 = inp_data4[i_idx];

    // cout << "inp_data1[" << i_idx << "], " << inp_data1[i_idx] << endl;
    // cout << "inp_data2[" << i_idx << "], " << inp_data2[i_idx] << endl;
    // cout << "inp_data3[" << i_idx << "], " << inp_data3[i_idx] << endl;
    // cout << "inp_data4[" << i_idx << "], " << inp_data4[i_idx] << endl;
    sc_bigint<32 * 4> data;
    data.range(31, 0) = data1;
    data.range(63, 32) = data2;
    data.range(95, 64) = data3;
    data.range(127, 96) = data4;
    vars.inp_write(data, 0);
    i_idx++;
    DWAIT(3);
  }
}

// void ACCNAME::start_VMM(int id, int w_idx, int wsum_idx, int params[13]) {
//   schS.write(40 + (id * 2) + 1);
//   // wait();
//   DWAIT();
//   start_compute_VMM(id, w_idx, wsum_idx, depth);
//   schS.write(40 + (id * 2) + 2);
//   // wait();
//   DWAIT();

//   for (int i = 0; i < 1; i++) {
// #pragma HLS pipeline II = 1
//     vars.post_write(params[i], id);
//   }

//   // DWAIT(26);
//   // DWAIT(1);
//   // wait();
// }

void ACCNAME::start_VMM(int id, int w_idx, int wsum_idx, int params) {
  schS.write(40 + (id * 2) + 1);
  // wait();
  DWAIT();
  start_compute_VMM(id, w_idx, wsum_idx, depth);
  schS.write(40 + (id * 2) + 2);
  // wait();
  DWAIT();

  //   for (int i = 0; i < 1; i++) {
  // #pragma HLS pipeline II = 1
  //     vars.post_write(params[i], id);
  //   }
  vars.post_write(params, id);

  // DWAIT(26);
  // DWAIT(1);
  // wait();
}

void ACCNAME::schedule_vmm_unit(int unit_counter, int w_idx, int wsum_idx,
                                int l, int r) {
  //   int params[13];
  // #pragma HLS array_partition variable = params complete dim = 0

  // params[0] = inp_sum1[r];
  // params[1] = inp_sum2[r];
  // params[2] = inp_sum3[r];
  // params[3] = inp_sum4[r];
  // params[4] = wgt_sum1[l];
  // params[5] = wgt_sum2[l];
  // params[6] = wgt_sum3[l];
  // params[7] = wgt_sum4[l];
  // params[8] = crf1[l];
  // params[9] = crf2[l];
  // params[10] = crf3[l];
  // params[11] = crf4[l];
  // params[12] = crx[l];

  // params[0] = wsum_idx;

  // DWAIT(7);
  DWAIT();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    if (unit_counter == i) {
      // start_VMM(i, w_idx, wsum_idx, params);
      start_VMM(i, w_idx, wsum_idx, wsum_idx);
    }
    // DWAIT(4);
  }
}

void ACCNAME::Scheduler() {
  init_VMM();
  unit_counter = 0;
  schS.write(0);
  wait();
  while (1) {
    schS.write(10);
    while (!schedule.read()) wait();

    schS.write(1);

    // Expecting depth should be equally divided between two VM PE
    int dm = (depth / (4 * DEPTHTILE));
    int depthTemp = depth / 4;
    // if (dm < 1) {
    //   dm = 1;
    // }
    DWAIT(1);
    for (int r = 0; r < inp_block; r += 4) {
      int r4 = r / 4;
      int i_idx = r4 * depthTemp;
      schS.write(2);
      vmm_ready_VMM();
      // post_ready_VMM();
      unit_counter = 0;
      schS.write(3);
      load_inputs(i_idx, depthTemp);
      schS.write(4);
      for (int l = 0; l < wgt_block; l += 4) {
        schS.write(5);
        int l4 = l / 4;
        int w_idx = 0;
        w_idx = (l / (4 * VMM_COUNT)) * dm;
        int wsum_idx = 0;
        wsum_idx = l / (4 * VMM_COUNT);
        // if (r < 4) {
        //   cout << "unit_counter: " << unit_counter << endl;
        //   cout << "w_idx: " << w_idx << " wsum_idx: " << wsum_idx
        //        << " l4: " << l4 << " r4: " << r4 << endl;
        // }
        schedule_vmm_unit(unit_counter, w_idx, wsum_idx, l4, r4);
        arranger_fifo.write(unit_counter);
        unit_counter = vars.next(unit_counter);
        schS.write(6);
        DWAIT();
        // wait();
      }
    }

    schS.write(7);
    schedule.write(0);
    wait_ready_VMM();
    send_done_write_VMM(unit_counter);
    arranger_fifo.write(unit_counter);
    unit_counter = 0;
    schS.write(8);
    DWAIT();
    // wait();
  }
}
