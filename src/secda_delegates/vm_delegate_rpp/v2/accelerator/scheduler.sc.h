void ACCNAME::load_inputs(int i_idx, int d) {
  inp_len_VMM(d);
  for (int i = 0; i < d; i++) {
#pragma HLS pipeline II = 1
    ACC_DTYPE<32> data1 = inp_data1[i_idx];
    ACC_DTYPE<32> data2 = inp_data2[i_idx];
    ACC_DTYPE<32> data3 = inp_data3[i_idx];
    ACC_DTYPE<32> data4 = inp_data4[i_idx];
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

void ACCNAME::start_VMM(int id, int w_idx, int params[13]) {
  schS.write(40 + (id * 2) + 1);
  // wait();
  DWAIT();
  start_compute_VMM(id, w_idx, depth);
  schS.write(40 + (id * 2) + 2);
  // wait();
  DWAIT();
//   for (int i = 0; i < 13; i++) {
// #pragma HLS pipeline II = 1
//     vars.post_write(params[i], id);
//   }
  for (int i = 0; i < 9; i++) {
#pragma HLS pipeline II = 1
    vars.post_write(params[i], id);
  }
  // DWAIT(26);
  // DWAIT(1);
  // wait();
}

void ACCNAME::schedule_vmm_unit(int unit_counter, int w_idx, int l, int r) {
  int params[13];
#pragma HLS array_partition variable = params complete dim = 0

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

  params[0] = wgt_sum1[l];
  params[1] = wgt_sum2[l];
  params[2] = wgt_sum3[l];
  params[3] = wgt_sum4[l];
  params[4] = crf1[l];
  params[5] = crf2[l];
  params[6] = crf3[l];
  params[7] = crf4[l];
  params[8] = crx[l];


  // DWAIT(7);
  DWAIT();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    if (unit_counter == i) start_VMM(i, w_idx, params);
    // DWAIT(4);
  }
}

void ACCNAME::Scheduler() {
  init_VMM();
  int unit_counter = 0;
  schS.write(0);
  wait();
  while (1) {
    schS.write(10);
    while (!schedule.read()) wait();

    schS.write(1);
    int dm = depth / 4;
    DWAIT(1);
    for (int r = 0; r < inp_block; r += 4) {
      int r4 = r / 4;
      int i_idx = r4 * dm;
      schS.write(2);
      vmm_ready_VMM();
      schS.write(3);
      load_inputs(i_idx, dm);
      schS.write(4);
      for (int l = 0; l < wgt_block; l += 4) {
        schS.write(5);
        int l4 = l / 4;
        int w_idx = l4 * dm;
        schedule_vmm_unit(unit_counter, w_idx, l4, r4);
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
    unit_counter = vars.next(unit_counter);
    schS.write(8);
    DWAIT();
    // wait();
  }
}
