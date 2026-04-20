void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();

  DWAIT(2);

  while (1) {

    while (biasReadReadyS.read() == 0) wait();

    readTimeBiasS.write(1);

//     // zero crf_v and crx_v
//     for (int a = 0; a < pmF; a++) {
// #pragma HLS pipeline II = 1
//       crf_v[a] = 0;
//       crx_v[a] = 0;
//     }
    wait();

    int pnr = pN_rem;
    int pmr = pM_rem;

    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;
    DWAIT(3);
          wait();


    //+
    //=========================================================================
    //+ FIX: Read the per-channel crf and crx values for the current block
    //+
    //=========================================================================
    //+ Read the per-channel multipliers (crf) from the bias/parameter stream
    for (int a = 0; a < pmr; a++) {
      // Read the integer multiplier for the current channel
      crf_v[a] = din4->read().data;
      // cout << "[HW] Read crf_v[" << a << "] = " << crf_v[a] << endl;
    }
    DWAIT(2);
    wait();

    //+ Read the per-channel shifts (crx)
    for (int a = 0; a < pmr; a += 4) {
      ADATA d_crx = din4->read();
      crx_v[a + 0] = d_crx.data.range(7, 0);
      crx_v[a + 1] = d_crx.data.range(15, 8);
      crx_v[a + 2] = d_crx.data.range(23, 16);
      crx_v[a + 3] = d_crx.data.range(31, 24);
      DWAIT(1);
      // cout << "[HW] Read crx_v[" << a << "] = " << crx_v[a] << endl;
      // cout << "[HW] Read crx_v[" << a+1 << "] = " << crx_v[a+1] << endl;
      // cout << "[HW] Read crx_v[" << a+2 << "] = " << crx_v[a+2] << endl;
      // cout << "[HW] Read crx_v[" << a+3 << "] = " << crx_v[a+3] << endl; 
    }
    // cout << "[HW] Finished reading crx_v" << endl << endl;
    DWAIT(2);
    wait();
    //+
    //=========================================================================

    // Copy the per-channel crx and crf values for the current block

    for (int a = 0; a < pmF; a++) {
#pragma HLS pipeline II = 1
      if (a < pmr) {
        bias[a] = din4->read().data;
        wt_sum[a] = din1->read().data;
      }
      DWAIT(2);
    }
    wait();

    for (int b = 0; b < pnF; b++) {
#pragma HLS pipeline II = 1
      if (b < pnr) {
        in_sum[b] = din4->read().data;
      }
      DWAIT(2);
    }
    wait();

    DWAIT(5); //? Not too sure about this one

    for (int a = 0; a < pnr; a++) {
      for (int b = 0; b < pmr; b++) {
#pragma HLS pipeline II = 1
        // prec[a][b] = bias[b] + wt_sum[b] + in_sum[a];
        prec[a][b] =
            bias[b] + (wt_sum[b] * rhs_offset) + (in_sum[a] * lhs_offset);
        DWAIT(10); // ? Not sure about this
      }
    }
    wait();
    readTimeBiasS.write(0);
    biasReadReadyS.write(0);
    wait();
  }
}