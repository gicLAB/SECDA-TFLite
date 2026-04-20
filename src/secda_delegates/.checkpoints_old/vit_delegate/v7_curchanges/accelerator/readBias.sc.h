void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();

  DWAIT(2);

  while (1) {

    while (biasReadReadyS.read() == 0) wait();

    readTimeBiasS.write(1);

    int pnr = pN_rem;
    int pmr = pM_rem;

    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;
    DWAIT(3);

        //+ =========================================================================
    //+ FIX: Read the per-channel crf and crx values for the current block
    //+ =========================================================================
    //+ Read the per-channel multipliers (crf) from the bias/parameter stream
    for (int a = 0; a < pmr; a++) {
// #pragma HLS pipeline II = 1
      // if (a < pmr) {
        // Read the integer multiplier for the current channel
        crf_v[a] = din4->read().data;
      // }
    }
    DWAIT(2);

    //+ Read the per-channel shifts (crx)
    for (int a = 0; a < pmr; a++) {
// #pragma HLS pipeline II = 1
      // if (a < pmr) {
        // Read the shift value and cast it to int8_t
        //! This could ba causing an issue
        // crx_v[a] = (int8_t)din4->read().data;
        crx_v[a] = din4->read().data;

      // }
    }
    DWAIT(2);
    //+ =========================================================================


    // Copy the per-channel crx and crf values for the current block

    for (int a = 0; a < pmr; a++) {
// #pragma HLS pipeline II = 1
      // if (a < pmr) {
        bias[a] = din4->read().data;
        wt_sum[a] = din1->read().data;
      // }
      DWAIT(2);
    }

    for (int b = 0; b < pnr; b++) {
// #pragma HLS pipeline II = 1
      // if (b < pnr) {
        in_sum[b] = din4->read().data;
      // }
      DWAIT(2);
    }
    
    DWAIT(5); //? Not too sure about this one

    for (int a = 0; a < pnr; a++) {
      for (int b = 0; b < pmr; b++) {
#pragma HLS pipeline II = 1
        // prec[a][b] = bias[b] + wt_sum[b] + in_sum[a];
            prec[a][b] = bias[b] + (wt_sum[b] * rhs_offset) + (in_sum[a] * lhs_offset);
            DWAIT(10); // ? Not sure about this
      }
    }
    readTimeBiasS.write(0);
    biasReadReadyS.write(0);
    wait();
  }
}