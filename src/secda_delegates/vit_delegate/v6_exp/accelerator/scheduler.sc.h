void ACCNAME::Scheduler() {
#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = \
    out_sig

  // Initialize signals
  readTimeParamS.write(0);
  readTimeInpS.write(0);
  readTimeWgtS.write(0);
  readTimeBiasS.write(0);
  out_sig.write(0);
  wait();

  // For HLX
  DATA a;
  dout2.write(a);
  dout3.write(a);
  dout4.write(a);
  DWAIT(2);
  wait(); // a

  while (1) {
    readTimeParamS.write(1);
    out_sig.write(1);

    // Once per layer
    pN = din1->read().data;
    pM = din1->read().data;
    pK = din1->read().data;
    crx = din1->read().data;
    crf = din1->read().data;
    ra = din1->read().data;
    is_bias = din1->read().data;
    readTimeParamS.write(0);

    if (crx > 0) {
      pl = crx;
      pr = 0;
      msk = 0;
      sm = 0;
    } else {
      pl = 1;
      pr = -crx;
      msk = (1 << -crx) - 1;
      sm = msk >> 1;
    }
    DWAIT(11);
    wait(); // a

    //! Tile over N Dimension
    for (int n = 0; n < pN; n += pn_tile) {
      //! Tile over M dimension
      for (int m = 0; m < pM; m += pm_tile)

        //! Tile over K dimension
        for (int k = 0; k < pK; k += pk_tile) {
          // Tile-specific parameters
          pN_rem = din1->read().data;
          pM_rem = din1->read().data;
          pK_rem = din1->read().data;
          is_first_k = din1->read().data;
          is_last_k = din1->read().data;
          wait();
          if (is_first_k) {
            biasReadReadyS.write(1);
          }
          // Read inp and wgt for current slice
          inpReadReadyS.write(1);
          wait(); // a
          wgtReadReadyS.write(1);
          wait();
          while (inpReadReadyS.read() == 1 || wgtReadReadyS.read() == 1) wait();
          wait();
          while (ppuReadyS.read() == 1) wait();

          // Compute current slice
          peReadyS.write(1);
          wait();
          while (peReadyS.read() == 1) wait();
          wait();
          if (is_last_k) {
            while (biasReadReadyS.read() == 1) wait();

            ppuReadyS.write(1);
            wait();
          }
        }
    }
  }
}
