// readbias_v6.cpp
void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();

  while (1) {
    while (biasReadReadyS.read() == 0) wait();

    // cout << "READBIAS START" << endl;
    readTimeBiasS.write(1);

    int bias_pnr = pN_rem;
    int bias_pmr = pM_rem;

    // Read all data from the single bias stream (din4)
    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;
    DWAIT(3);

    // Read bias values
    for (int a = 0; a < bias_pmr; a++) {
      bias[a] = din4->read().data;
      DWAIT(2);
    }
    
    // FIX: Read wt_sum from din4, NOT din1
    for (int a = 0; a < bias_pmr; a++) {
      wt_sum[a] = din4->read().data;
      DWAIT(2);
    }

    // Read in_sum values
    for (int b = 0; b < bias_pnr; b++) {
      in_sum[b] = din4->read().data;
      DWAIT(2);
    }

    DWAIT(5);

    // This loop calculates the initial value for the accumulator before PPU
    for (int a = 0; a < bias_pnr; a++) {
      for (int b = 0; b < bias_pmr; b++) {
#pragma HLS pipeline II = 1
            prec[a][b] = bias[b] + (wt_sum[b] * rhs_offset) + (in_sum[a] * lhs_offset);
      }
    }
    // cout << "READBIAS END" << endl;
    readTimeBiasS.write(0);
    biasReadReadyS.write(0);
    wait();
  }
}
