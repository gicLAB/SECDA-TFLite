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

    for (int a = 0; a < pmF; a++) {
#pragma HLS pipeline II = 1
      if (a < pmr) {
        bias[a] = din4->read().data;
        wt_sum[a] = din1->read().data;
      }
      DWAIT(2);
    }

    for (int b = 0; b < pnF; b++) {
#pragma HLS pipeline II = 1
      if (b < pnr) {
        in_sum[b] = din4->read().data;
      }
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