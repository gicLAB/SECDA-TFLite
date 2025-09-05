void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();

  while (1) {

    while (biasReadReadyS.read() == 0) wait();

    int pnr = pN_rem;
    int pmr = pM_rem;

    // readTimeBiasS.write(1);
    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;
    for (int a = 0; a < pmF; a++) {
#pragma HLS pipeline II = 1
      if (a < pmr) {
        bias[a] = din4->read().data;
        wt_sum[a] = din1->read().data;
      }
    }

    for (int b = 0; b < pnF; b++) {
#pragma HLS pipeline II = 1
      if (b < pnr) {
        in_sum[b] = din4->read().data;
      }
    }

    for (int a = 0; a < pnr; a++) {
      for (int b = 0; b < pmr; b++) {
#pragma HLS pipeline II = 1
        res[a][b] =
            bias[b] + (wt_sum[b] * rhs_offset) + (in_sum[a] * lhs_offset);
      }
    }
    biasReadReadyS.write(0);
    wait();
  }
}