void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();

  DWAIT(2);

  while (1) {
    wait(); // a

    while (biasReadReadyS.read() == 0) wait();

    wait();

    int pnr = pN_rem;
    int pmr = pM_rem;

    wait(); // a

    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;
    wait();

    if (layer_t == 1) {
      for (int a = 0; a < pmr; a++) {
        crf_v[a] = din4->read().data;
      }
      wait();

      //+ Read the per-channel shifts (crx)
      for (int a = 0; a < pmr; a += 4) {
        ADATA d_crx = din4->read();
        crx_v[a + 0] = d_crx.data.range(7, 0);
        crx_v[a + 1] = d_crx.data.range(15, 8);
        crx_v[a + 2] = d_crx.data.range(23, 16);
        crx_v[a + 3] = d_crx.data.range(31, 24);
        wait();
        DWAIT(1);
      }
      DWAIT(2);
      wait();
    }

    for (int a = 0; a < pmr; a++) {

      bias[a] = din4->read().data;
      wt_sum[a] = din1->read().data;
    }
    wait();

    for (int b = 0; b < pnr; b++) {
      in_sum[b] = din4->read().data;
    }

    wait();

    for (int a = 0; a < pnr; a++) {
      for (int b = 0; b < pmr; b++) {
        // prec[a][b] = bias[b] + wt_sum[b] + in_sum[a];
        prec[a][b] =
            bias[b] + (wt_sum[b] * rhs_offset) + (in_sum[a] * lhs_offset);
      }
    }
    wait();
    readTimeBiasS.write(0);
    biasReadReadyS.write(0);
    wait();
  }
}