void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();

  DWAIT(2);

  while (1) {
    // wait(); //r

    while (biasReadReadyS.read() == 0) wait();

    // wait();//r

    int pnr = pN_rem;
    int pmr = pM_rem;

    // wait(); // r

    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;
    // wait(); //r

    if (layer_t == 1) {
      for (int a = 0; a < pmr; a += 4) {

        crf_v[a + 0] = din1->read().data;
        crf_v[a + 1] = din1->read().data;
        crf_v[a + 2] = din1->read().data;
        crf_v[a + 3] = din1->read().data;

        ADATA d_crx = din4->read();
        crx_v[a + 0] = d_crx.data.range(7, 0);
        crx_v[a + 1] = d_crx.data.range(15, 8);
        crx_v[a + 2] = d_crx.data.range(23, 16);
        crx_v[a + 3] = d_crx.data.range(31, 24);
        // wait(); //r
      }
      // wait(); //r
    }

    if (is_bias == 1) {
      for (int a = 0; a < pmr; a++) {
        bias[a] = din4->read().data;
      }
    }

    for (int a = 0; a < pmr; a++) {
      int wt_data = din1->read().data;
      wt_sum[a] = wt_data * rhs_offset;
      if (!is_bias) {
        bias[a] = 0;
      }
    }

    wait(); // r

    for (int a = 0; a < pn_block; a++) {
#pragma HLS pipeline II = 1
      if (a < pnr) {
        int in_data = din4->read().data;
        in_sum[a] = in_data * lhs_offset;
        for (int b = 0; b < pmr; b++)
          prec[a][b] = bias[b] + wt_sum[b] + in_sum[a];
      }
    }
    wait(); //r
    // cout << "After final part" << endl;
    readTimeBiasS.write(0);
    biasReadReadyS.write(0);
    wait(); //! Do not remove this
  }
}