void ACCNAME::Data_In() {
  // clang-format off
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=data_loadS
  // clang-format on

  data_loadS.write(0);
  wait();
  while (1) {
    data_loadS.write(1);
    while (!load_data.read()) wait();
    int cpf = cols_per_filter;
    if (load_wgt) {
      data_loadS.write(2);
      DWAIT();
      wgt_packet wp = wgt_packet(&din1);
      nfilters = din1.read().data;
      int temp = nfilters;
      int pe_dex = 0;
      for (int pe_dex = 0; pe_dex < PE_COUNT; pe_dex++) {
#pragma HLS unroll
        // data_loadS.write(20 + pe_dex);
        for (int i = 0; i < cpf; i++) {
          for (int j = 0; j < wp.wgt_depth; j++) {
            bUF d1;
            acc_dt array[UF];
            for (int u = 0; u < UF; u += 4) {
#pragma HLS unroll
              sc_uint<32> data = din1.read().data;
              array[u + 0] = data.range(7, 0).to_int();
              array[u + 1] = data.range(15, 8).to_int();
              array[u + 2] = data.range(23, 16).to_int();
              array[u + 3] = data.range(31, 24).to_int();
            }
            d1.insert(array);
            vars.wgt_write(d1, pe_dex);
            DWAIT(4);
          }
          sc_int<32> data = din1.read().data.to_int();
          vars.wgt_sum_fifo_write(data, pe_dex);
          DWAIT();
        }
      }
      // data_loadS.write(3);
      // separate bias, crf, crx
      for (int i = 0; i < nfilters; i++) {
        sc_int<32> data = din1.read().data.to_int();
        bias_buf[i] = data;
        DWAIT();
      }
      for (int i = 0; i < nfilters; i++) {
        sc_int<32> data = din1.read().data.to_int();
        crf_buf[i] = data;
        DWAIT();
      }
      for (int i = 0; i < nfilters; i+=4) {
        sc_int<32> data = din1.read().data.to_int();
        crx_buf[i] = data.range(7, 0);
        crx_buf[i+1] = data.range(15, 8);
        crx_buf[i+2] = data.range(23, 16);
        crx_buf[i+3] = data.range(31, 24);
        DWAIT();
      }


      // data_loadS.write(4);
      while (!wgt_loaded()) wait();
    }

    DWAIT();
    load_data.write(false);
    wait();
  }
}
