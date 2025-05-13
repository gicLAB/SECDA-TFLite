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

      // activates only the PEs that are needed
      activate_PEs_lim(nfilters);
      wait();
      wait();
      int temp = nfilters;
      int pe_dex = 0;
      for (int pe_dex = 0; pe_dex < nfilters; pe_dex++) {

        for (int i = 0; i < cpf; i++) {
          data_loadS.write(3);
          DWAIT();
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
          data_loadS.write(4);

          vars.wgt_sum_fifo_write(data, pe_dex);
          DWAIT();
        }
      }
      data_loadS.write(5);
      wait();
      DWAIT(2);
      // separate bias, crf, crx
      for (int i = 0; i < nfilters; i++) {
        sc_int<32> data = din1.read().data.to_int();
        bias_buf[i] = data;
        DWAIT();
      }
      data_loadS.write(6);
      wait();

      for (int i = 0; i < nfilters; i++) {
        sc_int<32> data = din1.read().data.to_int();
        crf_buf[i] = data;
        DWAIT();
      }
      data_loadS.write(7);
      wait();

      for (int i = 0; i < nfilters; i++) {
        sc_uint<32> data1 = din1.read().data.to_uint();
        sc_uint<32> data2 = din1.read().data.to_uint();
        sc_uint<64> data;
        data.range(31, 0) = data1;
        data.range(63, 32) = data2;
        uint64_t scale_u64 = data;
        double scale = double_from_bits(scale_u64);
        crx_scale_buf[i] = scale;
        DWAIT();
      }
      data_loadS.write(7);
      wait();

      for (int i = 0; i < nfilters; i += 4) {
#pragma HLS PIPELINE II = 1
        sc_int<32> data = din1.read().data.to_int();
        crx_buf[i] = data.range(7, 0);
        crx_buf[i + 1] = data.range(15, 8);
        crx_buf[i + 2] = data.range(23, 16);
        crx_buf[i + 3] = data.range(31, 24);
        DWAIT();
      }

      data_loadS.write(8);
      while (!wgt_loaded_lim(nfilters)) wait();
    }

    DWAIT();
    load_data.write(false);
    wait();
  }
}
