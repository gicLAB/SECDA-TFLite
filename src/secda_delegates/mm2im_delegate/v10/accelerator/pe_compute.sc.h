#include "acc_config.sc.h"

void PE::Compute() {
  compute_done.write(false);
  wgt_loaded.write(false);
  computeS.write(0);
  wait();
  while (1) {

    computeS.write(1);
    wgt_loaded.write(false);
    DWAIT();
    while (!online) {
      wait();
    }
    sc_int<32> depth_32 = depth.read();
    depth_16 = depth_32.range(15, 0);

    // load weights
    int i = 0;
    computeS.write(2);
    for (int c = 0; c < cols_per_filter; c++) {
      for (int d = 0; d < depth_16; d++) {
        bUF data = wgt_fifo_in.read();
        data.retrieve(wgt_cols_buf, i);
        i++;
        DWAIT(1);
      }
      wgt_col_sum[c] = wgt_sum_fifo_in.read();
      DWAIT(1);
    }
    wait();
    computeS.write(3);
    wgt_loaded.write(true);
    wait();

    computeS.write(4);
    // PE is active (activate_PEs() called)
    while (online) {
      computeS.write(5);

      // waiting for start_compute() call
      while (!compute) {
        computeS.write(6);
        if (!online) break;
        wait();
      }
      if (!online) break;

      computeS.write(7);
      // loads inputs
      for (int d = 0; d < depth_16; d++) {
#pragma HLS PIPELINE II = 1
        bUF data = inp_fifo_in.read();
        data.retrieve(inp_row_buf, d);
        DWAIT();
      }
      DWAIT(3);

      computeS.write(8);
      DWAIT();
      int pouts = 0;
      ADATA d = col_indices_fifo.read();
      while (!d.tlast) {
#pragma HLS PIPELINE II = 1
        sc_uint<8> data = d.data.range(7, 0);
        col_offset[pouts] = data * depth_16;
        out_buf[pouts++] = wgt_col_sum[d.data];
        computeS.write(9);
        d = col_indices_fifo.read();
        DWAIT();
      }
      DWAIT(3);

      computeS.write(10);

      for (int d = 0; d < depth_16; d++) {
        for (int u = 0; u < UF; u++) {
#pragma HLS UNROLL
          inp_temp[u] = inp_row_buf[d][u];
        }
        DWAIT(2);
        for (int i = 0; i < pouts; i++) {
#pragma HLS loop_tripcount min = 20 max = 20 avg = 20
#pragma HLS PIPELINE II = 1
          int col_off = col_offset[i];
          int sum = 0;
          for (int u = 0; u < UF; u++) {
#pragma HLS UNROLL
            acc_dt wt1 = wgt_cols_buf[col_off + d][u];
            sum += wt1 * inp_temp[u];
          }
          out_buf[i] += sum;
          DWAIT();
        }
        DWAIT(11);
      }

      for (int i = 0; i < pouts; i++) {
#pragma HLS PIPELINE II = 1
        int dout = out_buf[i];
        temp_fifo_out.write(out_buf[i]);
        DWAIT();
      }
      DWAIT(3);

      computeS.write(11);
      DWAIT();

      computeS.write(12);
      compute_done.write(true);
      while (!reset_compute) {
        computeS.write(13);
        DWAIT();
      }

      compute_done.write(false);
      computeS.write(14);
    }
    computeS.write(15);
    DWAIT();
  }
}