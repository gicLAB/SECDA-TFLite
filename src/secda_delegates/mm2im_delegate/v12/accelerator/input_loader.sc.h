#include "acc.sc.h"

void ACCNAME::FIFO_Loader() {
  wait();
  while (1) {
    while (!load_fifo) wait();
    for (int r = 0; r < number_of_rows; r++) {
      for (int d = 0; d < depth; d++) {
#pragma HLS PIPELINE II = 1
        bUF d1[PE_COUNT];
        acc_dt array[UF];
        for (int u = 0; u < UF; u += 4) {
#pragma HLS unroll
          sc_uint<32> data = in_fifo.read().data;
          array[u + 0] = data.range(7, 0).to_int();
          array[u + 1] = data.range(15, 8).to_int();
          array[u + 2] = data.range(23, 16).to_int();
          array[u + 3] = data.range(31, 24).to_int();
        }
        for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
          d1[i].insert(array);
        }

        for (int i = 0; i < nfilters; i++) {
#pragma HLS PIPELINE II = 1
          vars.inp_write(d1[i], i);
          DWAIT(1);
        }
        DWAIT(4);
      }
      DWAIT(1);
    }
    load_fifo.write(false);
    DWAIT();
  }
}


void ACCNAME::Din1_Thread() {
  wait();
  while (1) {
    ADATA data = din1.read();
    in_fifo.write(data);
    DWAIT();
  }
}