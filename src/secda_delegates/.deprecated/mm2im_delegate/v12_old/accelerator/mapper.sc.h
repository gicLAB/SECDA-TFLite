#include "acc.sc.h"

void ACCNAME::Pattern_Decoder() {
#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = \
    pdS
  pdS.write(0);
  tempS.write(0);
  wait();
  while (1) {
    pdS.write(1);
    while (!start_decode.read()) wait();
#pragma HLS PIPELINE II = 1
    sc_int<32> crow = srow;
    sc_int<32> width_col_n = width_col;
    for (int i = 0; i < number_of_rows; i++) {
#pragma HLS PIPELINE II = 1
      pdS.write(2);
      sc_int<32> cal_col = crow % width_col_n;
      sc_int<32> cal_row = crow / width_col_n;
      int h_pad = -pt + (stride_y * cal_row);
      int w_pad = -pl + (stride_x * cal_col);
      int im_dex = (h_pad * ow + w_pad);
      int row = 0;
      int pouts = 0;
      DWAIT(48);
      pdS.write(3);
      int nwdw = number_of_rows;
      for (int ih = 0; ih < kernel_size; ih++) {
#pragma HLS PIPELINE II = 1
        for (int iw = 0; iw < kernel_size; iw++) {
#pragma HLS PIPELINE II = 1
          pdS.write(31);
          if (ih + h_pad >= 0 and ih + h_pad < oh and iw + w_pad >= 0 and
              iw + w_pad < ow) {
            col_out_indices_write_lim(row, im_dex, false, nfilters);
          }
          pdS.write(32);
          im_dex += 1;
          row += 1;
          DWAIT();
        }
        im_dex += (ow - kernel_size);
        DWAIT(3);
      }
      col_out_indices_write_lim(0, 0, true, nfilters);
      crow += 1;
      DWAIT();
    }
    start_decode.write(false);
    DWAIT();
  }
}
