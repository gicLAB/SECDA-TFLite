
void ACCNAME::Arranger() {
  DATA d1, d2, d3, d4;
  wait();
  while (true) {
    for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
      d1 = vars.dout_read(i, 0);
      d2 = vars.dout_read(i, 1);
      d3 = vars.dout_read(i, 2);
      d4 = vars.dout_read(i, 3);
      int d1_data = d1.data;
      int d2_data = d2.data;
      int d3_data = d3.data;
      int d4_data = d4.data;
      dout1.write(d1);
      dout2.write(d2);
      dout3.write(d3);
      dout4.write(d4);
      wait();
    }
  }
}
