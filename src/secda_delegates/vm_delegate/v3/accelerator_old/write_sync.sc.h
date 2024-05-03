
void ACCNAME::Arranger() {
  DATA d1, d2, d3, d4;
  d1.tlast = false;
  d2.tlast = false;
  d3.tlast = false;
  d4.tlast = false;
  wait();
  while (true) {
    for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
      d1.data = vars.dout_read(i, 0);
      d2.data = vars.dout_read(i, 1);
      d3.data = vars.dout_read(i, 2);
      d4.data = vars.dout_read(i, 3);
      dout1.write(d1);
      dout2.write(d2);
      dout3.write(d3);
      dout4.write(d4);
      wait();
    }
  }
}
