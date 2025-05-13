
void ACCNAME::Output_Handler() {
  outS.write(0);
  int data[PE_COUNT];
  bool tlast = false;
  ADATA d;
  ADATA last = {5000, 1};
  d.tlast = 0;
  bool once = true;
  wait();
  while (1) {
    if (once) {
      dout2.write(d);
      dout3.write(d);
      dout4.write(d);
      once = false;
    }
    outS.write(1);
    wait();

    while (!output_handler) wait();
    tlast = false;
    for (int j = 0; j < send_len; j++) {
      for (int i = 0; i < nfilters; i++) {
        ADATA d = vars.get(i);
        data[i] = d.data;
        d.tlast = (d.tlast && (i == nfilters - 1) && (j == send_len - 1));
        // cout << "Output_Handler: " << i << " " << j << " " << (int) ((int8_t) d.data.range(7, 0)) << " " << d.tlast << endl;
        dout1.write(d);
      }
    }
    outS.write(5);
    DWAIT(13);
  }
}