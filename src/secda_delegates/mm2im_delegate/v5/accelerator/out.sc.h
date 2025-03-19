

void ACCNAME::Output_Handler() {
  outS.write(0);
  int data[PE_COUNT];
  bool tlast = false;
  ADATA d;
  DATA_PACKED datap;
  ADATA last = {5000, 1};
  d.tlast = 0;
  wait();
  while (1) {
    outS.write(1);
    wait();
    tlast = false;

    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      ADATA d = vars.get(i);
      data[i] = d.data;
      tlast = tlast || d.tlast;
    }

    outS.write(2);
    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      datap.insert(data[i]);
      if (i % 4 == 3) {
        d.data = datap.data;
        dout1.write(d);
      }
    }
    if (tlast) {
      dout1.write(last);
    }
    outS.write(5);
    DWAIT(13);
  }
}