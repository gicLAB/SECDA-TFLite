
void ACCNAME::Arranger() {
  DATA d1, d2, d3, d4;
  wait();
  while (true) {
    // read the fifo
    int i = arranger_fifo.read();
    // DLOG << "arranger_fifo: " << i << endl;
    d1 = vars.dout_read(i, 0);
    d2 = vars.dout_read(i, 1);
    d3 = vars.dout_read(i, 2);
    d4 = vars.dout_read(i, 3);

    sc_uint<8> d11_data8 = d1.data.range(7, 0);
    sc_uint<8> d12_data8 = d1.data.range(15, 8);
    sc_uint<8> d13_data8 = d1.data.range(23, 16);
    sc_uint<8> d14_data8 = d1.data.range(31, 24);

    sc_int<8> d11_dataint8 = d11_data8.to_int();
    sc_int<8> d12_dataint8 = d12_data8.to_int();
    sc_int<8> d13_dataint8 = d13_data8.to_int();
    sc_int<8> d14_dataint8 = d14_data8.to_int();

    int d1_data = d1.data;
    int d2_data = d2.data;
    int d3_data = d3.data;
    int d4_data = d4.data;

    // DLOG << "d11_data8: " << (int)d11_dataint8
    //      << " d12_data8: " << (int)d12_dataint8
    //      << " d13_data8: " << (int)d13_dataint8
    //      << " d14_data8: " << (int)d14_dataint8 << endl;
    dout1.write(d1);
    dout2.write(d2);
    dout3.write(d3);
    dout4.write(d4);
    wait();
  }
}
