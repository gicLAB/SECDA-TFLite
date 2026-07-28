// HWC (ReadInp):
//     0 Idle
//     1 Reading

void ACCNAME::ReadInp() {
  wait();
  while (1) {

    // HWC_SIG(ReadInp, 0);
    // ReadInp_si.write(0);

    while (inpReadReadyS.read() == 0) wait();

    // HWC_SIG(ReadInp, 1);
    // ReadInp_si.write(1);

    readTimeInpS.write(1);

    int count = inpReadReadyS.read();
    DWAIT(2);

    int words_per_row = pK >> 2;

    if (count == 1) { 
      // BROADCAST MODE
      for (int i = 0; i < pn_block; i++) {
        #pragma HLS LOOP_FLATTEN off
        for (int w = 0; w < words_per_row; w++) {
          #pragma HLS PIPELINE II=1
          ADATA d = din2->read();
          for (int core = 0; core < NUM_CORES; core++) {
            #pragma HLS UNROLL
            rows_packed[core][i][w] = d.data;
          }
        }
      }
    } else { 
      // SEQUENTIAL MODE
      for (int c = 0; c < count; c++) {
        #pragma HLS LOOP_FLATTEN off
        for (int i = 0; i < pn_block; i++) {
          #pragma HLS LOOP_FLATTEN off
          for (int w = 0; w < words_per_row; w++) {
            #pragma HLS PIPELINE II=1
            ADATA d = din2->read();
            rows_packed[c][i][w] = d.data;
          }
        }
      }
    }

    DWAIT(1);
    inpReadReadyS.write(0);
    readTimeInpS.write(0);
    wait();
  }
}