// HWC (ReadWgt):
//     0 Idle
//     1 Reading

void ACCNAME::ReadWgt() {
  wait(); 
  while(1) {

   //  HWC_SIG(ReadWgt, 0);
   //  ReadWgt_si.write(0);

    while(wgtReadReadyS.read() == 0) wait();

   //  HWC_SIG(ReadWgt, 1);
   //  ReadWgt_si.write(1);

    readTimeWgtS.write(1);
    
    int count = wgtReadReadyS.read(); 
    DWAIT(2);

    int words_per_row = pK >> 2; 
    
    if (count == 1) {
      // BROADCAST MODE
      for (int i = 0; i < pm_block; i++) {
        #pragma HLS LOOP_FLATTEN off
        for (int w = 0; w < words_per_row; w++) {
          #pragma HLS PIPELINE II=1
          ADATA d = din3->read();
          for(int core = 0; core < NUM_CORES; core++) {
            #pragma HLS UNROLL
            cols_packed[core][i][w] = d.data;
          }
        }
      }
    } else {
      // SEQUENTIAL MODE
      for (int c = 0; c < count; c++) {
        #pragma HLS LOOP_FLATTEN off
        for (int i = 0; i < pm_block; i++) {
          #pragma HLS LOOP_FLATTEN off
          for (int w = 0; w < words_per_row; w++) {
            #pragma HLS PIPELINE II=1
            ADATA d = din3->read();
            cols_packed[c][i][w] = d.data;
          }
        }
      }
    }

    DWAIT(1);
    wgtReadReadyS.write(0);
    readTimeWgtS.write(0);
    wait();
  }
}