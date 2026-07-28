void ACCNAME::ReadWgt() {
  wait(); 
  while(1) {
    while(wgtReadReadyS.read() == 0) wait();

    readTimeWgtS.write(1);
    
    int count = wgtReadReadyS.read(); 

    DWAIT(2);
    
    for (int c = 0; c < count; c++) {
      DWAIT(1);
       for (int i = 0; i < pm_block; i++) {
         DWAIT(1);
         for (int k = 0; k < pK; k+=4) {
            ADATA d = din3->read();
            
            if (count == 1) {
               for(int core=0; core<NUM_CORES; core++) {
                  // OPTIMIZATION: Write 32-bit chunk directly
                  cols_packed[core][i][k/4] = d.data;
                  DWAIT(3);
               }
            } else {
               cols_packed[c][i][k/4] = d.data;
               DWAIT(4);
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

