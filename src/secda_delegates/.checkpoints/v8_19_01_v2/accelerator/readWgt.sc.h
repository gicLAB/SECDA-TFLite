void ACCNAME::ReadWgt() {
  wait(); // FIX: Just wait for Scheduler init
  while(1) {
    while(wgtReadReadyS.read() == 0) wait();
    
    int count = wgtReadReadyS.read(); // 4 (Dense) or 1 (Mobile)
    
    for (int c = 0; c < count; c++) {
       for (int i = 0; i < pm_block; i++) {
         for (int k = 0; k < pK; k+=4) {
            ADATA d = din3->read();
            
            if (count == 1) {
               // Broadcast to all cores (Mobile Mode)
               for(int core=0; core<NUM_CORES; core++) {
                  cols[core][i][k+0] = d.data.range(7,0);
                  cols[core][i][k+1] = d.data.range(15,8);
                  cols[core][i][k+2] = d.data.range(23,16);
                  cols[core][i][k+3] = d.data.range(31,24);
               }
            } else {
               // Unique load to core 'c' (Dense Mode)
               cols[c][i][k+0] = d.data.range(7,0);
               cols[c][i][k+1] = d.data.range(15,8);
               cols[c][i][k+2] = d.data.range(23,16);
               cols[c][i][k+3] = d.data.range(31,24);
            }
         }
       }
    }
    wgtReadReadyS.write(0);
    wait();
  }
}