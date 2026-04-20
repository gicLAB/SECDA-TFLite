// readwgt.sc.h

void ACCNAME::ReadWgt() {
  wait(); 
  while(1) {
    while(wgtReadReadyS.read() == 0) wait();
    
    int count = wgtReadReadyS.read(); 
    
    // In V8 original: You expected 'count' separate DMA packets.
    // In Fix: You get 1 BIG packet containing 'count' blocks.
    
    // Note: The scheduler sends '4' to wgtReadReadyS. 
    // We will just loop over the single stream.
    
    for (int c = 0; c < count; c++) {
       for (int i = 0; i < pm_block; i++) {
         for (int k = 0; k < pK; k+=4) {
            ADATA d = din3->read();
            
            if (mode.read() == MODE_MOBILE) { // Broadcast
               for(int core=0; core<NUM_CORES; core++) {
                  cols[core][i][k+0] = d.data.range(7,0);
                  cols[core][i][k+1] = d.data.range(15,8);
                  cols[core][i][k+2] = d.data.range(23,16);
                  cols[core][i][k+3] = d.data.range(31,24);
               }
            } else { // Dense (Unique per core)
               // Since we packed them sequentially in driver, 
               // the first block goes to core 0, second to core 1, etc.
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