void ACCNAME::PE() {
  wait();
  while(1) {
    while(peReadyS.read() == 0) wait();
    
    peTotalS.write(1);
    
    // Main Compute Loop
    for (int k = 0; k < pK; k += pkF) {
       for (int core = 0; core < NUM_CORES; core++) {
          #pragma HLS UNROLL 
          for (int i = 0; i < pn_block; i++) {
             for (int j = 0; j < pm_block; j++) {
                #pragma HLS PIPELINE II=1
                int sum = 0;
                for (int l = 0; l < pkF; l++) {
                   int r = rows[core][i][k+l];
                   int c = cols[core][j][k+l];
                   sum += r * c;
                }
                if (k == 0) res[core][i][j] = sum;
                else res[core][i][j] += sum;
             }
          }
       }
    }
    
    peTotalS.write(0);
    peReadyS.write(0);
    wait();
  }
}