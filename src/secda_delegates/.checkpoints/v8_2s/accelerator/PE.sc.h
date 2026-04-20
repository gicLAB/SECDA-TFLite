// pe.sc.h

void ACCNAME::PE() {
  peTotalS.write(0);
  wait();
  DWAIT(2);

  while(1) {
    while(peReadyS.read() == 0) wait();
    
    peTotalS.write(1);
    
    // CHANGE 4: Loop Swap - Output Stationary
    // Outer Loops: Spatial (Rows/Cols)
    // Inner Loop: Depth (K)
    
    for (int i = 0; i < pn_block; i++) {
        for (int j = 0; j < pm_block; j++) {
            #pragma HLS PIPELINE II=1
            
            // Registers to hold sum for each core
            // (Using registers avoids BRAM latency penalties)
            int sum[NUM_CORES];
            #pragma HLS ARRAY_PARTITION variable=sum complete
            
            // Init
            for(int c=0; c<NUM_CORES; c++) sum[c] = 0;

            // Inner Loop: Depth
            // We iterate through the whole loaded depth chunk
            for (int k = 0; k < pK; k += pkF) {
                
                // Unroll across cores
                for (int core = 0; core < NUM_CORES; core++) {
                    #pragma HLS UNROLL
                    
                    // Unroll dot product
                    for (int l = 0; l < pkF; l++) {
                       #pragma HLS UNROLL
                       
                       sc_int<8> r = rows[core][i][k+l];
                       sc_int<8> c = cols[core][j][k+l];
                       sum[core] += r * c;
                    }
                }
            }
            
            // Write to BRAM only ONCE per pixel
            for (int core = 0; core < NUM_CORES; core++) {
                #pragma HLS UNROLL
                // We accumulate into the existing res buffer (if pK > loaded chunk)
                // But usually PE handles the *loaded* chunk. 
                // Since Scheduler handles 'k' internally via pK, this is correct.
                res[core][i][j] += sum[core]; 
            }
        }
    }
    
    peTotalS.write(0);
    peReadyS.write(0);
    wait();
  }
}