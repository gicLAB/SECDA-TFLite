void ACCNAME::PE() {
//   peReadyS.write(0);
  peTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (peReadyS.read() == 0) wait();

    // Local variables for loop bounds
    int pnr = pn_block;
    int pmr = pm_block;
    int pkr = pK;

    peTotalS.write(1);
    DWAIT(1);

    // -------------------------------------------------------------------
    // MAIN LOOP STRUCTURE (Output Stationary)
    // i = Rows, j = Columns. 
    // We compute a full block of pixels before moving to the next.
    // -------------------------------------------------------------------
    for (int i = 0; i < pnr; i++) {
      for (int j = 0; j < pmr; j += 4) {
        
        DWAIT(2);
        
        // Reset accumulators
        for (int c = 0; c < NUM_CORES; c++) {
            #pragma HLS UNROLL
            for (int k = 0; k < pkF; k++) {
                #pragma HLS UNROLL
                temp_reg[c][k][0] = 0;
                temp_reg[c][k][1] = 0;
                temp_reg[c][k][2] = 0;
                temp_reg[c][k][3] = 0;
            }
        }
        DWAIT(1);

        // 2. The PIPELINE Loop (Depth K)
        // This traverses the depth. Because it is the innermost loop, 
        // we read weights/inputs continuously, achieving II=1.
        for (int k = 0; k < pkr; k += pkF) {
            #pragma loop_tripcount min = pKT max = pKT
            #pragma HLS PIPELINE II = 1

            // Parallelize across Cores
            for (int c = 0; c < NUM_CORES; c++) {
                #pragma HLS UNROLL
                // Existing V7 Logic: First half (Integer Math)
                for (int l = 0; l < (pkF / 2); l++) {
                    int curRow = rows[c][i][k + l];
                    temp_reg[c][l][0] += curRow * cols[c][j + 0][k + l];
                    temp_reg[c][l][1] += curRow * cols[c][j + 1][k + l];
                    temp_reg[c][l][2] += curRow * cols[c][j + 2][k + l];
                    temp_reg[c][l][3] += curRow * cols[c][j + 3][k + l];
                }

                // Existing V7 Logic: Second half (DSP/Custom Math)
                for (int l = (pkF / 2); l < pkF; l++) {
                    int curRow = rows[c][i][k + l];
                    temp_reg[c][l][0] += mul_s8(curRow, cols[c][j + 0][k + l]);
                    temp_reg[c][l][1] += mul_s8(curRow, cols[c][j + 1][k + l]);
                    temp_reg[c][l][2] += mul_s8(curRow, cols[c][j + 2][k + l]);
                    temp_reg[c][l][3] += mul_s8(curRow, cols[c][j + 3][k + l]);
                }
            }
            DWAIT(5);
        }

        // 3. Reduction (Summing up the partial 'temp' results)
        // This happens once per pixel block, so it doesn't break the pipeline.
        for (int l = 1; l < pkF; l++) {
            #pragma HLS UNROLL
            for (int c = 0; c < NUM_CORES; c++) {
                #pragma HLS UNROLL
                temp_reg[c][0][0] += temp_reg[c][l][0];
                temp_reg[c][0][1] += temp_reg[c][l][1];
                temp_reg[c][0][2] += temp_reg[c][l][2];
                temp_reg[c][0][3] += temp_reg[c][l][3];
            }
            DWAIT(4);
        }
        DWAIT(4);

        // 4. Final Writeback to BRAM
        for (int c = 0; c < NUM_CORES; c++) {
            #pragma HLS UNROLL
            res[c][i][j + 0] += temp_reg[c][0][0];
            res[c][i][j + 1] += temp_reg[c][0][1];
            res[c][i][j + 2] += temp_reg[c][0][2];
            res[c][i][j + 3] += temp_reg[c][0][3];
        }
        DWAIT(1);
      }
    }
    
    DWAIT(1);
    peReadyS.write(0);
    peTotalS.write(0);
    wait();
  }
}