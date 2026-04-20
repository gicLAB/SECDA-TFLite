#ifndef PE_H
#define PE_H

#include "acc_config.sc.h"

// Defined in acc.sc.h
// void ACCNAME::PE() { ... }

void ACCNAME::PE() {
  peReadyS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (peReadyS.read() == 0) wait();

    int pnr = pN_rem;
    int pmr = pM_rem;

    peTotalS.write(1);
    DWAIT(1);
    
    // Iterate over Output Rows (Height)
    for (int i = 0; i < pnr; i++) {
      
      // Iterate over Output Cols (Width)
      for (int j = 0; j < pmr; j += 4) {
        
        // [FIX] Explicitly zero the accumulator for this pixel group
        // BEFORE starting the depth accumulation loop.
        // This prevents "garbage" data from previous tiles/layers
        // from corrupting the start of the sum.
        // res[i][j + 0] = 0;
        // res[i][j + 1] = 0;
        // res[i][j + 2] = 0;
        // res[i][j + 3] = 0;

        DWAIT(2);
        
        // Clear the temporary partial-sum registers
        for (int k = 0; k < pkF; k++) {
// #pragma HLS unroll
          temp[k][0] = 0;
          temp[k][1] = 0;
          temp[k][2] = 0;
          temp[k][3] = 0;
        }
        DWAIT(1);

        // Iterate over Input Depth (K) - The Accumulation Loop
        for (int k = 0; k < pK; k += pkF) {
// #pragma loop_tripcount min = pKT max = pKT
// #pragma HLS pipeline II = 1
          
          // 1. Calculate Dot Products
          for (int l = 0; l < (pkF / 2); l++) {
            int curRow = rows[i][k + l];
            temp[l][0] += curRow * cols[j + 0][k + l];
            temp[l][1] += curRow * cols[j + 1][k + l];
            temp[l][2] += curRow * cols[j + 2][k + l];
            temp[l][3] += curRow * cols[j + 3][k + l];
          }
          for (int l = (pkF / 2); l < pkF; l++) {
            int curRow = rows[i][k + l];
            temp[l][0] += mul_s8(curRow, cols[j + 0][k + l]);
            temp[l][1] += mul_s8(curRow, cols[j + 1][k + l]);
            temp[l][2] += mul_s8(curRow, cols[j + 2][k + l]);
            temp[l][3] += mul_s8(curRow, cols[j + 3][k + l]);
          }
          DWAIT(5);
        }

        // 2. Reduction (Summing up the partial products in 'temp')
        for (int l = 1; l < pkF; l++) {
// #pragma HLS unroll
          temp[0][0] += temp[l][0];
          temp[0][1] += temp[l][1];
          temp[0][2] += temp[l][2];
          temp[0][3] += temp[l][3];
          DWAIT(4);
        }
        DWAIT(4);

        // 3. Accumulate into Result
        // We use += here because 'temp' contains the sum for a chunk of K.
        // Since we zeroed 'res' at the top of the 'j' loop, this is safe.
        res[i][j + 0] += temp[0][0];
        res[i][j + 1] += temp[0][1];
        res[i][j + 2] += temp[0][2];
        res[i][j + 3] += temp[0][3];
        
        DWAIT(1);
      }
    }
    DWAIT(1);
    peReadyS.write(0);
    peTotalS.write(0);
    wait();
  }
}

#endif // PE_H