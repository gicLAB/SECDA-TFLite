void ACCNAME::PE() {
  peTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (peReadyS.read() == 0) wait();

    int pnr = pn_block;
    int pmr = pm_block;
    int pkr = pK;

    peTotalS.write(1);
    DWAIT(1);

    for (int i = 0; i < pnr; i++) {
      for (int j = 0; j < pmr; j += 4) {
        
        DWAIT(2);
        
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

        for (int k = 0; k < pkr; k += pkF) {
            #pragma loop_tripcount min = pKT max = pKT
            #pragma HLS PIPELINE II = 1

            for (int c = 0; c < NUM_CORES; c++) {
                #pragma HLS UNROLL
                
                // OPTIMIZATION: Local registers to hold the packed words
                sc_int<32> r_pack[4];
                sc_int<32> c_pack[4][4]; 
                
                // Fetch the packed words for this cycle (4 words = 16 bytes)
                for (int w = 0; w < 4; w++) {
                    #pragma HLS UNROLL
                    r_pack[w] = rows_packed[c][i][(k/4) + w];
                    c_pack[0][w] = cols_packed[c][j + 0][(k/4) + w];
                    c_pack[1][w] = cols_packed[c][j + 1][(k/4) + w];
                    c_pack[2][w] = cols_packed[c][j + 2][(k/4) + w];
                    c_pack[3][w] = cols_packed[c][j + 3][(k/4) + w];
                }

                // Unpack via bit-slicing and compute
                for (int l = 0; l < pkF; l++) {
                    #pragma HLS UNROLL
                    int w_idx = l / 4;
                    int b_idx = (l % 4) * 8; 
                    
                    sc_int<8> curRow = r_pack[w_idx].range(b_idx + 7, b_idx);
                    sc_int<8> col0 = c_pack[0][w_idx].range(b_idx + 7, b_idx);
                    sc_int<8> col1 = c_pack[1][w_idx].range(b_idx + 7, b_idx);
                    sc_int<8> col2 = c_pack[2][w_idx].range(b_idx + 7, b_idx);
                    sc_int<8> col3 = c_pack[3][w_idx].range(b_idx + 7, b_idx);

                    if (l < (pkF / 2)) {
                        temp_reg[c][l][0] += curRow * col0;
                        temp_reg[c][l][1] += curRow * col1;
                        temp_reg[c][l][2] += curRow * col2;
                        temp_reg[c][l][3] += curRow * col3;
                    } else {
                        temp_reg[c][l][0] += mul_s8(curRow, col0);
                        temp_reg[c][l][1] += mul_s8(curRow, col1);
                        temp_reg[c][l][2] += mul_s8(curRow, col2);
                        temp_reg[c][l][3] += mul_s8(curRow, col3);
                    }
                }
            }
            DWAIT(5);
        }

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

        for (int c = 0; c < NUM_CORES; c++) {
            #pragma HLS UNROLL
            res[c][i][j + 0] += temp_reg[c][0][0];
            res[c][i][j + 1] += temp_reg[c][0][1];
            res[c][i][j + 2] += temp_reg[c][0][2];
            res[c][i][j + 3] += temp_reg[c][0][3];
        }
        DWAIT(2);
      }
    }
    
    DWAIT(1);
    peReadyS.write(0);
    peTotalS.write(0);
    wait();
  }
}