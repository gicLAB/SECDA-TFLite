#include "acc_config.sc.h"

void ACCNAME::PPU() {
  pePostTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (ppuReadyS.read() == 0) wait();
    pePostTotalS.write(1);

    int cur_ra = ra;
    DWAIT(3);

    for (int core = 0; core < NUM_CORES; core++) {
        for (int i = 0; i < pn_block; i++) {
            for (int j = 0; j < pm_block; j += 4) {
                #pragma HLS PIPELINE II = 1

                // Accumulate
                res[core][i][j + 0] += prec[core][i][j + 0];
                res[core][i][j + 1] += prec[core][i][j + 1];
                res[core][i][j + 2] += prec[core][i][j + 2];
                res[core][i][j + 3] += prec[core][i][j + 3];

                wait();
                
                if (layer_t == 1) { // CONV
                    wait();


                    for (int c = 0; c < 4; c++) {
                        #pragma HLS UNROLL
                        local_crf[c] = crf_v[core][j + c];
                        local_crx[c] = crx_v[core][j + c];
                        local_res[c] = res[core][i][j + c];
                    }
                    wait();

                    for (int c = 0; c < 4; c++) {
                        #pragma HLS UNROLL
                        if (local_crx[c] > 0) {
                            local_pl[c] = local_crx[c];
                            local_pr[c] = 0;
                            local_msk[c] = 0;
                            local_sm[c] = 0;
                        } else {
                            local_pl[c] = 1;
                            local_pr[c] = -local_crx[c];
                            local_msk[c] = (1 << -local_crx[c]) - 1;
                            local_sm[c] = local_msk[c] >> 1;
                        }
                    }
                    wait();

                    #ifndef __SYNTHESIS__
                    value1 = Quantised_Multiplier_Conv(local_res[0], local_crf[0], local_crx[0]);
                    value2 = Quantised_Multiplier_Conv(local_res[1], local_crf[1], local_crx[1]);
                    value3 = Quantised_Multiplier_Conv(local_res[2], local_crf[2], local_crx[2]);
                    value4 = Quantised_Multiplier_Conv(local_res[3], local_crf[3], local_crx[3]);
                    #else 
                    value1 = Quantised_Multiplier_Conv(local_res[0], local_crf[0], local_pl[0], local_pr[0], local_msk[0], local_sm[0]);
                    value2 = Quantised_Multiplier_Conv(local_res[1], local_crf[1], local_pl[1], local_pr[1], local_msk[1], local_sm[1]);
                    value3 = Quantised_Multiplier_Conv(local_res[2], local_crf[2], local_pl[2], local_pr[2], local_msk[2], local_sm[2]);
                    value4 = Quantised_Multiplier_Conv(local_res[3], local_crf[3], local_pl[3], local_pr[3], local_msk[3], local_sm[3]);
                    #endif
                    wait();

                } else { // FC Layer (Layer 0)
                    int cur_crf = crf;
                    int cur_crx = crx;
                    wait();
                    if (cur_crx > 0) {
                        pl =  cur_crx;
                        pr = 0; msk = 0; sm = 0;
                    } else {
                        pl = 1;
                        pr = -cur_crx;
                        msk = (1 << -cur_crx) - 1;
                        sm = msk >> 1;
                    }
                    wait();
                    value1 = Quantised_Multiplier_FC(res[core][i][j + 0], cur_crf, cur_crx);
                    value2 = Quantised_Multiplier_FC(res[core][i][j + 1], cur_crf, cur_crx);
                    value3 = Quantised_Multiplier_FC(res[core][i][j + 2], cur_crf, cur_crx);
                    value4 = Quantised_Multiplier_FC(res[core][i][j + 3], cur_crf, cur_crx);
                    wait();
                }

                svalue1 = value1 + cur_ra;
                svalue2 = value2 + cur_ra;
                svalue3 = value3 + cur_ra;
                svalue4 = value4 + cur_ra;
                wait();

                if (svalue1 > MAX8) svalue1 = MAX8; else if (svalue1 < MIN8) svalue1 = MIN8;
                if (svalue2 > MAX8) svalue2 = MAX8; else if (svalue2 < MIN8) svalue2 = MIN8;
                if (svalue3 > MAX8) svalue3 = MAX8; else if (svalue3 < MIN8) svalue3 = MIN8;
                if (svalue4 > MAX8) svalue4 = MAX8; else if (svalue4 < MIN8) svalue4 = MIN8;
                wait();

                dout_1 = svalue1.range(7, 0);
                dout_2 = svalue2.range(7, 0);
                dout_3 = svalue3.range(7, 0);
                dout_4 = svalue4.range(7, 0);

                wait(); // a

                res[core][i][j + 0] = 0;
                res[core][i][j + 1] = 0;
                res[core][i][j + 2] = 0;
                res[core][i][j + 3] = 0;
                wait();

                ADATA d_out;
                d_out.data = Clamp_Combine(dout_1, dout_2, dout_3, dout_4, MAX8, MIN8);
                if (core == NUM_CORES-1 && i == pn_block-1 && j == pm_block-4) {
                     d_out.tlast = true;
                } else {
                     d_out.tlast = false;
                }
                dout1.write(d_out);
            }
        }
    }
    wait();
    ppuReadyS.write(0);
    pePostTotalS.write(0);
    wait(); 
  }
}