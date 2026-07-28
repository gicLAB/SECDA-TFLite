#include "acc_config.sc.h"

void ACCNAME::PPU() {
  pePostTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (ppuReadyS.read() == 0) wait();
    pePostTotalS.write(1);

    int cur_ra = ra;
    DWAIT(1);

    for (int core = 0; core < NUM_CORES; core++) {
      for (int i = 0; i < pn_block; i++) {
        for (int j = 0; j < pm_block; j += 4) {
// OPTIMIZATION: Dropped II constraint from 3 to 1.
#pragma HLS PIPELINE II = 1

          int value1;
          int value2;
          int value3;
          int value4;

          sc_int<64> svalue1;
          sc_int<64> svalue2;
          sc_int<64> svalue3;
          sc_int<64> svalue4;

          int local_crf[4];
          sc_int<8> local_crx[4];
          int local_res[4];
          ACC_DTYPE<64> local_pl[4];
          ACC_DTYPE<32> local_pr[4];
          ACC_DTYPE<32> local_msk[4];
          ACC_DTYPE<32> local_sm[4];
#pragma HLS array_partition variable = local_crf complete dim = 0
#pragma HLS array_partition variable = local_crx complete dim = 0
#pragma HLS array_partition variable = local_res complete dim = 0
#pragma HLS array_partition variable = local_pl complete dim = 0
#pragma HLS array_partition variable = local_pr complete dim = 0
#pragma HLS array_partition variable = local_msk complete dim = 0
#pragma HLS array_partition variable = local_sm complete dim = 0

          int i_sum_val = in_sum[core][i];

          int prec_0 = bias[core][j + 0] + wt_sum[core][j + 0] + i_sum_val;
          int prec_1 = bias[core][j + 1] + wt_sum[core][j + 1] + i_sum_val;
          int prec_2 = bias[core][j + 2] + wt_sum[core][j + 2] + i_sum_val;
          int prec_3 = bias[core][j + 3] + wt_sum[core][j + 3] + i_sum_val;

          res[core][i][j + 0] += prec_0;
          res[core][i][j + 1] += prec_1;
          res[core][i][j + 2] += prec_2;
          res[core][i][j + 3] += prec_3;

          if (layer_t == 1) { 
            for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
              local_crf[c] = crf_v[core][j + c];
              local_crx[c] = crx_v[core][j + c];
              local_res[c] = res[core][i][j + c];
            }

            for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
              sc_int<8> cmp_crx = local_crx[c];
              if (cmp_crx > 0) {
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

          } else { 
            int cur_crf = crf;
            int cur_crx = crx;
            sc_int<8> cmp_crx = cur_crx;
            if (cmp_crx > 0) {
              pl = cur_crx;
              pr = 0;
              msk = 0;
              sm = 0;
            } else {
              pl = 1;
              pr = -cur_crx;
              msk = (1 << -cur_crx) - 1;
              sm = msk >> 1;
            }
            value1 = Quantised_Multiplier_FC(res[core][i][j + 0], cur_crf, cur_crx);
            value2 = Quantised_Multiplier_FC(res[core][i][j + 1], cur_crf, cur_crx);
            value3 = Quantised_Multiplier_FC(res[core][i][j + 2], cur_crf, cur_crx);
            value4 = Quantised_Multiplier_FC(res[core][i][j + 3], cur_crf, cur_crx);
          }

          svalue1 = value1 + cur_ra;
          svalue2 = value2 + cur_ra;
          svalue3 = value3 + cur_ra;
          svalue4 = value4 + cur_ra;

          if (svalue1 > MAX8) svalue1 = MAX8;
          else if (svalue1 < MIN8) svalue1 = MIN8;
          if (svalue2 > MAX8) svalue2 = MAX8;
          else if (svalue2 < MIN8) svalue2 = MIN8;
          if (svalue3 > MAX8) svalue3 = MAX8;
          else if (svalue3 < MIN8) svalue3 = MIN8;
          if (svalue4 > MAX8) svalue4 = MAX8;
          else if (svalue4 < MIN8) svalue4 = MIN8;

          dout_1 = svalue1.range(7, 0);
          dout_2 = svalue2.range(7, 0);
          dout_3 = svalue3.range(7, 0);
          dout_4 = svalue4.range(7, 0);

          res[core][i][j + 0] = 0;
          res[core][i][j + 1] = 0;
          res[core][i][j + 2] = 0;
          res[core][i][j + 3] = 0;

          ADATA d_out;
          d_out.data = Clamp_Combine(dout_1, dout_2, dout_3, dout_4, MAX8, MIN8);
          if (core == NUM_CORES - 1 && i == pn_block - 1 && j == pm_block - 4) {
            d_out.tlast = true;
          } else {
            d_out.tlast = false;
          }
          dout1.write(d_out);
          DWAIT(27);
        }
      }
    }
    wait();
    DWAIT(2);
    ppuReadyS.write(0);
    pePostTotalS.write(0);
    wait();
  }
}