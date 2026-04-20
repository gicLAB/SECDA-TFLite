#include "acc_config.sc.h"

// Isolated pure function to prevent Vivado HLS scheduling bugs in unrolled loops.
// This enforces a strict 32x32 multiplier boundary to save DSPs, while shielding
// the complex bit-shifts from the AXI-Stream state machine.
inline int hw_quantize(int x, int qm, signed char crx) {
    long long pl = (crx > 0) ? crx : 1;
    int pr = (crx <= 0) ? -crx : 0;
    int msk = (crx <= 0) ? ((1 << pr) - 1) : 0;
    int sm = (crx <= 0) ? (msk >> 1) : 0;

    long long val = (long long)x * pl;
    if (val > MAX) val = MAX;
    if (val < MIN) val = MIN;

    // Strict 32-bit cast guarantees a 4-DSP multiplier instead of a 16-DSP multiplier
    int val_32 = (int)val;
    long long val_2 = (long long)val_32 * (long long)qm;

    int temp_1 = (int)((val_2 + POS) / DIVMAX);
    if (val_2 < 0) temp_1 = (int)((val_2 + NEG) / DIVMAX);

    int val_3 = temp_1 >> pr;
    int temp_2 = temp_1 & msk;
    int temp_5 = (temp_2 > (sm + ((temp_1 < 0) ? 1 : 0))) ? 1 : 0;

    return val_3 + temp_5;
}

void ACCNAME::PPU() {
  pePostTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (ppuReadyS.read() == 0) wait();
    pePostTotalS.write(1);

    // Cache parameters to prevent concurrent access issues
    int cur_ra = ra;
    int cur_layer_t = layer_t;
    int cur_fc_crf = crf;
    signed char cur_fc_crx = (signed char)crx;

    DWAIT(1);

    for (int i = 0; i < pn_block; i++) {
      // RESTORED: j+=4 for maximum inference speed
      for (int j = 0; j < pm_block; j += 4) {
#pragma HLS PIPELINE II = 1

        ADATA d_out1, d_out2, d_out3, d_out4;
        long long svalue[NUM_CORES][4];

        for (int core = 0; core < NUM_CORES; core++) {
#pragma HLS UNROLL
          int i_sum_val = in_sum[core][i];

          res[core][i][j + 0] += bias[core][j + 0] + wt_sum[core][j + 0] + i_sum_val;
          res[core][i][j + 1] += bias[core][j + 1] + wt_sum[core][j + 1] + i_sum_val;
          res[core][i][j + 2] += bias[core][j + 2] + wt_sum[core][j + 2] + i_sum_val;
          res[core][i][j + 3] += bias[core][j + 3] + wt_sum[core][j + 3] + i_sum_val;

          int value[4];

          if (cur_layer_t == 1) { // CONV Layer
            for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
              int local_crf = crf_v[core][j + c];
              signed char cmp_crx = (signed char)crx_v[core][j + c];
              value[c] = hw_quantize(res[core][i][j + c], local_crf, cmp_crx);
            }
          } else { // FC Layer
            for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
              value[c] = hw_quantize(res[core][i][j + c], cur_fc_crf, cur_fc_crx);
            }
          }

          svalue[core][0] = (long long)value[0] + cur_ra;
          svalue[core][1] = (long long)value[1] + cur_ra;
          svalue[core][2] = (long long)value[2] + cur_ra;
          svalue[core][3] = (long long)value[3] + cur_ra;

          for (int c = 0; c < 4; c++) {
#pragma HLS UNROLL
            if (svalue[core][c] > MAX8) svalue[core][c] = MAX8;
            else if (svalue[core][c] < MIN8) svalue[core][c] = MIN8;
          }

          res[core][i][j + 0] = 0;
          res[core][i][j + 1] = 0;
          res[core][i][j + 2] = 0;
          res[core][i][j + 3] = 0;
        }

        bool is_last = (i == pn_block - 1 && j == pm_block - 4);

        d_out1.data = Clamp_Combine((int)(svalue[0][0] & 0xFF), (int)(svalue[0][1] & 0xFF), (int)(svalue[0][2] & 0xFF), (int)(svalue[0][3] & 0xFF), MAX8, MIN8);
        d_out2.data = Clamp_Combine((int)(svalue[1][0] & 0xFF), (int)(svalue[1][1] & 0xFF), (int)(svalue[1][2] & 0xFF), (int)(svalue[1][3] & 0xFF), MAX8, MIN8);
        d_out3.data = Clamp_Combine((int)(svalue[2][0] & 0xFF), (int)(svalue[2][1] & 0xFF), (int)(svalue[2][2] & 0xFF), (int)(svalue[2][3] & 0xFF), MAX8, MIN8);
        d_out4.data = Clamp_Combine((int)(svalue[3][0] & 0xFF), (int)(svalue[3][1] & 0xFF), (int)(svalue[3][2] & 0xFF), (int)(svalue[3][3] & 0xFF), MAX8, MIN8);

        d_out1.tlast = is_last;
        d_out2.tlast = is_last;
        d_out3.tlast = is_last;
        d_out4.tlast = is_last;

        dout1.write(d_out1);
        dout2.write(d_out2);
        dout3.write(d_out3);
        dout4.write(d_out4);
        
        DWAIT(27);
      }
    }
    wait();
    DWAIT(2);
    ppuReadyS.write(0);
    pePostTotalS.write(0);
    wait();
  }
}