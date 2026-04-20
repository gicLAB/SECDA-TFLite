#include "acc_config.sc.h"

void ACCNAME::Scheduler() {
#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = \
    out_sig

  readTimeParamS.write(0);
  inpReadReadyS.write(0);
  wgtReadReadyS.write(0);
  biasReadReadyS.write(0);
  peReadyS.write(0);
  ppuReadyS.write(0);
  out_sig.write(0);
  ADATA a;
  wait();
  dout2.write(a);
  dout3.write(a);
  dout4.write(a);
  DWAIT(2);

  while (1) {
    readTimeParamS.write(1);
    layer_t = din1->read().data;
    out_sig.write(1);
    int _mode = din1->read().data;
    mode.write(_mode);
    pN = din1->read().data;
    pM = din1->read().data;
    pK = din1->read().data;
    ra = din1->read().data;
    is_bias = din1->read().data;
    no_rows = din1->read().data;
    no_cols = din1->read().data;

    if (layer_t == 0) {
      crf = din1->read().data;
      crx = din1->read().data;
    }
    readTimeParamS.write(0);
    DWAIT(10);

    if (_mode == MODE_DENSE) {
      for (int n = 0; n < pN; n += pn_block) {
        // [v7 Optimization] Trigger Outer Loop Data Load (Input)
        // Do NOT wait here. Allow Input load to overlap with inner loop setup.
        inpReadReadyS.write(1);
        wait();
        
        for (int m = 0; m < pM; m += (pm_block * NUM_CORES)) {
          // [v7 Optimization] Trigger Inner Loop Data Load (Weights)
          wgtReadReadyS.write(NUM_CORES);
          wait();

          // [v7 Optimization] Wait for PREVIOUS PPU to finish before starting new work
          // This allows PPU to run in parallel with the data loading above.
          while (ppuReadyS.read() != 0) wait();

          // Trigger Bias Load
          biasReadReadyS.write(1);
          wait();

          // [v7 Optimization] Wait for ALL Data Loads (Input + Weights) to complete
          while (inpReadReadyS.read() != 0 || wgtReadReadyS.read() != 0) wait();
          
          // Trigger PE
          peReadyS.write(1);
          wait();

          // [v7 Optimization] Wait for Bias + PE to complete
          while (biasReadReadyS.read() != 0 || peReadyS.read() != 0) wait();

          // Trigger PPU (Fire and Forget - will be checked at start of next loop)
          ppuReadyS.write(1);
          wait();
        }
        // Safety: Ensure input load is fully done before looping (usually handled by inner wait)
        while (inpReadReadyS.read() != 0) wait();
      }
    } else {
      // MODE_MOBILE: Logic swapped (Weights Outer, Inputs Inner)
      for (int m = 0; m < pM; m += pm_block) {
        // Trigger Outer Loop Data Load (Weights)
        wgtReadReadyS.write(1);
        wait();

        for (int n = 0; n < pN; n += (pn_block * NUM_CORES)) {
          // Trigger Inner Loop Data Load (Inputs)
          inpReadReadyS.write(NUM_CORES);
          wait();

          // Wait for PREVIOUS PPU
          while (ppuReadyS.read() != 0) wait();

          // Trigger Bias
          biasReadReadyS.write(1);
          wait();

          // Wait for ALL Data Loads (Weight + Input)
          while (wgtReadReadyS.read() != 0 || inpReadReadyS.read() != 0) wait();

          // Trigger PE
          peReadyS.write(1);
          wait();

          // Wait for Bias + PE
          while (biasReadReadyS.read() != 0 || peReadyS.read() != 0) wait();

          // Trigger PPU
          ppuReadyS.write(1);
          wait();
        }
        // Safety check
        while (wgtReadReadyS.read() != 0) wait();
      }
    }
  }
}

// --- MATH HELPERS (MATCHING V7 EXACTLY) ---
sc_int<64> ACCNAME::mul_s64(int a, sc_int<64> b) { return a * b; }
sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) { return a * b; }
ACC_DTYPE<32> ACCNAME::Clamp_Combine(int i1, int i2, int i3, int i4, int max,
                                     int min) {
  ACC_DTYPE<32> d;
  d.range(7, 0) = i1;
  d.range(15, 8) = i2;
  d.range(23, 16) = i3;
  d.range(31, 24) = i4;
  return d;
}

#ifndef __SYNTHESIS__
// SIMULATION
int ACCNAME::Quantised_Multiplier_Conv(int x, sc_int<32> qm, sc_int<8> shift) {
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> qm64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1);
  sc_int<64> result = x_64 * qm64 + round;
  result = result >> total_shift;
  if (result > MAX32) result = MAX32;
  if (result < MIN32) result = MIN32;
  return (int)result;
}
int ACCNAME::Quantised_Multiplier_FC(int x, int qm, int shift) {
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> qm64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1);
  sc_int<64> result = x_64 * qm64 + round;
  result = result >> total_shift;
  if (result > MAX32) result = MAX32;
  if (result < MIN32) result = MIN32;
  return (int)result;
}
#else
// SYNTHESIS
int ACCNAME::Quantised_Multiplier_Conv(int x, int qm, sc_int<64> pl,
                                       sc_int<32> pr, sc_int<32> msk,
                                       sc_int<32> sm) {
  sc_int<64> val = mul_s64(x, pl);
  if (val > MAX) val = MAX;
  if (val < MIN) val = MIN;
  sc_int<64> val_2 = mul_s64(qm, val);
  sc_int<32> t1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) t1 = (val_2 + NEG) / DIVMAX;
  return (t1 >> pr) + (((t1 & msk) > (sm + ((t1 < 0) & 1))) & 1);
}
int ACCNAME::Quantised_Multiplier_FC(int x, int qm, int shift) {
  sc_int<64> val = mul_s64(x, pl);
  if (val > MAX) val = MAX;
  if (val < MIN) val = MIN;
  sc_int<64> val_2 = mul_s64(qm, val);
  sc_int<32> t1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) t1 = (val_2 + NEG) / DIVMAX;
  return (t1 >> pr) + (((t1 & msk) > (sm + ((t1 < 0) & 1))) & 1);
}
#endif