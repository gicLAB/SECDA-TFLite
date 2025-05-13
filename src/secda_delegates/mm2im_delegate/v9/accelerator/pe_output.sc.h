#include "acc_config.sc.h"

int PE::Quantised_Multiplier_gemmlowp(int x, int qm, sc_int<8> shift) {
  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;
  if (shift > 0) {
    pl = shift;
    pr = 0;
    msk = 0;
    sm = 0;
  } else {
    // pl = 1;
    pl = 0;
    pr = -shift;
    msk = (1 << -shift) - 1;
    sm = msk >> 1;
  }
  sc_int<64> val = x * (1 << pl);
  if (val > MAX) val = MAX;
  if (val < MIN) val = MIN;
  sc_int<64> val_2 = val * qm;
  sc_int<32> temp_1;
  temp_1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
  sc_int<32> val_3 = temp_1;
  val_3 = val_3 >> pr;
  sc_int<32> temp_2 = temp_1 & msk;
  sc_int<32> temp_3 = (temp_1 < 0) & 1;
  sc_int<32> temp_4 = sm + temp_3;
  sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
  sc_int<32> result_32 = val_3 + temp_5;
  int res = result_32;
  return result_32;
}

int PE::Quantised_Multiplier_ruy_reference(int x, int qm, sc_int<8> shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
  sc_int<64> result =
      x_64 * quantized_multiplier_64 + round; // ALU ADD + ALU MUL
  result = result >> total_shift;             // ALU SHRI
  int nresult = result;
  if (result > MAX) result = MAX; // ALU MIN
  if (result < MIN) result = MIN; // ALU MAX
  sc_int<32> result_32 = result;
  return result_32;
}

void PE::Out() {
  ADATA last = {4000, 1};
  ADATA d = {0, 0};
  int start_addr = 0;
  int send_len = 0;
  int bias;
  int crf;
  int ra = 0;
  out_done.write(false);
  send_done.write(false);
  sendS.write(0);
  sc_int<32> crx;
  
  wait();
  while (1) {
    while (!out.read() && !send.read()) {
      sendS.write(1);
      DWAIT();
    }
    sendS.write(2);
    if (send) {
      sendS.write(3);
      start_addr = start_addr_p.read();
      send_len = send_len_p.read();
      bias = bias_data.read();
      crf = crf_data.read();
      crx = crx_data.read();
      ra = ra_data.read();
      sendS.write(31);
      // sendS.write(send_len);

      for (int i = 0; i < send_len; i++) {
#pragma HLS PIPELINE II = 1
        int dex = (start_addr + i) % PE_ACC_BUF_SIZE;
#ifndef __SYNTHESIS__
        int qm_ret = ra + Quantised_Multiplier_gemmlowp(acc_store[dex] + bias,
                                                        crf, crx.range(7, 0));
#else
        // int qm_ret = ra + Quantised_Multiplier_ruy_reference(
        // acc_store[dex] + bias, crf, crx.range(7, 0));
        int qm_ret = ra + Quantised_Multiplier_gemmlowp(acc_store[dex] + bias,
                                                        crf, crx.range(7, 0));
#endif

        if (qm_ret > MAX8) qm_ret = MAX8;
        else if (qm_ret < MIN8) qm_ret = MIN8;
        d.data = qm_ret;

        if (i + 1 == send_len) d.tlast = 1;
        else d.tlast = 0;
        out_fifo_out.write(d);
        // sendS.write(32);
        DWAIT(2);
      }
      DWAIT(19);
      for (int i = 0; i < send_len; i++) {
        int dex = (start_addr + i) % PE_ACC_BUF_SIZE;
        acc_store[dex] = 0;
        DWAIT(3);
      }
      sendS.write(33);
      send_done.write(true);
      DWAIT();
    }

    if (out) {
      sendS.write(4);
      ADATA d = out_indices_fifo.read();
      while (!d.tlast) {
        int c = d.data;
        int dex = d.data % PE_ACC_BUF_SIZE;
        sendS.write(41);
        int out_data = temp_fifo_in.read();
        acc_store[dex] += out_data;
        sendS.write(42);
        d = out_indices_fifo.read();
        DWAIT(4);
      }

      out_done.write(true);
    }

    sendS.write(5);
    wait();
    while (out || send) {
      sendS.write(6);
      DWAIT();
    }
    out_done.write(false);
    send_done.write(false);
    DWAIT();
  }
}