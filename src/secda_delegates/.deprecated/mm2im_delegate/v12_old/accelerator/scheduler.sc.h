#include "acc.sc.h"

// Generalise for any number of PEs
void ACCNAME::load_inp_PEs() {
  // load & process one row inputs to PE at a time
  load_fifo.write(true);
  start_decode.write(true);
  DWAIT();
  for (int r = 0; r < number_of_rows; r++) {
    scheduleS.write(62);
    wait();
    start_compute_lim(nfilters);
    scheduleS.write(63);
    DWAIT(5);

    while (compute_done_lim(nfilters)) {
      scheduleS.write(66);
      DWAIT();
    }
    // wait();
    scheduleS.write(67);
    stop_compute_lim(nfilters);
    DWAIT();

    while (compute_resetted_lim(nfilters)) {
      scheduleS.write(68);
      DWAIT();
    }

    for (int i = 0; i < nfilters; i++) {
      reset_compute_reg[i] = false;
    }
    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      vars[i].reset_compute.write(reset_compute_reg[i]);
    }
    DWAIT();
    scheduleS.write(69);
    // wait();
  }
  while (load_fifo || start_decode) {
    // scheduleS.write(691);
    DWAIT();
  }
}

void ACCNAME::store(int start, int length) {
#pragma HLS inline OFF
  scheduleS.write(71);
  output_handler.write(true);
  wait();
  output_handler.write(false);
  wait();

  for (int i = 0; i < nfilters; i++) {
    start_addr_p_reg[i] = start;
    send_len_p_reg[i] = length;
    bias_data_reg[i] = bias_buf[i];
    crf_data_reg[i] = crf_buf[i];
    crx_data_reg[i] = crx_buf[i];
    crx_scale_reg[i] = crx_scale_buf[i];
    ra_data_reg[i] = ra;
    send_reg[i] = true;
  }

  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].start_addr_p.write(start_addr_p_reg[i]);
    vars[i].send_len_p.write(send_len_p_reg[i]);
    vars[i].bias_data.write(bias_data_reg[i]);
    vars[i].crf_data.write(crf_data_reg[i]);
    vars[i].crx_data.write(crx_data_reg[i]);
    // vars[i].crx_scale_data.write(crx_scale_reg[i]);
    vars.crx_scale_data_write(crx_scale_reg[i], i);
    vars[i].ra_data.write(ra_data_reg[i]);
  }
  scheduleS.write(72);
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].send.write(send_reg[i]);
  }

  scheduleS.write(73);
  DWAIT(5);
  while (store_done_lim(nfilters)) {
    // scheduleS.write(73);
    DWAIT();
  }
  scheduleS.write(74);

  for (int i = 0; i < nfilters; i++) {
    send_reg[i] = false;
  }

  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].send.write(send_reg[i]);
  }
  scheduleS.write(75);
  DWAIT();
}

void ACCNAME::Scheduler() {
  bool ready = true;
  scheduleS.write(0);
  schedule.write(false);
  init_PE_signals();
  wait();
  while (1) {
    scheduleS.write(1);
    while (!schedule.read()) wait();
    scheduleS.write(2);
    wait();
    wait();
    wait();
    scheduleS.write(3);
    while (ready) {
      scheduleS.write(4);
      wait();
      opcode op = opcode(din1.read().data.to_uint());
      DWAIT();
      ready = op.schedule;
      if (op.load_inp) {
        inp_packet ip = inp_packet(&din1);
        srow = ip.srow;
        number_of_rows = ip.inp_rows;
        DWAIT();
        scheduleS.write(6);
        wait();
        wait();
        DWAIT(2);
        load_inp_PEs();
      }
      if (op.store) {
        scheduleS.write(7);
        int send_start = din1.read().data;
        int s_len = din1.read().data;
        send_len = s_len;
        DWAIT(2);
        store(send_start, send_len);
      }
    }
    scheduleS.write(9);
    deactivate_PEs_lim(nfilters);
    ready = true;
    schedule.write(false);
    wait();
  }
}
