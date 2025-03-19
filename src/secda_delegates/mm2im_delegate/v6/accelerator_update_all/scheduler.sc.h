#include "acc.sc.h"

void ACCNAME::Pattern_Decoder() {
  pdS.write(0);
  wait();
  while (1) {
    pdS.write(1);
    while (!start_decode.read()) wait();
      // us there anyway to calculate this once and then simply increment
#pragma HLS PIPELINE II = 1
    int crow = srow;
    for (int i = 0; i < number_of_rows; i++) {
#pragma HLS PIPELINE II = 1
      pdS.write(2);
      int cal_col = crow % width_col;
      int cal_row = crow / width_col;
      int h_pad = -pt + (stride_y * cal_row);
      int w_pad = -pl + (stride_x * cal_col);
      int im_dex = (h_pad * ow + w_pad);
      int row = 0;
      int pouts = 0;
      DWAIT(48);
      pdS.write(3);
      for (int ih = 0; ih < kernel_size; ih++) {
#pragma HLS PIPELINE II = 1
        for (int iw = 0; iw < kernel_size; iw++) {
#pragma HLS PIPELINE II = 1
          pdS.write(31);
          if (ih + h_pad >= 0 and ih + h_pad < oh and iw + w_pad >= 0 and
              iw + w_pad < ow) {
            col_indices_write_lim(row, false, nfilters);
            out_indices_write_lim(im_dex, false, nfilters);
          }
          pdS.write(32);
          im_dex += 1;
          row += 1;
          DWAIT();
        }
        im_dex += (ow - kernel_size);
        DWAIT(3);
      }
      col_indices_write_lim(0, true, nfilters);
      out_indices_write_lim(0, true, nfilters);
      crow += 1;
      DWAIT();
    }
    start_decode.write(false);
    DWAIT();
  }
}

void ACCNAME::FIFO_Loader() {
  wait();
  while (1) {
    while (!load_fifo) wait();
    for (int r = 0; r < number_of_rows; r++) {
      for (int d = 0; d < depth; d++) {
#pragma HLS PIPELINE II = 4
        bUF d1[PE_COUNT];
        acc_dt array[UF];
        for (int u = 0; u < UF; u += 4) {
#pragma HLS unroll
          sc_uint<32> data = din1.read().data;
          array[u + 0] = data.range(7, 0).to_int();
          array[u + 1] = data.range(15, 8).to_int();
          array[u + 2] = data.range(23, 16).to_int();
          array[u + 3] = data.range(31, 24).to_int();
        }
        for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
          d1[i].insert(array);
        }

        for (int i = 0; i < nfilters; i++) {
          vars.inp_write(d1[i], i);
        }
        DWAIT(4);
      }
      DWAIT(2);
    }
    load_fifo.write(false);
    DWAIT();
  }
}

// Generalise for any number of PEs
void ACCNAME::load_inp_PEs() {
  // load & process one row inputs to PE at a time

  // scheduleS.write(61);
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
    update_writes();

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
    ra_data_reg[i] = ra;
    send_reg[i] = true;
  }
  update_writes();

  // scheduleS.write(711);

  // scheduleS.write(72);
  DWAIT(5);
  while (store_done_lim(nfilters)) {
    // scheduleS.write(73);
    DWAIT();
  }
  scheduleS.write(74);

  for (int i = 0; i < nfilters; i++) {
    send_reg[i] = false;
  }
  update_writes();

  scheduleS.write(75);
  DWAIT();
}

void ACCNAME::Scheduler() {
  bool ready = true;
  scheduleS.write(0);
  schedule.write(false);
  tempS.write(0);
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
