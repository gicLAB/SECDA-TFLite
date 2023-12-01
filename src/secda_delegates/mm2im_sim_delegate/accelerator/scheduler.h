#include "acc.sc.h"

// Remove PE specific code as much as possible below
// Generalise for any number of PEs
void ACCNAME::load_wgt_PEs() {
  int ocols[PE_COUNT];
#pragma HLS array_partition variable = ocols complete

  scheduleS.write(31);
  wait();

  for (int c = 0; c < cols_per_filter; c++) {
    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      int pe_dex = c + pe_cols[i];
      ocols[i] = pe_dex * depth;
      vars[i].col_dexs_fifo.write(wgt_sum_buf[pe_dex]);
    }
    for (int d = 0; d < depth; d++) {
#pragma HLS loop_tripcount min = 16 max = 16 avg = 16
      bUF d1[8];
      for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
        d1[i].insert(wgt_buf, ocols[i] + d);
      }
      for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
        vars.wgt_write(d1[i], i);
      }
      DWAIT(7);
    }
    DWAIT(7);
  }

  scheduleS.write(32);
  wait();

  while (!wgt_loaded()) wait();

  scheduleS.write(33);
  wait();
}

void ACCNAME::FIFO_Loader() {
  wait();
  while (1) {

    while (!load_fifo) wait();

    for (int r = 0; r < number_of_rows; r++) {
      int col_indice_start = col_indice_starts[r];
      int col_indice_len = col_indice_lens[r];
      int orow = r * depth;
      DWAIT(5);

      for (int d = 0; d < depth; d++) {
#pragma HLS PIPELINE II = 1
        bUF d1[8];
        for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
          d1[i].insert(inp_buf, orow + d);
        }
        for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
          vars.inp_write(d1[i], i);
        }
        DWAIT(3);
      }
      DWAIT();

      for (int j = 0; j < col_indice_len; j++) {
        for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
          vars[i].col_dexs_fifo.write(col_indices[col_indice_start + j]);
        }
        DWAIT(4);
      }
      for (int j = 0; j < col_indice_len; j++) {
        for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
          vars[i].dex_fifo.write(out_indices[col_indice_start + j]);
        }
        DWAIT(4);
      }
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
  for (int r = 0; r < number_of_rows; r++) {
    // send the cols_indices to perform vector product with the input row
    int col_indice_start = col_indice_starts[r];
    int col_indice_len = col_indice_lens[r];
    int orow = r * depth;

    // scheduleS.write(62);
    start_compute(col_indice_len);
    // scheduleS.write(63);
    DWAIT(5);

    while (compute_done()) {
      // scheduleS.write(66);
      DWAIT();
    }
    // wait();
    // scheduleS.write(67);
    stop_compute();
    DWAIT();

    while (compute_resetted()) {
      // scheduleS.write(68);
      DWAIT();
    }

    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      vars[i].reset_compute.write(false);
    }
    DWAIT();

    // scheduleS.write(69);
    // wait();
  }
  while (load_fifo) {
    // scheduleS.write(691);
    DWAIT();
  }
}

void ACCNAME::store(int start, int length) {
#pragma HLS inline OFF
  scheduleS.write(71);
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    // vars[i].send.write(true);
    vars[i].start_addr_p.write(start);
    vars[i].send_len_p.write(length);
    vars[i].bias_data.write(bias_buf[i]);
    vars[i].crf_data.write(crf_buf[i]);
    vars[i].crx_data.write(crx_buf[i]);
    vars[i].ra_data.write(ra);
  }
  // scheduleS.write(711);

  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].send.write(true);
  }

  tempS.write(length);
  // scheduleS.write(72);
  DWAIT(5);
  while (store_done()) {
    // scheduleS.write(73);
    DWAIT();
  }
  scheduleS.write(74);
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].send.write(false);
  }
  scheduleS.write(75);
  DWAIT();
}

void ACCNAME::Scheduler() {
  init_PE_signals();
  bool ready = true;
  scheduleS.write(0);
  schedule.write(false);
  tempS.write(0);
  wait();
  while (1) {

    scheduleS.write(1);
    while (!schedule.read()) wait();

    scheduleS.write(2);
    wait();
    if (schedule) activate_PEs();
    wait();
    wait();
    scheduleS.write(3);
    load_wgt_PEs();

    while (ready) {
      scheduleS.write(4);
      wait();
      opcode op = opcode(din1.read().data.to_uint());
      DWAIT();
      ready = op.schedule;
      if (op.load_inp || op.load_col_map) {
        if (op.load_inp) load_inp.write(true);
        if (op.load_col_map) load_col_map.write(true);
        load_data.write(true);
        scheduleS.write(5);
        wait();
        while (load_data.read()) wait();
        load_inp.write(false);
        load_col_map.write(false);
        DWAIT(3);
      }
      if (op.load_col_map) {
        scheduleS.write(6);
        load_inp_PEs();
      }
      if (op.store) {
        scheduleS.write(7);
        int send_start = din1.read().data;
        int send_len = din1.read().data;
        DWAIT(2);
        store(send_start, send_len);
      }
    }
    scheduleS.write(9);
    deactivate_PEs();
    ready = true;
    schedule.write(false);
    wait();
  }
}
