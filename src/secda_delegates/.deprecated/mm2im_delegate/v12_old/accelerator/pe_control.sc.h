void ACCNAME::start_compute_lim(int limit) {
  for (int i = 0; i < limit; i++) {
    compute_reg[i] = true;
  }
  wait();
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].compute.write(true);
  }
}

bool ACCNAME::compute_done_lim(int limit) {
#pragma HLS inline OFF
  bool loop = false;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    compute_done_reg[i] = vars[i].compute_done.read();
  }
  wait();
  for (int i = 0; i < limit; i++) {
    loop = loop || !compute_done_reg[i];
  }
  DWAIT(2);
  return loop;
}

void ACCNAME::stop_compute_lim(int limit) {
  for (int i = 0; i < limit; i++) {
    compute_reg[i] = false;
    reset_compute_reg[i] = true;
  }
  wait();
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].compute.write(compute_reg[i]);
    vars[i].reset_compute.write(reset_compute_reg[i]);
  }
  DWAIT();
}

bool ACCNAME::compute_resetted_lim(int limit) {
#pragma HLS inline OFF
  bool loop = false;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    compute_done_reg[i] = vars[i].compute_done.read();
  }
  wait();
  for (int i = 0; i < limit; i++) {
    loop = loop || compute_done_reg[i];
  }
  DWAIT(2);
  return loop;
}

void ACCNAME::init_PE_signals() {
#pragma HLS inline OFF
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    online_reg[i] = false;
    compute_reg[i] = false;
    start_addr_p_reg[i] = 0;
    send_len_p_reg[i] = 0;
    bias_data_reg[i] = 0;
    crf_data_reg[i] = 0;
    crx_data_reg[i] = 0;
    ra_data_reg[i] = 0;
    send_reg[i] = false;
    cols_per_filter_reg[i] = cols_per_filter;
    depth_reg[i] = depth;
    process_cal_reg[i] = false;
  }
  wait();
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].online.write(online_reg[i]);
    vars[i].compute.write(compute_reg[i]);
    vars[i].start_addr_p.write(start_addr_p_reg[i]);
    vars[i].send_len_p.write(send_len_p_reg[i]);
    vars[i].bias_data.write(bias_data_reg[i]);
    vars[i].crf_data.write(crf_data_reg[i]);
    vars[i].crx_data.write(crx_data_reg[i]);
    vars[i].ra_data.write(ra_data_reg[i]);
    vars[i].send.write(send_reg[i]);
    vars[i].cols_per_filter.write(cols_per_filter_reg[i]);
    vars[i].depth.write(depth_reg[i]);
    vars[i].process_cal.write(process_cal_reg[i]);
  }
  DWAIT();
}

bool ACCNAME::wgt_loaded_lim(int limit) {
#pragma HLS inline OFF
  bool loop = true;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    wgt_loaded_reg[i] = vars[i].wgt_loaded.read();
  }
  wait();
  for (int i = 0; i < limit; i++) {
    loop = loop && wgt_loaded_reg[i];
  }
  return loop;
  DWAIT(2);
}

bool ACCNAME::store_done_lim(int limit) {
#pragma HLS inline OFF
  bool loop = false;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    send_done_reg[i] = vars[i].send_done.read();
  }
  wait();
  for (int i = 0; i < limit; i++) {
    if (!send_done_reg[i]) loop = true;
  }
  DWAIT(2);
  return loop;
}

void ACCNAME::activate_PEs_lim(int limit) {
  // #pragma HLS inline OFF
  for (int i = 0; i < limit; i++) {
    online_reg[i] = true;
    cols_per_filter_reg[i] = cols_per_filter;
    depth_reg[i] = depth;
    oh_reg[i] = oh;
    ow_reg[i] = ow;
    kernel_size_reg[i] = kernel_size;
    stride_x_reg[i] = stride_x;
    stride_y_reg[i] = stride_y;
    pt_reg[i] = pt;
    pl_reg[i] = pl;
    width_col_reg[i] = width_col;
  }
  wait();
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].online.write(online_reg[i]);
    vars[i].cols_per_filter.write(cols_per_filter_reg[i]);
    vars[i].depth.write(depth_reg[i]);
    vars[i].oh.write(oh_reg[i]);
    vars[i].ow.write(ow_reg[i]);
    vars[i].kernel_size.write(kernel_size_reg[i]);
    vars[i].stride_x.write(stride_x_reg[i]);
    vars[i].stride_y.write(stride_y_reg[i]);
    vars[i].pt.write(pt_reg[i]);
    vars[i].pl.write(pl_reg[i]);
    vars[i].width_col.write(width_col_reg[i]);
  }
  wait();
}

void ACCNAME::deactivate_PEs_lim(int limit) {
#pragma HLS inline OFF
  for (int i = 0; i < limit; i++) {
    online_reg[i] = false;
  }
  wait();
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].online.write(online_reg[i]);
  }
  DWAIT();
}

void ACCNAME::col_indices_write_lim(int data, bool tlast, int limit) {
  for (int i = 0; i < limit; i++) {
    vars.col_indices_fifo_write(data, tlast, i);
  }
}

void ACCNAME::out_indices_write_lim(int data, bool tlast, int limit) {
  for (int i = 0; i < limit; i++) {
    vars.out_indices_fifo_write(data, tlast, i);
  }
}


void ACCNAME::col_out_indices_write_lim(int col, int out, bool tlast, int limit) {
  for (int i = 0; i < limit; i++) {
    #pragma HLS pipeline II=1
    {
      #pragma HLS latency min=1 max=1
      vars.col_indices_fifo_write(col, tlast, i);
      vars.out_indices_fifo_write(out, tlast, i);
    }
  }
}

