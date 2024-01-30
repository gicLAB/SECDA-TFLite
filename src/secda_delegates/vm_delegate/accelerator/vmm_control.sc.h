
void ACCNAME::init_VMM() {
#pragma HLS inline OFF
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].load_inp.write(false);
    vars[i].compute.write(false);
    vars[i].depth.write(0);
    vars[i].w_idx.write(0);
    vars[i].inp_len.write(0);
    vars[i].send_done.write(0);
  }
  DWAIT();
}

void ACCNAME::init_wgts_VMM() {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].load_wgt.write(false);
    vars[i].wgt_len.write(0);
  }
}

void ACCNAME::wgt_len_VMM(unsigned int len) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].wgt_len.write(len);
  }
  // wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wgt_write(true, i);
  }
  wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wgt_write(false, i);
  }
}

void ACCNAME::fill_wgts_VMM(sc_bigint<32 * 4> _data) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    byteToUF data;
    data.data = _data;
    vars.wgt_write(data, i);
  }
}

void ACCNAME::inp_len_VMM(unsigned int len) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.inp_len_write(len, i);
  }
  // wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_inp_write(true, i);
  }
  wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_inp_write(false, i);
  }
}

void ACCNAME::fill_inps_VMM(sc_bigint<32 * 4> _data) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    bUF data = _data;
    vars.inp_write(data, i);
  }
}

void ACCNAME::wait_ready_VMM() {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    while (!vars.check_ready(i)) wait();
    DWAIT();
  }
}

void ACCNAME::vmm_ready_VMM() {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    while (!vars.check_vmm_ready(i)) wait();
    DWAIT();
  }
}

void ACCNAME::send_done_write_VMM(int unit) {
  vars.send_done_write(true, unit);
  while (!vars[unit].ppu_done.read()) wait();
  vars.send_done_write(false, unit);
  wait();
}

void ACCNAME::start_compute_VMM(unsigned int unit, unsigned int w_idx,
                                unsigned int depth) {
  while (!vars.check_ready(unit)) wait();
  vars.start_compute(unit, w_idx, depth, ra);
  while (vars.check_ready(unit)) wait();
  vars.set_compute(unit, false);

}