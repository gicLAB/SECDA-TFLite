
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
    vars[i].wsum_len.write(0);
    vars[i].depthLoadWgt.write(0);
    vars[i].wgt_colno.write(0);
  }
}

void ACCNAME::wgt_len_VMM_arr2(unsigned int depthSwitch,
                               unsigned int wgtBlockArray[VMM_COUNT],
                               bool loadWgtArr[VMM_COUNT],
                               unsigned int depthDiv4) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].wgt_len.write(depthSwitch);
    vars[i].wgt_colno.write(wgtBlockArray[i]);
    vars[i].depthLoadWgt.write(depthDiv4);
  }
  // wait();

  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wgt_write(loadWgtArr[i], i);
  }
  wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wgt_write(false, i);
  }
}

void ACCNAME::wsum_len_VMM_arr2(unsigned int wgtBlockArray[VMM_COUNT],
                                bool loadWgtArr[VMM_COUNT]) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].wsum_len.write(wgtBlockArray[i]);
  }
  // wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wsum_write(loadWgtArr[i], i);
  }
  wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wsum_write(false, i);
  }
}

void ACCNAME::wgt_len_VMM_arr(unsigned int depthSwitch,
                              unsigned int wgtBlockArray[VMM_COUNT],
                              unsigned int depthDiv4) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].wgt_len.write(depthSwitch);
    vars[i].wgt_colno.write(wgtBlockArray[i]);
    vars[i].depthLoadWgt.write(depthDiv4);
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

void ACCNAME::wsum_len_VMM_arr(unsigned int wgtBlockArray[VMM_COUNT]) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].wsum_len.write(wgtBlockArray[i]);
  }
  // wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wsum_write(true, i);
  }
  wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wsum_write(false, i);
  }
}

// noOfWgtBlocks: number of blocks consits of 4 lines each will be transferred
// to specific VMM
// getExtraWgtBlock: some VMM will get extra block of weights when wgtblocks
// are not divisible by VMM_COUNT
void ACCNAME::wgt_len_VMM(unsigned int len, unsigned int noOfWgtBlocks,
                          unsigned int getExtraWgtBlock,
                          unsigned int depthDiv4) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars[i].wgt_len.write(len);
    if (i < getExtraWgtBlock) {
      vars[i].wgt_colno.write(noOfWgtBlocks + 1);
    } else {
      vars[i].wgt_colno.write(noOfWgtBlocks);
    }
    // vars[i].wgt_colno.write(noOfWgtBlocks);
    vars[i].depthLoadWgt.write(depthDiv4);
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

void ACCNAME::wsum_len_VMM(unsigned int noOfWgtBlocks,
                           unsigned int getExtraWgtBlock) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    if (i < getExtraWgtBlock) {
      vars[i].wsum_len.write(noOfWgtBlocks + 1);
    } else {
      vars[i].wsum_len.write(noOfWgtBlocks);
    }
    // vars[i].wsum_len.write(noOfWgtBlocks);
  }
  // wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wsum_write(true, i);
  }
  wait();
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.load_wsum_write(false, i);
  }
}

void ACCNAME::wgt_VMM_write_enable(int index) {
  vars.load_wgt_write(true, index);
  wait();
  vars.load_wgt_write(false, index);
}

void ACCNAME::fill_wgts_VMM(sc_bigint<32 * 4> _data) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    byteToUF data;
    data.data = _data;
    vars.wgt_write(data, i);
  }
}

void ACCNAME::fill_wgts_VMM_individually(sc_bigint<32 * 4> _data, int index) {
  byteToUF data;
  data.data = _data;
  vars.wgt_write(data, index);
}

void ACCNAME::fill_wsums_VMM_all(sc_bigint<32 * 4> _data) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    byteToUF data;
    data.data = _data;
    vars.wsum_write(data, i);
  }
}

void ACCNAME::fill_wsums_VMM_individually(sc_bigint<32 * 4> _data, int index) {
  byteToUF data;
  data.data = _data;
  vars.wsum_write(data, index);
}

void ACCNAME::fill_crf_VMM_all(sc_bigint<32 * 4> _data) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    byteToUF data;
    data.data = _data;
    vars.crf_write(data, i);
  }
}

void ACCNAME::fill_crf_VMM_individually(sc_bigint<32 * 4> _data, int index) {
  byteToUF data;
  data.data = _data;
  vars.crf_write(data, index);
}

void ACCNAME::fill_crx_VMM_all(sc_int<32> _data) {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    vars.crx_write(_data, i);
  }
}

void ACCNAME::fill_crx_VMM_individually(sc_int<32> _data, int index) {
  vars.crx_write(_data, index);
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
    // cout << "vars[" << i << "].ready: " << vars.check_ready(i) << endl;
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

void ACCNAME::post_ready_VMM() {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    while (!vars.check_post_ready(i)) wait();
    DWAIT();
  }
}

void ACCNAME::ppu_done_VMM() {
  for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
    while (!vars.check_ppu_done(i)) wait();
    // cout << "vars[" << i << "].ppu_done: " << vars.check_ppu_done(i) <<
    // endl;
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
                                unsigned int wsum_idx, unsigned int depth) {
  while (!vars.check_ready(unit)) wait();
  vars.start_compute(unit, w_idx, wsum_idx, depth, ra);
  while (vars.check_ready(unit)) wait();
  vars.set_compute(unit, false);
}