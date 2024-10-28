int ACCNAME::SHR(int value, int shift) { return value >> shift; }

void ACCNAME::out_dbg(bool block_done, bool vmm_done, bool post_done,
                      bool arranger_done) {
  // sc_int<32> outS_val = 0;
  // bool var0_empty = vars.check_douts_empty(0);
  // bool var1_empty = vars.check_douts_empty(1);
  // bool var2_empty = vars.check_douts_empty(2);
  // bool var3_empty = vars.check_douts_empty(3);
  // outS_val.range(0, 0) = block_done;
  // outS_val.range(1, 1) = vmm_done;
  // outS_val.range(2, 2) = post_done;
  // outS_val.range(3, 3) = arranger_done;

  // outS_val.range(4, 4) = vars.vars_0.dout1.num_available();
  // outS_val.range(5, 5) = vars.vars_0.dout2.num_available();
  // outS_val.range(6, 6) = vars.vars_0.dout3.num_available();
  // outS_val.range(7, 7) = vars.vars_0.dout4.num_available();

  // outS_val.range(8, 8) = vars.vars_1.dout1.num_available();
  // outS_val.range(9, 9) = vars.vars_1.dout2.num_available();
  // outS_val.range(10, 10) = vars.vars_1.dout3.num_available();
  // outS_val.range(11, 11) = vars.vars_1.dout4.num_available();

  // outS_val.range(12, 12) = vars.vars_2.dout1.num_available();
  // outS_val.range(13, 13) = vars.vars_2.dout2.num_available();
  // outS_val.range(14, 14) = vars.vars_2.dout3.num_available();
  // outS_val.range(15, 15) = vars.vars_2.dout4.num_available();

  // outS_val.range(16, 16) = vars.vars_3.dout1.num_available();
  // outS_val.range(17, 17) = vars.vars_3.dout2.num_available();
  // outS_val.range(18, 18) = vars.vars_3.dout3.num_available();
  // outS_val.range(19, 19) = vars.vars_3.dout4.num_available();

  // outS_val.range(20, 20) = var0_empty;
  // outS_val.range(21, 21) = var1_empty;
  // outS_val.range(22, 22) = var2_empty;
  // outS_val.range(23, 23) = var3_empty;
  // outS.write(outS_val);
}

void ACCNAME::Output_Handler() {
  bool ready = false;
  bool resetted = true;
  DATA last = {5000, 1};

  outS.write(0);
  wait();
  while (1) {
//     while (out_check.read() && !ready && resetted) {
//       // check if all VMM dout fifos are empty
//       // check if all VMM compute units are done
//       // check if all VMM post-processing units are done
//       bool post_done, vmm_done, arranger_done;
//       post_done = vmm_done = arranger_done = true;
//       for (int i = 0; i < VMM_COUNT; i++) {
// #pragma HLS unroll
//         vmm_done = vmm_done && vars.check_ready(i);
//         post_done = post_done && vars[i].ppu_done.read();
//         arranger_done = arranger_done && vars.check_douts_empty(i);
//       }
//       bool block_done = !schedule.read();
//       out_dbg(block_done, vmm_done, post_done, arranger_done);
//       ready = block_done && vmm_done && post_done && arranger_done;
//       if (ready) {
//         DWAIT(100);
//         dout1.write(last);
//         dout2.write(last);
//         dout3.write(last);
//         dout4.write(last);
//         out_check.write(0);
//         resetted = false;
//       }
//       wait();
//       // DWAIT(8);
//     }

//     if (!out_check.read()) {
//       resetted = true;
//       ready = false;
//     }
//     wait();
    DWAIT();
  }
}