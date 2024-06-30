
void ACCNAME::Input_Handler() {
  // clang-format off
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=outS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=schS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=p1S

  // clang-format on

  inS.write(0);
  wait();
  while (1) {

    inS.write(1);
    opcode op = opcode(din1.read().data.to_uint());
    if (op.config) {
      config_packet cnp = config_packet(&din1);
      depth = cnp.depth;
      ra = cnp.ra;
      inS.write(2);
      DWAIT();
    }
    if (op.load_wgt || op.load_inp) {
      load_wgt.write(op.load_wgt);
      load_inp.write(op.load_inp);
      load_data.write(1);
      inS.write(3);
      DWAIT();
      while (load_data.read()) wait();
    }
    if (op.compute) {
      compute_packet cp = compute_packet(&din1);
      inp_block = cp.inp_block;
      wgt_block = cp.wgt_block;
      schedule.write(1);
      out_check.write(1);
      inS.write(4);
      DWAIT();
      while (schedule.read()) wait();
    }
  }
}
