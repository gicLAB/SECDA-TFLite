
void ACCNAME::Input_Handler() {
  // clang-format off
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=rmax
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=lmax
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=outS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=schS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=p1S

#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=w1SS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=w2SS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=w3SS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=w4SS

// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_0.computeS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_1.computeS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_2.computeS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_3.computeS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_0.postS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_1.postS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_2.postS
// #pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_3.postS
  // clang-format on

#ifdef __SYNTHESIS__
  // vars.vars_0.computeS.write(0);
  // vars.vars_1.computeS.write(0);
  // vars.vars_2.computeS.write(0);
  // vars.vars_3.computeS.write(0);
  // vars.vars_0.postS.write(0);
  // vars.vars_1.postS.write(0);
  // vars.vars_2.postS.write(0);
  // vars.vars_3.postS.write(0);
#endif

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
