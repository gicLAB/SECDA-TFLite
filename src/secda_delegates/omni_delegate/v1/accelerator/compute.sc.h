ACC_DTYPE<32> ACCNAME::Clamp_Combine(int i1, int i2, int i3, int i4, int qa_max,
                                     int qa_min) {
  if (i1 < qa_min) i1 = qa_min;
  if (i1 > qa_max) i1 = qa_max;
  if (i2 < qa_min) i2 = qa_min;
  if (i2 > qa_max) i2 = qa_max;
  if (i3 < qa_min) i3 = qa_min;
  if (i3 > qa_max) i3 = qa_max;
  if (i4 < qa_min) i4 = qa_min;
  if (i4 > qa_max) i4 = qa_max;

  ACC_DTYPE<32> d;
  d.range(7, 0) = i1;
  d.range(15, 8) = i2;
  d.range(23, 16) = i3;
  d.range(31, 24) = i4;

  return d;
}

void ACCNAME::send_parameters_omni_PE(int length, sc_fifo_in<ADATA> *din) {
  int lshift = (1 << din->read().data);
  int in1_off = din->read().data;
  int in1_sv = din->read().data;
  int in1_mul = din->read().data;
  int in2_off = din->read().data;
  int in2_sv = din->read().data;
  int in2_mul = din->read().data;
  int out1_off = din->read().data;
  int out1_sv = din->read().data;
  int out1_mul = din->read().data;

  for (int i = 0; i < omni_PE_COUNT; i++) {
#pragma HLS unroll
    omni_pe_array[i].length = length;
    omni_pe_array[i].lshift = lshift;
    omni_pe_array[i].in1_off = in1_off;
    omni_pe_array[i].in1_sv = in1_sv;
    omni_pe_array[i].in1_mul = in1_mul;
    omni_pe_array[i].in2_off = in2_off;
    omni_pe_array[i].in2_sv = in2_sv;
    omni_pe_array[i].in2_mul = in2_mul;
    omni_pe_array[i].out1_off = out1_off;
    omni_pe_array[i].out1_sv = out1_sv;
    omni_pe_array[i].out1_mul = out1_mul;
  }
};

#ifndef __SYNTHESIS__
void ACCNAME::Counter() {
  wait();
  while (1) {
    per_batch_cycles->value++;
    if (computeS.read() == 1) active_cycles->value++;
    wait();
  }
}
#endif

void ACCNAME::Compute() {

#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = \
    computeSS

  ADATA d;
  int f_out[4];
  computeS.write(0);
  computeSS.write(0);
  omni_pe_array[0].start.write(0);
  int submodule_idx = 0;
  wait();
  while (1) {
    computeS.write(0);
    DWAIT();

    int length = din1.read().data;
    computeS.write(1);
    computeSS.write(2);
    DWAIT();

    send_parameters_omni_PE(length, &din1);
    int qa_max = din1.read().data;
    int qa_min = din1.read().data;
    wait();
    computeS.write(2);
    computeSS.write(2);

    for (int i = 0; i < length; i++) {
      omni_pe_array.input_fifo_write(submodule_idx,
                                          din1.read().data.to_int());
      omni_pe_array.input_fifo_write(submodule_idx,
                                          din1.read().data.to_int());
      computeS.write(3);
      computeSS.write(3);
      wait();
      f_out[0] = omni_pe_array.output_fifo_read(submodule_idx);
      f_out[1] = omni_pe_array.output_fifo_read(submodule_idx);
      f_out[2] = omni_pe_array.output_fifo_read(submodule_idx);
      f_out[3] = omni_pe_array.output_fifo_read(submodule_idx);
      d.data =
          Clamp_Combine(f_out[0], f_out[1], f_out[2], f_out[3], qa_max, qa_min);
      if (i + 1 == length) d.tlast = true;
      else d.tlast = false;
      dout1.write(d);
      submodule_idx = (submodule_idx + 1) % omni_PE_COUNT;
    }
    DWAIT();
  }
}