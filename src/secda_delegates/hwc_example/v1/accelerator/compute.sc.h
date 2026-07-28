void ACCNAME::Recv() {

  done.write(0);
  bool started = start.read();
  compute.write(false);
  send.write(false);
  HWC_SIG(Recv, 0);
  wait();
  while (1) {
    HWC_SIG(Recv, 1);
    wait();

    opcode packet(din1.read().data);
    code_extension op_args(din1.read().data);
    acc_args = op_args;

    if (packet.load_X) {
      unsigned int read_length = op_args.L;
      for (int i = 0; i < read_length; i++) {
        X_buffer[i] = din1.read().data;
        DWAIT();
      }
    }

    if (packet.load_Y) {
      unsigned int read_length = op_args.L;
      for (int i = 0; i < read_length; i++) {
        Y_buffer[i] = din1.read().data;
        DWAIT();
      }
    }

    // Computes Z if true
    if (packet.compute_Z) {
      compute.write(true);
      wait();
    }

    HWC_SIG(Recv, 2);
    while (compute) wait();

    // Sends Z if true
    if (packet.send_Z) {
      send.write(true);
      wait();
    }

    HWC_SIG(Recv, 3);
    while (send) wait();

    wait();
  }
}

void ACCNAME::Compute() {
  Compute_si.write(0);
  wait();
  while (1) {
    Compute_si.write(1);
    while (!compute) wait();
    Compute_si.write(2);
    DWAIT();

    // Element-wise vector addition: Z[i] = X[i] + Y[i]
    for (int i = 0; i < acc_args.L; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = VEC_LENGTH max = VEC_LENGTH
#pragma HLS UNROLL factor = VEC_LENGTH
      Z_buffer[i] = X_buffer[i] + Y_buffer[i];
    }

    wait();
    compute.write(false);
    wait();
  }
}

void ACCNAME::Send() {
  testS.write(0);
  Send_si.write(0);
  wait();
  while (1) {
    Send_si.write(1);
    while (!send) wait();
    Send_si.write(2);
    DWAIT();

    for (int i = 0; i < acc_args.L; i++) {
      ADATA d;
      d.tlast = (i + 1 == acc_args.L);
      d.data = Z_buffer[i];
      dout1.write(d);
      testS.write(d.data);
      wait();
      DWAIT();
    }
    send.write(false);
    wait();
  }
}
