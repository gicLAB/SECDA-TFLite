
#ifndef HW_MODULES_SC_H
#define HW_MODULES_SC_H

void ACCNAME::print_profile() {
  ALOG("++++++++++++++++++++++++++++++++++++++++");
  ALOG("Read A data_len: " << read_A_len);
  ALOG("Read B data_len: " << read_B_len);
  ALOG("MACs count: " << compute_C_len);
  ALOG("Send C data_len: " << send_C_len);
  ALOG("++++++++++++++++++++++++++++++++++++++++");
  ALOG("Executed with :" << __FILE__);
  ALOG("- - - - - - - - - - - - - - - - - - - - ");
}

void ACCNAME::Counter() {
  wait();
  while (1) {
    per_batch_cycles->value++;
    if (compute.read() == 1)
      active_cycles->value++;
    wait();
  }
}

void ACCNAME::Recv() {

  wait();
  while (1) {
    opcode packet(din1.read().data);
    code_extension op_args(din1.read().data);
    acc_args = op_args;

    if (packet.read_A) {
      int read_length = op_args.M * op_args.K16;
      for (int i = 0; i < read_length; i++) {
        for (int j = 0; j < 16; j++) {
          A_buffer[i][j] = din1.read().data;
          read_A_len++;
          DWAIT();
        }
      }
    }

    if (packet.read_B) {
      int read_length = op_args.K * op_args.N16;
      for (int i = 0; i < read_length; i++) {
        for (int j = 0; j < 16; j++) {
          B_buffer[i][j] = din1.read().data;
          read_B_len++;
          DWAIT();
        }
      }
    }

    // Computes C if true
    if (packet.compute_C) {
      compute.write(true);
      wait();
    }

    while (compute)
      wait();

    // Sends then clears C if true
    if (packet.send_C) {
      send.write(true);
      wait();
    }

    while (send)
      wait();

    wait();
  }
}

void ACCNAME::LoadA(sc_int<32> A[PE_M][PE_K], su10 M, su10 K, su10 in_stride) {
  su12 base = M * in_stride + K;
  su12 offset = 0;
  for (su10 m = 0; m < PE_M; m++) {
    for (su10 k = 0; k < PE_K; k++) {
      // #pragma HLS unroll
      A[m][k] = A_buffer[base + offset][k];
    }
    offset += in_stride;
  }
}

void ACCNAME::LoadB(sc_int<32> B[PE_K][PE_N], su10 K, su10 N, su10 in_stride) {
  su12 base = K * in_stride + N;
  su12 offset = 0;
  for (su10 k = 0; k < PE_K; k++) {
    for (su10 n = 0; n < PE_N; n++) {
      // #pragma HLS unroll
      B[k][n] = B_buffer[base + offset][n];
    }
    offset += in_stride;
  }
}

void ACCNAME::Compute(sc_int<32> A[PE_M][PE_K], sc_int<32> B[PE_K][PE_N],
                      sc_int<32> C[PE_M][PE_N]) {
  for (int m = 0; m < PE_M; m++) {
    for (int n = 0; n < PE_N; n++) {
      // #pragma HLS pipeline
      // #pragma HLS unroll factor 4
      int acc = 0;
      for (int k = 0; k < PE_K; k++) {
        int x = A[m][k];
        int y = B[k][n];
        acc += mul_int32(x, y);
        compute_C_len++;
      }
      C[m][n] = acc;
    }
  }
}

void ACCNAME::Store(sc_int<32> C[PE_M][PE_N], su10 M, su10 N, su10 out_stride) {
  su12 base = M * out_stride + N;
  su12 offset = 0;
  for (su10 m = 0; m < PE_M; m++) {
    // #pragma HLS pipeline
    for (su10 n = 0; n < PE_N; n++) {
      // #pragma HLS unroll
      C_buffer[base + offset][n] += C[m][n];
    }
    offset += out_stride;
  }
}

void ACCNAME::Schedule_Compute() {
  sc_int<32> A[PE_M][PE_K];
  sc_int<32> B[PE_K][PE_N];
  sc_int<32> C[PE_M][PE_N];
  // #pragma HLS array_partition variable = A complete dim = 2
  // #pragma HLS array_partition variable = B complete dim = 2
  // #pragma HLS array_partition variable = C complete dim = 2

  wait();
  while (1) {
    while (!compute)
      wait();

    unsigned int ks = 0;
    for (su10 k = 0; k < acc_args.K; k += PE_K) {
      for (su10 m = 0; m < acc_args.M; m += PE_M) {
        LoadA(A, m, ks, acc_args.K16);
        for (su10 n = 0; n < acc_args.N16; n++) {
          LoadB(B, k, n, acc_args.N16);
          Compute(A, B, C);
          Store(C, m, n, acc_args.N16);
        }
      }
      ks++;
    }

    wait();
    compute.write(false);
    wait();
  }
}

void ACCNAME::Send() {
  wait();
  while (1) {
    while (!send)
      wait();

    unsigned int write_length = acc_args.M * acc_args.N16;
    for (su10 m = 0; m < write_length; m++) {
      for (su10 n = 0; n < 16; n++) {
        DATA d;
        d.tlast = false;
        d.data = C_buffer[m][n];
        if (n + 1 == 16 && m + 1 == write_length)
          d.tlast = true;
        dout1.write(d);
        send_C_len++;
        wait();
        C_buffer[m][n] = 0;
        DWAIT();
      }
    }
    send.write(false);
    wait();
  }
}

int ACCNAME::mul_int32(int x, int y) { return x * y; }

#endif
