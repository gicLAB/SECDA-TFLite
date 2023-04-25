#ifndef ACCNAME_H
#define ACCNAME_H

#include "sysc_integrator/sysc_types.h"
#include "sysc_profiler/profiler.h"
#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

// #define VERBOSE_ACC
#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif

#define ACCNAME MM_16x16v4
#define PE_M 16
#define PE_N 16
#define PE_K 16

// OP-Code Stuct
// 0000 : 0 = NOP;
// 0001 : 1 = read_A;
// 0010 : 2 = read_B;
// 0011 : 3 = read_A -> read_B;
// 0100 : 4 = compute_C;
// 0101 : 5 = read_A -> compute_C;
// 0110 : 6 = read_B -> compute_C;
// 0111 : 7 = read_A -> read_B -> compute_C;

// 1000 : 8 = send_C;
// 1001 : 9 = read_A -> send_C;
// 1010 : 10 = read_B -> send_C;
// 1011 : 11 = read_A -> read_B -> send_C;
// 1100 : 12 = compute_C -> send_C;
// 1101 : 13 = read_A -> compute_C -> send_C;
// 1110 : 14 = read_B -> compute_C -> send_C;
// 1111 : 15 = read_A -> read_B -> compute_C -> send_C;

#define su10 sc_uint<10>
#define su12 sc_uint<12>
// MAX M, N, K = 256
struct opcode {
  unsigned int packet;
  bool read_A;
  bool read_B;
  bool compute_C;
  bool send_C;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    read_A = _packet.range(0, 0);
    read_B = _packet.range(1, 1);
    compute_C = _packet.range(2, 2);
    send_C = _packet.range(3, 3);
  }
};

struct code_extension {
  su10 N;
  su10 M;
  su10 K;
  su10 K16;
  su10 N16;

  code_extension(sc_uint<32> _packetA) {
    M = _packetA.range(9, 0);
    N = _packetA.range(19, 10);
    K = _packetA.range(29, 20);
    N16 = _packetA.range(19, 10) / PE_N;
    K16 = _packetA.range(29, 20) / PE_K;
    ALOG("packetA: " << _packetA);
    ALOG("Time: " << sc_time_stamp());
    ALOG("N: " << N << ", M: " << M << ", K: " << K);
  }
};

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> A_buffer[256][16];
  sc_int<32> B_buffer[256][16];
  sc_int<32> C_buffer[256][16];
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send;
#else
  sc_signal<bool> compute;
  sc_signal<bool> send;
#endif

  code_extension acc_args = code_extension(0);

  // Debug variables
  int process_blocks;
  int read_A_len;
  int read_B_len;
  int compute_C_len;
  int send_C_len;
  bool verbose;

  // Profiling variable
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles};

  void Counter();

  void Recv();

  void Compute(sc_int<32>[PE_M][PE_K], sc_int<32>[PE_K][PE_N],
               sc_int<32>[PE_M][PE_N]);

  void LoadA(sc_int<32>[PE_M][PE_K], su10, su10, su10);

  void LoadB(sc_int<32>[PE_K][PE_N], su10, su10, su10);

  void Store(sc_int<32>[PE_M][PE_N], su10, su10, su10);

  void Schedule_Compute();

  void Send();

  void print_profile();

  int mul_int32(int, int);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Schedule_Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Counter, clock);
    reset_signal_is(reset, true);

    process_blocks = 0;
    read_A_len = 0;
    read_B_len = 0;
    compute_C_len = 0;
    send_C_len = 0;
    verbose = false;
  }
};

#endif
