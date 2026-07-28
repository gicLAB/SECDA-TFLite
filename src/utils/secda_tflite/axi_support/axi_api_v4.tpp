#ifndef SYSC

// ================================================================================
// AXI4Lite API
// ================================================================================

template <typename T>
acc_regmap<T>::acc_regmap(size_t base_addr, size_t length) {
  acc_addr = getAccBaseAddress<int>(base_addr, length);
}

template <typename T>
void acc_regmap<T>::writeAccReg(uint32_t offset, unsigned int val) {
  void *base_addr = (void *)acc_addr;
  *((volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset)) =
      val;
}

template <typename T>
unsigned int acc_regmap<T>::readAccReg(uint32_t offset) {
  void *base_addr = (void *)acc_addr;
  return *(
      (volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset));
}

// TODO: parse JSON file to load offset map for control and status registers
template <typename T>
void acc_regmap<T>::parseOffsetJSON() {}

// TODO: checks control and status register arrays to find the offsets for the
// register
template <typename T>
uint32_t acc_regmap<T>::findRegOffset(string reg_name) {
  uint32_t offset = 0;
  return offset;
}

template <typename T>
void acc_regmap<T>::writeToControlReg(string reg_name, unsigned int val) {
  writeAccReg(findRegOffset(reg_name), val);
}

template <typename T>
unsigned int acc_regmap<T>::readToControlReg(string reg_name) {
  return readAccReg(findRegOffset(reg_name));
}

// ================================================================================
// Memory Map API
// ================================================================================

// Make this into a struct based API

// ================================================================================
// Stream DMA API
// ================================================================================
template <int B, int T>
int stream_dma<B, T>::s_id = 0;
// int sr_id = 0;

template <int B, int T>
stream_dma<B, T>::stream_dma(unsigned int _dma_addr, unsigned int _input,
                             unsigned int _input_size, unsigned int _output,
                             unsigned int _output_size)
    : id(s_id++) {
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

template <int B, int T>
stream_dma<B, T>::stream_dma() : id(s_id++){};

template <int B, int T>
void stream_dma<B, T>::dma_init(unsigned int _dma_addr, unsigned int _input,
                                unsigned int _input_size, unsigned int _output,
                                unsigned int _output_size) {
  dma_addr = mm_alloc_rw<unsigned int>(_dma_addr, PAGE_SIZE);
  input = mm_alloc_rw<int>(_input, _input_size);
  output = mm_alloc_r<int>(_output, _output_size);
  input_size = _input_size;
  output_size = _output_size;
  input_addr = _input;
  output_addr = _output;
  initDMA(_input, _output);
  mlock(input, _input_size);
  mlock(output, _output_size);
}

template <int B, int T>
void stream_dma<B, T>::writeMappedReg(uint32_t offset, unsigned int val) {
  void *base_addr = (void *)dma_addr;
  *((volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset)) =
      val;
}

template <int B, int T>
unsigned int stream_dma<B, T>::readMappedReg(uint32_t offset) {
  void *base_addr = (void *)dma_addr;
  return *(
      (volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset));
}

template <int B, int T>
void stream_dma<B, T>::dma_mm2s_sync() {
  msync(dma_addr, PAGE_SIZE, MS_SYNC);
  unsigned int mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  while (!(mm2s_status & 1 << 12) || !(mm2s_status & 1 << 1)) {
    msync(dma_addr, PAGE_SIZE, MS_SYNC);
    mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  }
}

template <int B, int T>
void stream_dma<B, T>::dma_s2mm_sync() {
  msync(dma_addr, PAGE_SIZE, MS_SYNC);
  unsigned int s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  while (!(s2mm_status & 1 << 12) || !(s2mm_status & 1 << 1)) {
    msync(dma_addr, PAGE_SIZE, MS_SYNC);
    s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  }
}

template <int B, int T>
void stream_dma<B, T>::dma_change_start(int offset) {
  writeMappedReg(MM2S_START_ADDRESS, input_addr + offset);
}

template <int B, int T>
void stream_dma<B, T>::dma_change_start(unsigned int addr, int offset) {
  writeMappedReg(MM2S_START_ADDRESS, addr + offset);
}

template <int B, int T>
void stream_dma<B, T>::dma_change_end(int offset) {
  writeMappedReg(S2MM_DESTINATION_ADDRESS, output_addr + offset);
}

template <int B, int T>
void stream_dma<B, T>::initDMA(unsigned int src, unsigned int dst) {
  writeMappedReg(S2MM_CONTROL_REGISTER, 4);
  writeMappedReg(MM2S_CONTROL_REGISTER, 4);
  writeMappedReg(S2MM_CONTROL_REGISTER, 0);
  writeMappedReg(MM2S_CONTROL_REGISTER, 0);
  writeMappedReg(S2MM_DESTINATION_ADDRESS, dst);
  writeMappedReg(MM2S_START_ADDRESS, src);
  writeMappedReg(S2MM_CONTROL_REGISTER, 0xf001);
  writeMappedReg(MM2S_CONTROL_REGISTER, 0xf001);
}

template <int B, int T>
void stream_dma<B, T>::dma_free() {
  munmap(input, input_size);
  munmap(output, output_size);
  munmap(dma_addr, getpagesize());
  munlockall();
}

template <int B, int T>
int *stream_dma<B, T>::dma_get_inbuffer() {
  return input;
}

template <int B, int T>
int *stream_dma<B, T>::dma_get_outbuffer() {
  return output;
}

// template <int B, int T>
// void stream_dma<B, T>::dma_start_send(int length) {
//   prf_start(0);
// #ifdef ACC_PROFILE
//   data_transfered += length * (B / 8);
//   data_send_count++;
// #endif
//   msync(input, input_size, MS_SYNC);
//   writeMappedReg(MM2S_LENGTH, length * (B / 8));
//   prf_end(0, send_wait);
// }

// template <int B, int T>
// void stream_dma<B, T>::dma_wait_send() {
//   prf_start(0);
//   dma_mm2s_sync();
//   prf_end(0, send_wait);
// }

template <int B, int T>
void stream_dma<B, T>::dma_start_send(int length) {
#ifndef DISABLE_DMA
  prf_start_x(send_start);
#ifdef ACC_PROFILE
  data_transfered += length * (B / 8);
  data_send_count++;
#endif
  msync(input, input_size, MS_SYNC);
  writeMappedReg(MM2S_LENGTH, length * (B / 8));
#endif
}

template <int B, int T>
void stream_dma<B, T>::dma_wait_send() {
#ifndef DISABLE_DMA
  dma_mm2s_sync();
  prf_end_x(0, send_start, send_wait);
#endif
}

template <int B, int T>
int stream_dma<B, T>::dma_check_send() {
  unsigned int mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  bool done = !((!(mm2s_status & 1 << 12)) || (!(mm2s_status & 1 << 1)));
  return done ? 0 : -1;
}

template <int B, int T>
void stream_dma<B, T>::dma_start_recv(int length) {
#ifndef DISABLE_DMA
  writeMappedReg(S2MM_LENGTH, length * (B / 8));
#endif
}

template <int B, int T>
void stream_dma<B, T>::dma_wait_recv() {
#ifndef DISABLE_DMA
  // prf_start(0);
  dma_s2mm_sync();
  msync(output, output_size, MS_SYNC);
  // prf_end(0, recv_wait);
  prf_end_x(0, send_start, recv_wait);
#ifdef ACC_PROFILE
  data_transfered_recv += readMappedReg(S2MM_LENGTH);
  data_recv_count++;
#endif
#endif
}

template <int B, int T>
int stream_dma<B, T>::dma_check_recv() {
  unsigned int s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  bool done = !((!(s2mm_status & 1 << 12)) || (!(s2mm_status & 1 << 1)));
  return done ? 0 : -1;
}

template <int B, int T>
void stream_dma<B, T>::print_times() {
#ifdef ACC_PROFILE
  cerr << "================================================" << endl;
  cerr << "-----------"
       << "DMA: " << id << "-----------" << endl;
  cerr << "Data Transfered: " << data_transfered << " bytes" << endl;
  cerr << "Data Transfered Recv: " << data_transfered_recv << " bytes" << endl;
  prf_out(TSCALE, send_wait);
  prf_out(TSCALE, recv_wait);
  float sendtime = (float)prf_count(TSCALE, send_wait) / 1000000;
  float recvtime = (float)prf_count(TSCALE, recv_wait) / 1000000;
  float data_transfered_MB = (float)data_transfered / 1000000;
  float data_recv_MB = (float)data_transfered_recv / 1000000;

  if (duration_cast<TSCALE>(send_wait).count() == 0) send_wait = nanoseconds(1);
  if (duration_cast<TSCALE>(recv_wait).count() == 0) recv_wait = nanoseconds(1);
  if (data_send_count == 0) data_send_count = 1;
  if (data_recv_count == 0) data_recv_count = 1;

  cerr << "Send speed: " << (data_transfered_MB / sendtime) << " MB/s" << endl;
  cerr << "Recv speed: " << (data_recv_MB / recvtime) << " MB/s" << endl;
  cerr << "Data Send Count: " << data_send_count << endl;
  cerr << "Data Recv Count: " << data_recv_count << endl;
  int data_per_send = data_transfered / data_send_count;
  int data_per_recv = data_transfered_recv / data_recv_count;
  cerr << "Data per Send: " << data_per_send << " bytes" << endl;
  cerr << "Data per Recv: " << data_per_recv << " bytes" << endl;
  cerr << "================================================" << endl;
  std::ofstream file("dma" + std::to_string(id) + ".csv", std::ios::out);
  // csv file header
  file << "Data Transfered,Data Transfered Recv,Send Time,Recv Time,Send "
          "Speed,Recv Speed,Data Send Count,Data Recv Count,Data per Send,Data "
          "per Recv"
       << std::endl;
  file << data_transfered << "," << data_transfered_recv << "," << sendtime
       << "," << recvtime << "," << (data_transfered_MB / sendtime) << ","
       << (data_recv_MB / recvtime) << "," << data_send_count << ","
       << data_recv_count << "," << data_per_send << "," << data_per_recv
       << std::endl;

#endif
}

// =========================== Multi DMAs
template <int B, int T>
multi_dma<B, T>::multi_dma(int _dma_count, unsigned int *_dma_addrs,
                           unsigned int *_dma_addrs_in,
                           unsigned int *_dma_addrs_out,
                           unsigned int _buffer_size) {
  dma_count = _dma_count;
  dmas = new stream_dma<B, T>[dma_count];
  dma_addrs = _dma_addrs;
  dma_addrs_in = _dma_addrs_in;
  dma_addrs_out = _dma_addrs_out;
  buffer_size = _buffer_size;

  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_init(dma_addrs[i], dma_addrs_in[i], buffer_size * 1,
                     dma_addrs_out[i], buffer_size * 1);
}

template <int B, int T>
multi_dma<B, T>::~multi_dma() {
  print_times();
}

template <int B, int T>
void multi_dma<B, T>::multi_free_dmas() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_free();
  }
  delete[] dmas;
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_change_start(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_start(offset);
  }
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_change_start_4(int offset) {
  dmas[0].dma_change_start(offset);
  dmas[1].dma_change_start(offset);
  dmas[2].dma_change_start(offset);
  dmas[3].dma_change_start(offset);
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_change_end(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_end(offset);
  }
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_start_send(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_send(length);
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_wait_send() {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_wait_send();
}

template <int B, int T>
int multi_dma<B, T>::multi_dma_check_send() {
  bool done = true;
  for (int i = 0; i < dma_count; i++)
    done = done && (dmas[i].dma_check_send() == 0);
  return done ? 0 : -1;
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_start_recv(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_recv(length);
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_start_recv() {
  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_start_recv(dmas[i].output_size);
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_wait_recv() {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_wait_recv();
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_wait_recv_4() {
  dmas[0].dma_wait_recv();
  dmas[1].dma_wait_recv();
  dmas[2].dma_wait_recv();
  dmas[3].dma_wait_recv();
}

template <int B, int T>
int multi_dma<B, T>::multi_dma_check_recv() {
  bool done = true;
  for (int i = 0; i < dma_count; i++)
    done = done && (dmas[i].dma_check_recv() == 0);
  return done ? 0 : -1;
}

template <int B, int T>
void multi_dma<B, T>::print_times() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].print_times();
  }
}

#endif