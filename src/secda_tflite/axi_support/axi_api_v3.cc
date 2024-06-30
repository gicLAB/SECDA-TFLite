#ifndef SYSC

#include "axi_api_v3.h"
// ================================================================================
// AXI4Lite API
// ================================================================================
acc_regmap::acc_regmap(size_t base_addr, size_t length) {
  acc_addr = getAccBaseAddress<int>(base_addr, length);
}

void acc_regmap::writeAccReg(uint32_t offset, unsigned int val) {
  void *base_addr = (void *)acc_addr;
  *((volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset)) =
      val;
}

unsigned int acc_regmap::readAccReg(uint32_t offset) {
  void *base_addr = (void *)acc_addr;
  return *(
      (volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset));
}

// TODO: parse JSON file to load offset map for control and status registers
void acc_regmap::parseOffsetJSON() {}

// TODO: checks control and status register arrays to find the offsets for the
// register
uint32_t acc_regmap::findRegOffset(string reg_name) {
  uint32_t offset = 0;
  return offset;
}

void acc_regmap::writeToControlReg(string reg_name, unsigned int val) {
  writeAccReg(findRegOffset(reg_name), val);
}

unsigned int acc_regmap::readToControlReg(string reg_name) {
  return readAccReg(findRegOffset(reg_name));
}

// ================================================================================
// Memory Map API
// ================================================================================

// Make this into a struct based API

// ================================================================================
// Stream DMA API
// ================================================================================
template <int B>
int stream_dma<B>::s_id = 0;
int sr_id = 0;

template <int B>
stream_dma<B>::stream_dma(unsigned int _dma_addr, unsigned int _input,
                          unsigned int _input_size, unsigned int _output,
                          unsigned int _output_size)
    : id(sr_id++) {
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

template <int B>
stream_dma<B>::stream_dma() : id(sr_id++){};

template <int B>
void stream_dma<B>::dma_init(unsigned int _dma_addr, unsigned int _input,
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
}

template <int B>
void stream_dma<B>::writeMappedReg(uint32_t offset, unsigned int val) {
  void *base_addr = (void *)dma_addr;
  *((volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset)) =
      val;
}

template <int B>
unsigned int stream_dma<B>::readMappedReg(uint32_t offset) {
  void *base_addr = (void *)dma_addr;
  return *(
      (volatile unsigned int *)(reinterpret_cast<char *>(base_addr) + offset));
}

template <int B>
void stream_dma<B>::dma_mm2s_sync() {
  msync(dma_addr, PAGE_SIZE, MS_SYNC);
  unsigned int mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  while (!(mm2s_status & 1 << 12) || !(mm2s_status & 1 << 1)) {
    msync(dma_addr, PAGE_SIZE, MS_SYNC);
    mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  }
}

template <int B>
void stream_dma<B>::dma_s2mm_sync() {
  msync(dma_addr, PAGE_SIZE, MS_SYNC);
  unsigned int s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  while (!(s2mm_status & 1 << 12) || !(s2mm_status & 1 << 1)) {
    msync(dma_addr, PAGE_SIZE, MS_SYNC);
    s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  }
}

template <int B>
void stream_dma<B>::dma_change_start(int offset) {
  writeMappedReg(MM2S_START_ADDRESS, input_addr + offset);
}

template <int B>
void stream_dma<B>::dma_change_end(int offset) {
  writeMappedReg(S2MM_DESTINATION_ADDRESS, output_addr + offset);
}

template <int B>
void stream_dma<B>::initDMA(unsigned int src, unsigned int dst) {
  writeMappedReg(S2MM_CONTROL_REGISTER, 4);
  writeMappedReg(MM2S_CONTROL_REGISTER, 4);
  writeMappedReg(S2MM_CONTROL_REGISTER, 0);
  writeMappedReg(MM2S_CONTROL_REGISTER, 0);
  writeMappedReg(S2MM_DESTINATION_ADDRESS, dst);
  writeMappedReg(MM2S_START_ADDRESS, src);
  writeMappedReg(S2MM_CONTROL_REGISTER, 0xf001);
  writeMappedReg(MM2S_CONTROL_REGISTER, 0xf001);
}

template <int B>
void stream_dma<B>::dma_free() {
  munmap(input, input_size);
  munmap(output, output_size);
  munmap(dma_addr, getpagesize());
}

template <int B>
int *stream_dma<B>::dma_get_inbuffer() {
  return input;
}

template <int B>
int *stream_dma<B>::dma_get_outbuffer() {
  return output;
}

template <int B>
void stream_dma<B>::dma_start_send(int length) {
  prf_start(0);
#ifdef ACC_PROFILE
  data_transfered += length * (B / 8);
#endif
  msync(input, input_size, MS_SYNC);
  writeMappedReg(MM2S_LENGTH, length * (B / 8));
  prf_end(0, send_wait);
}

template <int B>
void stream_dma<B>::dma_wait_send() {
  prf_start(0);
  dma_mm2s_sync();
  prf_end(0, send_wait);
}

template <int B>
int stream_dma<B>::dma_check_send() {
  unsigned int mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  bool done = !((!(mm2s_status & 1 << 12)) || (!(mm2s_status & 1 << 1)));
  return done ? 0 : -1;
}

template <int B>
void stream_dma<B>::dma_start_recv(int length) {
  writeMappedReg(S2MM_LENGTH, length * (B / 8));
}

template <int B>
void stream_dma<B>::dma_wait_recv() {
  prf_start(0);
  dma_s2mm_sync();
  msync(output, output_size, MS_SYNC);
  prf_end(0, recv_wait);
#ifdef ACC_PROFILE
  data_transfered_recv += readMappedReg(S2MM_LENGTH);
#endif
}

template <int B>
int stream_dma<B>::dma_check_recv() {
  unsigned int s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  bool done = !((!(s2mm_status & 1 << 12)) || (!(s2mm_status & 1 << 1)));
  return done ? 0 : -1;
}

template <int B>
void stream_dma<B>::print_times() {
#ifdef ACC_PROFILE
  cout << "================================================" << endl;
  cout << "-----------"
       << "DMA: " << id << "-----------" << endl;
  cout << "Data Transfered: " << data_transfered << " bytes" << endl;
  cout << "Data Transfered Recv: " << data_transfered_recv << " bytes" << endl;
  prf_out(TSCALE, send_wait);
  prf_out(TSCALE, recv_wait);
  float sendtime = (float)prf_count(TSCALE, send_wait) / 1000000;
  float data_transfered_recv_MB = (float)data_transfered_recv / 1000000;
  cout << "Send speed: " << (data_transfered_recv_MB / sendtime) << " MB/s"
       << endl;
  float recvtime = (float)prf_count(TSCALE, recv_wait) / 1000000;
  float data_transfered_MB = (float)data_transfered / 1000000;
  cout << "Recv speed: " << (data_transfered_MB / recvtime) << " MB/s" << endl;
  cout << "================================================" << endl;
#endif
}

// =========================== Multi DMAs
template <int B>
multi_dma<B>::multi_dma(int _dma_count, unsigned int *_dma_addrs,
                        unsigned int *_dma_addrs_in,
                        unsigned int *_dma_addrs_out,
                        unsigned int _buffer_size) {
  dma_count = _dma_count;
  dmas = new stream_dma<B>[dma_count];
  dma_addrs = _dma_addrs;
  dma_addrs_in = _dma_addrs_in;
  dma_addrs_out = _dma_addrs_out;
  buffer_size = _buffer_size;

  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_init(dma_addrs[i], dma_addrs_in[i], buffer_size * 1,
                     dma_addrs_out[i], buffer_size * 1);
}

template <int B>
void multi_dma<B>::multi_free_dmas() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_free();
  }
  delete[] dmas;
}

template <int B>
void multi_dma<B>::multi_dma_change_start(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_start(offset);
  }
}

template <int B>
void multi_dma<B>::multi_dma_change_start_4(int offset) {
  dmas[0].dma_change_start(offset);
  dmas[1].dma_change_start(offset);
  dmas[2].dma_change_start(offset);
  dmas[3].dma_change_start(offset);
}

template <int B>
void multi_dma<B>::multi_dma_change_end(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_end(offset);
  }
}

template <int B>
void multi_dma<B>::multi_dma_start_send(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_send(length);
}

template <int B>
void multi_dma<B>::multi_dma_wait_send() {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_mm2s_sync();
}

template <int B>
int multi_dma<B>::multi_dma_check_send() {
  bool done = true;
  for (int i = 0; i < dma_count; i++)
    done = done && (dmas[i].dma_check_send() == 0);
  return done ? 0 : -1;
}

template <int B>
void multi_dma<B>::multi_dma_start_recv(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_recv(length);
}

template <int B>
void multi_dma<B>::multi_dma_start_recv() {
  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_start_recv(dmas[i].output_size);
}

template <int B>
void multi_dma<B>::multi_dma_wait_recv() {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_s2mm_sync();
}

template <int B>
void multi_dma<B>::multi_dma_wait_recv_4() {
  dmas[0].dma_s2mm_sync();
  dmas[1].dma_s2mm_sync();
  dmas[2].dma_s2mm_sync();
  dmas[3].dma_s2mm_sync();
}

template <int B>
int multi_dma<B>::multi_dma_check_recv() {
  bool done = true;
  for (int i = 0; i < dma_count; i++)
    done = done && (dmas[i].dma_check_recv() == 0);
  return done ? 0 : -1;
}

template <int B>
void multi_dma<B>::print_times() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].print_times();
  }
}

template class stream_dma<32>;
template class multi_dma<32>;

template class stream_dma<64>;
template class multi_dma<64>;

template class stream_dma<128>;
template class multi_dma<128>;

template class stream_dma<256>;
template class multi_dma<256>;

template class stream_dma<512>;
template class multi_dma<512>;

template class stream_dma<1024>;
template class multi_dma<1024>;

#endif