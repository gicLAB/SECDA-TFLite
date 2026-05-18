#ifdef SYSC

// TODO Implement SystemC Signal Read/Write
// ================================================================================
// AXI4Lite API
// ================================================================================

template <typename T>
acc_regmap<T>::acc_regmap(size_t base_addr, size_t length) {}

template <typename T>
void acc_regmap<T>::writeAccReg(uint32_t offset, unsigned int val) {}

template <typename T>
unsigned int acc_regmap<T>::readAccReg(uint32_t offset) {
  return 0;
}

// TODO: parse JSON file to load offset map for control and status registers
template <typename T>
void acc_regmap<T>::parseOffsetJSON() {}

// TODO: checks control and status register arrays to find the offsets for the
// register
template <typename T>
uint32_t acc_regmap<T>::findRegOffset(string reg_name) {
  return 0;
}

template <typename T>
void acc_regmap<T>::writeToControlReg(string reg_name, unsigned int val) {}

template <typename T>
unsigned int acc_regmap<T>::readToControlReg(string reg_name) {
  return 0;
}

// ================================================================================
// Memory Map API
// ================================================================================

// Make this into a struct based API (for SystemC)

// ================================================================================
// Stream DMA API
// ================================================================================

template <int B, int T>
int stream_dma<B, T>::s_id = 0;
// int sr_id = 0;

template <int B, int T>
stream_dma<B, T>::stream_dma(unsigned int _dma_addr, unsigned int _input,
                             unsigned int _r_paddr, unsigned int _input_size,
                             unsigned int _output, unsigned int _w_paddr,
                             unsigned int _output_size)
    : id(s_id++) {
  string name("SDMA" + to_string(id));
  dmad = new AXIS_ENGINE<B, AXI_TYPE>(&name[0]);

  dmad->id = id;
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->r_paddr = _r_paddr;
  dmad->w_paddr = _w_paddr;
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

template <int B, int T>
stream_dma<B, T>::stream_dma() : id(s_id++) {
  string name("MSDMA" + to_string(id));
  dmad = new AXIS_ENGINE<B, AXI_TYPE>(&name[0]);
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->id = id;
};

template <int B, int T>
void stream_dma<B, T>::dma_init(unsigned int _dma_addr, unsigned int _input,
                                unsigned int _input_size, unsigned int _output,
                                unsigned int _output_size) {
  input = (int *)malloc(_input_size * sizeof(int));
  output = (int *)malloc(_output_size * sizeof(int));

  // Initialize with zeros
  for (int64_t i = 0; i < _input_size; i++) {
    *(input + i) = 0;
  }

  for (int64_t i = 0; i < _output_size; i++) {
    *(output + i) = 0;
  }
  input_size = _input_size;
  output_size = _output_size;
  dmad->DMA_input_buffer = (int *)input;
  dmad->DMA_output_buffer = (int *)output;
  dmad->r_paddr = _input;
  dmad->w_paddr = _output;
}

template <int B, int T>
void stream_dma<B, T>::writeMappedReg(uint32_t offset, unsigned int val) {}

template <int B, int T>
unsigned int stream_dma<B, T>::readMappedReg(uint32_t offset) {
  return 0;
}

template <int B, int T>
void stream_dma<B, T>::dma_mm2s_sync() {
#ifndef DISABLE_SIM
  sc_start();
#endif
}
template <int B, int T>
void stream_dma<B, T>::dma_s2mm_sync() {
#ifndef DISABLE_SIM
  sc_start();
#endif
}

template <int B, int T>
void stream_dma<B, T>::dma_change_start(int offset) {
  dmad->input_offset = offset / 4;
}

template <int B, int T>
void stream_dma<B, T>::dma_change_start(unsigned int addr, int offset) {
  dmad->input_offset = offset / 4;
}

template <int B, int T>
void stream_dma<B, T>::dma_change_end(int offset) {
  dmad->output_offset = offset / 4;
}

template <int B, int T>
void stream_dma<B, T>::initDMA(unsigned int src, unsigned int dst) {}

template <int B, int T>
void stream_dma<B, T>::dma_free() {
  free(input);
  free(output);
}

template <int B, int T>
int *stream_dma<B, T>::dma_get_inbuffer() {
  return input;
}

template <int B, int T>
int *stream_dma<B, T>::dma_get_outbuffer() {
  return output;
}

template <int B, int T>
void stream_dma<B, T>::dma_start_send(int length) {
  dmad->input_len = length * (B / 32);
  dmad->send = true;
#ifdef ACC_PROFILE
  data_transfered += length * (B / 8);
  data_send_count++;
#endif
}

template <int B, int T>
void stream_dma<B, T>::dma_wait_send() {
  prf_start(0);
  if (dmad->send) dma_mm2s_sync();
  prf_end(0, send_wait);
}

template <int B, int T>
int stream_dma<B, T>::dma_check_send() {
  return 0;
}

template <int B, int T>
void stream_dma<B, T>::dma_start_recv(int length) {
  dmad->output_len = length * (B / 32);
  dmad->recv = true;
}

template <int B, int T>
void stream_dma<B, T>::dma_wait_recv() {
  prf_start(0);
  if (dmad->recv) dma_s2mm_sync();
  prf_end(0, recv_wait);
#ifdef ACC_PROFILE
  data_transfered_recv += dmad->output_len * (B / 8);
  data_recv_count++;
#endif
}

template <int B, int T>
int stream_dma<B, T>::dma_check_recv() {
  return 0;
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
    dmas[i].dma_init(dma_addrs[i], dma_addrs_in[i], buffer_size * 2,
                     dma_addrs_out[i], buffer_size * 2);
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
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_change_start(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_start(offset);
  }
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_change_start_4(int offset) {}

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
  bool loop = true;
  while (loop) {
    loop = false;
    for (int i = 0; i < dma_count; i++) {
      if (dmas[i].dmad->send) dmas[i].dma_wait_send();
      loop = loop || dmas[i].dmad->send;
    }
  }
}

template <int B, int T>
int multi_dma<B, T>::multi_dma_check_send() {
  return 0;
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
  bool loop = true;
  while (loop) {
    loop = false;
    for (int i = 0; i < dma_count; i++) {
      bool s = dmas[i].dmad->recv;
      if (dmas[i].dmad->recv) dmas[i].dma_wait_recv();
      loop = loop || dmas[i].dmad->recv;
      bool e = dmas[i].dmad->recv;
      int k = 0;
    }
  }
}

template <int B, int T>
void multi_dma<B, T>::multi_dma_wait_recv_4() {}

template <int B, int T>
int multi_dma<B, T>::multi_dma_check_recv() {
  return 0;
}

template <int B, int T>
void multi_dma<B, T>::print_times() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].print_times();
  }
}

#endif // SYSC