
#include "../secda_integrator/sysc_types.h"
#include "axi_api_v3.h"

// TODO Implement SystemC Signal Read/Write
// ================================================================================
// AXI4Lite API
// ================================================================================
acc_regmap::acc_regmap(size_t base_addr, size_t length) {}

void acc_regmap::writeAccReg(uint32_t offset, unsigned int val) {}

unsigned int acc_regmap::readAccReg(uint32_t offset) { return 0; }

// TODO: parse JSON file to load offset map for control and status registers
void acc_regmap::parseOffsetJSON() {}

// TODO: checks control and status register arrays to find the offsets for the
// register
uint32_t acc_regmap::findRegOffset(string reg_name) { return 0; }

void acc_regmap::writeToControlReg(string reg_name, unsigned int val) {}

unsigned int acc_regmap::readToControlReg(string reg_name) { return 0; }

// ================================================================================
// Memory Map API
// ================================================================================

// Make this into a struct based API (for SystemC)

// ================================================================================
// Stream DMA API
// ================================================================================

template <int B>
int stream_dma<B>::s_id = 0;
int sr_id = 0;

template <int B>
stream_dma<B>::stream_dma(unsigned int _dma_addr, unsigned int _input,
                          unsigned int _r_paddr, unsigned int _input_size,
                          unsigned int _output, unsigned int _w_paddr,
                          unsigned int _output_size)
    : id(sr_id++) {
  string name("SDMA" + to_string(id));
  dmad = new AXIS_ENGINE<B>(&name[0]);
  dmad->id = id;
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->r_paddr = _r_paddr;
  dmad->w_paddr = _w_paddr;
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

template <int B>
stream_dma<B>::stream_dma() : id(sr_id++) {
  string name("MSDMA" + to_string(id));
  dmad = new AXIS_ENGINE<B>(&name[0]);
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->id = id;
};

template <int B>
void stream_dma<B>::dma_init(unsigned int _dma_addr, unsigned int _input,
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

template <int B>
void stream_dma<B>::writeMappedReg(uint32_t offset, unsigned int val) {}

template <int B>
unsigned int stream_dma<B>::readMappedReg(uint32_t offset) {
  return 0;
}

template <int B>
void stream_dma<B>::dma_mm2s_sync() {
  sc_start();
}
template <int B>
void stream_dma<B>::dma_s2mm_sync() {
  sc_start();
}

template <int B>
void stream_dma<B>::dma_change_start(int offset) {
  dmad->input_offset = offset / 4;
}

template <int B>
void stream_dma<B>::dma_change_end(int offset) {
  dmad->output_offset = offset / 4;
}

template <int B>
void stream_dma<B>::initDMA(unsigned int src, unsigned int dst) {}

template <int B>
void stream_dma<B>::dma_free() {
  free(input);
  free(output);
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
  dmad->input_len = length * (B / 32);
  dmad->send = true;
  data_transfered += length * (B / 8);
}

template <int B>
void stream_dma<B>::dma_wait_send() {
  prf_start(0);
  dma_mm2s_sync();
  prf_end(0, send_wait);
}

template <int B>
int stream_dma<B>::dma_check_send() {
  return 0;
}

template <int B>
void stream_dma<B>::dma_start_recv(int length) {
  dmad->output_len = length * (B / 32);
  dmad->recv = true;
}

template <int B>
void stream_dma<B>::dma_wait_recv() {
  prf_start(0);
  dma_s2mm_sync();
  prf_end(0, recv_wait);
#ifdef ACC_PROFILE
  data_transfered_recv += dmad->output_len * (B / 8);
#endif
}

template <int B>
int stream_dma<B>::dma_check_recv() {
  return 0;
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
    dmas[i].dma_init(dma_addrs[i], dma_addrs_in[i], buffer_size * 2,
                     dma_addrs_out[i], buffer_size * 2);
}

template <int B>
void multi_dma<B>::multi_free_dmas() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_free();
  }
}

template <int B>
void multi_dma<B>::multi_dma_change_start(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_start(offset);
  }
}

template <int B>
void multi_dma<B>::multi_dma_change_start_4(int offset) {}

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
  bool loop = true;
  while (loop) {
    loop = false;
    for (int i = 0; i < dma_count; i++) {
      if (dmas[i].dmad->send) {
        dmas[i].dma_wait_send();
      }
      loop = loop || dmas[i].dmad->send;
    }
  }
}

template <int B>
int multi_dma<B>::multi_dma_check_send() {
  return 0;
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

template <int B>
void multi_dma<B>::multi_dma_wait_recv_4() {}

template <int B>
int multi_dma<B>::multi_dma_check_recv() {
  return 0;
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