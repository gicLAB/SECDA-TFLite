
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
int streams_dma<B>::s_id = 0;
int sr_id = 0;

template <int B>
streams_dma<B>::streams_dma(unsigned int _dma_addr, unsigned int _input,
                            unsigned int _input_size, unsigned int _output,
                            unsigned int _output_size)
    : id(sr_id++) {
  string name("SDMA" + to_string(id));
  dmad = new AXIS_ENGINE<B>(&name[0]);
  dmad->id = id;
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

template <int B>
streams_dma<B>::streams_dma() : id(sr_id++) {
  string name("MSDMA" + to_string(id));
  dmad = new AXIS_ENGINE<B>(&name[0]);
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->id = id;
};

template <int B>
void streams_dma<B>::dma_init(unsigned int _dma_addr, unsigned int _input,
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
}

template <int B>
void streams_dma<B>::writeMappedReg(uint32_t offset, unsigned int val) {}

template <int B>
unsigned int streams_dma<B>::readMappedReg(uint32_t offset) {
  return 0;
}

template <int B>
void streams_dma<B>::dma_mm2s_sync() {}

template <int B>
void streams_dma<B>::dma_s2mm_sync() {}

template <int B>
void streams_dma<B>::dma_change_start(int offset) {
  dmad->input_offset = offset;
}

template <int B>
void streams_dma<B>::dma_change_end(int offset) {
  dmad->output_offset = offset;
}

template <int B>
void streams_dma<B>::initDMA(unsigned int src, unsigned int dst) {}

template <int B>
void streams_dma<B>::dma_free() {
  free(input);
  free(output);
}

template <int B>
int *streams_dma<B>::dma_get_inbuffer() {
  return input;
}

template <int B>
int *streams_dma<B>::dma_get_outbuffer() {
  return output;
}

template <int B>
void streams_dma<B>::dma_start_send(int length) {
  dmad->input_len = length;
  dmad->send = true;
}

template <int B>
void streams_dma<B>::dma_wait_send() {
  sc_start();
}

template <int B>
int streams_dma<B>::dma_check_send() {
  return 0;
}

template <int B>
void streams_dma<B>::dma_start_recv(int length) {
  dmad->output_len = length;
  dmad->recv = true;
}

template <int B>
void streams_dma<B>::dma_wait_recv() {
  sc_start();
}

template <int B>
int streams_dma<B>::dma_check_recv() {
  return 0;
}

// =========================== Multi DMAs
multi_dma::multi_dma(int _dma_count, unsigned int *_dma_addrs,
                     unsigned int *_dma_addrs_in, unsigned int *_dma_addrs_out,
                     unsigned int _buffer_size) {
  dma_count = _dma_count;
  dmas = new streams_dma<32>[dma_count];
  dma_addrs = _dma_addrs;
  dma_addrs_in = _dma_addrs_in;
  dma_addrs_out = _dma_addrs_out;
  buffer_size = _buffer_size;

  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_init(dma_addrs[i], dma_addrs_in[i], buffer_size * 2,
                     dma_addrs_out[i], buffer_size * 2);
}

void multi_dma::multi_free_dmas() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_free();
  }
}

void multi_dma::multi_dma_change_start(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_start(offset);
  }
}

void multi_dma::multi_dma_change_start_4(int offset) {}

void multi_dma::multi_dma_change_end(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_end(offset);
  }
}

void multi_dma::multi_dma_start_send(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_send(length);
}

void multi_dma::multi_dma_wait_send() {
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

int multi_dma::multi_dma_check_send() { return 0; }

void multi_dma::multi_dma_start_recv(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_recv(length);
}

void multi_dma::multi_dma_start_recv() {
  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_start_recv(dmas[i].output_size);
}

void multi_dma::multi_dma_wait_recv() {
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

void multi_dma::multi_dma_wait_recv_4() {}

int multi_dma::multi_dma_check_recv() { return 0; }

template struct streams_dma<32>;
template struct streams_dma<64>;
template class AXIS_ENGINE<32>;
template class AXIS_ENGINE<64>;