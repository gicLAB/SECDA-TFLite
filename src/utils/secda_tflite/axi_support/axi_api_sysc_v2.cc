
#include "axi_api_v2.h"

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
int stream_dma::s_id = 0;

stream_dma::stream_dma(unsigned int _dma_addr, unsigned int _input,
                       unsigned int _input_size, unsigned int _output,
                       unsigned int _output_size)
    : id(s_id++) {
  string name("DMA" + to_string(id));
  dmad = new AXIS_ENGINE(&name[0]);
  dmad->id = id;
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

stream_dma::stream_dma(unsigned int _dma_addr, unsigned int _input,
                       unsigned int _r_paddr, unsigned int _input_size,
                       unsigned int _output, unsigned int _w_paddr,
                       unsigned int _output_size)
    : id(s_id++) {
  string name("DMA" + to_string(id));
  dmad = new AXIS_ENGINE(&name[0]);
  dmad->id = id;
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->r_paddr = _r_paddr;
  dmad->w_paddr = _w_paddr;
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

stream_dma::stream_dma() : id(s_id++) {
  string name("DMA" + to_string(id));
  dmad = new AXIS_ENGINE(&name[0]);
  dmad->input_len = 0;
  dmad->input_offset = 0;
  dmad->output_len = 0;
  dmad->output_offset = 0;
  dmad->id = id;
};

void stream_dma::dma_init(unsigned int _dma_addr, unsigned int _input,
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

void stream_dma::writeMappedReg(uint32_t offset, unsigned int val) {}

unsigned int stream_dma::readMappedReg(uint32_t offset) { return 0; }

void stream_dma::dma_mm2s_sync() { sc_start(); }

void stream_dma::dma_s2mm_sync() { sc_start(); }

void stream_dma::dma_change_start(int offset) {
  dmad->input_offset = offset / 4;
}

void stream_dma::dma_change_end(int offset) {
  dmad->output_offset = offset / 4;
}

void stream_dma::initDMA(unsigned int src, unsigned int dst) {}

void stream_dma::dma_free() {
  free(input);
  free(output);
}

int *stream_dma::dma_get_inbuffer() { return input; }

int *stream_dma::dma_get_outbuffer() { return output; }

void stream_dma::dma_start_send(int length) {
  dmad->input_len = length;
  dmad->send = true;
  data_transfered += length * 4;
}

void stream_dma::dma_wait_send() {
  prf_start(0);
  dma_mm2s_sync();
  prf_end(0, send_wait);
}

int stream_dma::dma_check_send() { return 0; }

void stream_dma::dma_start_recv(int length) {
  dmad->output_len = length;
  dmad->recv = true;
}

void stream_dma::dma_wait_recv() {
#ifdef ACC_PROFILE
  data_transfered_recv += dmad->output_len * 4;
#endif
  prf_start(0);
  dma_s2mm_sync();
  prf_end(0, recv_wait);
}

int stream_dma::dma_check_recv() { return 0; }

void stream_dma::print_times() {
#ifdef ACC_PROFILE
  cout << "-----------"
       << "DMA: " << id << "-----------" << endl;
  cout << "Data Transfered: " << data_transfered << " bytes" << endl;
  cout << "Data Transfered Recv: " << data_transfered_recv << " bytes" << endl;
  prf_out(TSCALE, send_wait);
  prf_out(TSCALE, recv_wait);
  cout << "================================================" << endl;
#endif
}

// =========================== Multi DMAs
multi_dma::multi_dma(int _dma_count, unsigned int *_dma_addrs,
                     unsigned int *_dma_addrs_in, unsigned int *_dma_addrs_out,
                     unsigned int _buffer_size) {
  dma_count = _dma_count;
  dmas = new stream_dma[dma_count];
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

void multi_dma::multi_dma_change_start_4(int offset) {
  dmas[0].dma_change_start(offset);
  dmas[1].dma_change_start(offset);
  dmas[2].dma_change_start(offset);
  dmas[3].dma_change_start(offset);
}

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

void multi_dma::multi_dma_wait_recv_4() { multi_dma_wait_recv(); }

int multi_dma::multi_dma_check_recv() { return 0; }

void multi_dma::print_times() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].print_times();
  }
}

// ================================================================================
// Memory Access API
// ================================================================================

// void massign(int *dst, int *src, int d_dex, int s_dex, int value) {

//   // unsigned int dst_addr = (&dst);
//   // unsigned int src_addr = (&src);
//   cout << "dst_addr: " << (void*)&dst[d_dex] << endl;
//   unsigned int dst_addr = 0;
//   unsigned int src_addr = 0;

//   // unsigned int dst_addr2 = (unsigned int)dst;

//   // create memory trace for ramulator and write to file
//   ofstream file;
//   std::string filename = ".data/secda_pim/data/write.trace";
//   file.open(filename, std::ios_base::app);
//   file << "0x" << std::setfill('0') << std::setw(8) << std::hex << (dst_addr)
//        << " W" << endl;
//   file.close();

//   // create memory trace for read, write trace to file
//   std::string filename2 = ".data/secda_pim/data/read.trace";
//   file.open(filename2, std::ios_base::app);
//   file << "0x" << std::setfill('0') << std::setw(8) << std::hex << (src_addr)
//        << " R" << endl;
//   file.close();
//   dst[d_dex] = value;
// }

void mcontroller::assign(int *dst, int *src, int value) {

  // unsigned int dst_addr = (unsigned int)dst;
  // unsigned int src_addr = (unsigned int)src;

  // // create memory trace for ramulator and write to file
  // ofstream file;
  // std::string filename = ".data/secda_pim/data/write.trace";
  // file.open(filename, std::ios_base::app);
  // file << "0x" << std::setfill('0') << std::setw(8) << std::hex << (dst_addr)
  //      << " W" << endl;
  // file.close();

  // // create memory trace for read, write trace to file
  // std::string filename2 = ".data/secda_pim/data/read.trace";
  // file.open(filename2, std::ios_base::app);
  // file << "0x" << std::setfill('0') << std::setw(8) << std::hex << (src_addr)
  //      << " R" << endl;
  // file.close();
  dst[0] = value;
}