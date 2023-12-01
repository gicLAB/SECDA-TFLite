#include "ramulator_connect.h"

rconnect::rconnect() {
  string name("PHY");
  phy = new PHY(&name[0]);
  phy->read_count = 0;
  phy->write_count = 0;
  // dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}