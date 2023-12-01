#ifndef RAMULATOR_CONNECT_H
#define RAMULATOR_CONNECT_H

// struct ramulator_connect {

//   void init();
// };

// struct memory_blocks {

//   int size;
//   int offset;
//   int starting_addr;
//   int ending_addr;

//   int *data;
// };

#ifdef SYSC
#include "../sysc_integrator/phy.sc.h"
#endif

#include <string>
using namespace std;

struct rconnect {

#ifdef SYSC
  PHY *phy;
#endif

  rconnect();
};

#endif // RAMULATOR_CONNECT_H

// instead of copying data from delegate define type T pointers we should use
// memory blocks to keep track of loading and writing between unmapped and
// mapped memory

// While keeping track we can use ramulator to simulate the memory access
// with metrics given by ramulator we can produce total timing report

// This can be a very light weight trace generator
// Design an extensible phy/ mem gen in SystemC similar to AXI-S engine