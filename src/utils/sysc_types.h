#ifndef SYSC_TYPES_H
#define SYSC_TYPES_H

#include "hwc.sc.h"
#include "secda_hw_utils.sc.h"

#include <iomanip>
#include <iostream>
#include <systemc.h>

#ifndef DWAIT
#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif
#endif

#ifndef AXI_TYPE
#define AXI_TYPE sc_uint
#endif

template <int W, template <int> class T>
struct _BDATA {
  T<W> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _BDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
  void operator=(_BDATA<W, T> _data) {
    data = _data.data;
    tlast = _data.tlast;
  }

  void pack(sc_uint<W / 4> a1, sc_uint<W / 4> a2, sc_uint<W / 4> a3,
            sc_uint<W / 4> a4) {
    data.range(((W * 1 / 4) - 1), 0) = a1;
    data.range(((W * 2 / 4) - 1), (W * 1 / 4)) = a2;
    data.range(((W * 3 / 4) - 1), (W * 2 / 4)) = a3;
    data.range(((W * 4 / 4) - 1), (W * 3 / 4)) = a4;
  }

  void pack(sc_int<W / 4> a1, sc_int<W / 4> a2, sc_int<W / 4> a3,
            sc_int<W / 4> a4) {
    data.range(((W * 1 / 4) - 1), 0) = a1;
    data.range(((W * 2 / 4) - 1), (W * 1 / 4)) = a2;
    data.range(((W * 3 / 4) - 1), (W * 2 / 4)) = a3;
    data.range(((W * 4 / 4) - 1), (W * 3 / 4)) = a4;
  }
};


#endif