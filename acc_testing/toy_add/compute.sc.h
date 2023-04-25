#ifndef COMPUTE_SC_H
#define COMPUTE_SC_H

void ACCNAME::Counter() {
  wait();
  while (1) {
    per_batch_cycles->value++;
    if (computeS.read() == 1)
      active_cycles->value++;
    wait();
  }
}

void ACCNAME::Compute() {
  ACC_DTYPE<32> i1;
  ACC_DTYPE<32> i2;
  ACC_DTYPE<32> sum;
  int length;

  DATA d;
  computeS.write(0);
  wait();
  while (1) {
    computeS.write(0);
    DWAIT();

    length = din1.read().data;
    computeS.write(1);
    DWAIT();

    for (int i = 0; i < length; i++) {
      i1 = din1.read().data;
      i2 = din1.read().data;
      sum = i1 + i2;
      d.data = sum;
      if (i + 1 == length)
        d.tlast = true;
      else
        d.tlast = false;
      dout1.write(d);
    }
    DWAIT();
  }
}

#endif

