#include <systemc.h>

void ACCNAME::ReadCols() {
  colReadReadyS.write(0);
  wait();
  while (1) {
    while (colReadReadyS.read() == 0) wait();

    DWAIT(2);

    // #pragma HLS pipeline II = 1
    for (i = 0; i < pM_rem; i++) { // Loop 1.1.1
      DWAIT(1);
      for (k = 0; k < pK; k += 4) { // Loop 1.1.1.1
        DATA d = din3->read();
        // colReads += 1;
        cols[i][k + 0] = d.data.range(7, 0);
        cols[i][k + 1] = d.data.range(15, 8);
        cols[i][k + 2] = d.data.range(23, 16);
        cols[i][k + 3] = d.data.range(31, 24);
        DWAIT(3);
      }
    }
    colReadReadyS.write(0);
    wait();
  }
}