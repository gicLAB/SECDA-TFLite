#include <systemc.h>

void ACCNAME::ReadInp() {
  inpReadReadyS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (inpReadReadyS.read() == 0) wait();

    wait();
    int pnr_ri = pN_rem;
    
    // no_inpR += 1;

    DWAIT(1);
    readTimeInpS.write(1);
    for (int i_row = 0; i_row < pnr_ri; i_row++) {
      DWAIT(1);
      for (int k_row = 0; k_row < pK; k_row += 4) {
        ADATA d_row = din2->read();
        rows[i_row][k_row + 0] = d_row.data.range(7, 0);
        rows[i_row][k_row + 1] = d_row.data.range(15, 8);
        rows[i_row][k_row + 2] = d_row.data.range(23, 16);
        rows[i_row][k_row + 3] = d_row.data.range(31, 24);
        DWAIT(6);
      }
    }
    readTimeInpS.write(0);
    inpReadReadyS.write(0);
    wait();
  }
}