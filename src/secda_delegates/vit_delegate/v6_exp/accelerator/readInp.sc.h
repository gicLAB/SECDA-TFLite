// readinp_v6.cpp
#include <systemc.h>

void ACCNAME::ReadInp() {
  inpReadReadyS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (inpReadReadyS.read() == 0) wait();

    wait(); // a

    // cout << "READINP START" << endl;

    int inp_pnr = pN_rem;
    int inp_pkr = pK_rem;

    wait(); // a

    DWAIT(1);
    readTimeInpS.write(1);
    for (int i_row = 0; i_row < inp_pnr; i_row++) {
      DWAIT(1);
      // V6: Loop bound changed from pK to pK_rem to read only the k-slice
      for (int k_row = 0; k_row < inp_pkr; k_row += 4) {
        DATA d_row = din2->read();
        // The destination 'rows' buffer is indexed from 0 for each slice
        rows[i_row][k_row + 0] = d_row.data.range(7, 0);
        rows[i_row][k_row + 1] = d_row.data.range(15, 8);
        rows[i_row][k_row + 2] = d_row.data.range(23, 16);
        rows[i_row][k_row + 3] = d_row.data.range(31, 24);
        DWAIT(6);
      }
    }

    // cout << "READINP END" << endl;
    readTimeInpS.write(0);
    inpReadReadyS.write(0);
    wait();
  }
}