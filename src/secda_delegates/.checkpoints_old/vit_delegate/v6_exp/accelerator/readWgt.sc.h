// readwgt_v6.cpp
#include <systemc.h>

void ACCNAME::ReadWgt() {
  wgtReadReadyS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (wgtReadReadyS.read() == 0) wait();
    DWAIT(1);

    // cout << "READWGT START" << endl;
    wait(); // a

    int wgt_pmr = pM_rem;
    int wgt_pkr = pK_rem;
    wait(); // a

    readTimeWgtS.write(1);
    for (int i = 0; i < wgt_pmr; i++) {
      DWAIT(1);
      // V6: Loop bound changed from pK to pK_rem to read only the k-slice
      for (int k = 0; k < wgt_pkr; k += 4) {
        DATA d = din3->read();
        // The destination 'cols' buffer is indexed from 0 for each slice
        cols[i][k + 0] = d.data.range(7, 0);
        cols[i][k + 1] = d.data.range(15, 8);
        cols[i][k + 2] = d.data.range(23, 16);
        cols[i][k + 3] = d.data.range(31, 24);
        DWAIT(6);
      }
    }
    // cout << "READWGT END" << endl;
    readTimeWgtS.write(0);
    wgtReadReadyS.write(0);
    wait();
  }
}
