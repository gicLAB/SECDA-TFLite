#include <systemc.h>

void ACCNAME::ReadRows() {
  rowReadReadyS.write(0);
  wait();
  while (1) {
    while (rowReadReadyS.read() == 0) wait();

    wait();

    // #pragma HLS pipeline II = 1
    for (int i_row = 0; i_row < pN_rem; i_row++) { // Loop 1.1.1
      DWAIT(1);
      for (int k_row = 0; k_row < pK; k_row += 4) { // Loop 1.1.1.1
        DATA d_row = din2->read();
        // rowReads += 1;
        rows[i_row][k_row + 0] = d_row.data.range(7, 0);
        rows[i_row][k_row + 1] = d_row.data.range(15, 8);
        rows[i_row][k_row + 2] = d_row.data.range(23, 16);
        rows[i_row][k_row + 3] = d_row.data.range(31, 24);

        // cout << "rows[" << i_row << "][" << k_row+0 << "] = " <<
        // rows[i_row][k_row+0] << endl;
        DWAIT(3); // TODO: For some reason this random DWAIT is breaking
                  // everything ???
      }
      // cout << "i = " << i << endl;
    }
    // cout << "Rows read"<<endl;
    rowReadReadyS.write(0);
    wait();
  }
}