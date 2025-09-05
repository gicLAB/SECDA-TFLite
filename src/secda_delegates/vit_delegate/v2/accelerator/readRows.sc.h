#include <systemc.h>

void ACCNAME::ReadRows() {
    int quarter_rem = pN_rem / 4;
    for(k = 0; k < pK; k += 4) {
        #pragma HLS pipeline II = 1
    
    // cout << "pN_rem: " << pN_rem << " quarter_rem: " << quarter_rem << endl;
    d1 = din1->read();
    d2 = din2->read();
    d3 = din3->read();
    d4 = din4->read();

    rows[(0 * quarter_rem) + i][k + 0] = d1.data.range(7, 0);
    rows[(0 * quarter_rem) + i][k + 1] = d1.data.range(15, 8);
    rows[(0 * quarter_rem) + i][k + 2] = d1.data.range(23, 16);
    rows[(0 * quarter_rem) + i][k + 3] = d1.data.range(31, 24);

    rows[(1 * quarter_rem) + i][k + 0] = d2.data.range(7, 0);
    rows[(1 * quarter_rem) + i][k + 1] = d2.data.range(15, 8);
    rows[(1 * quarter_rem) + i][k + 2] = d2.data.range(23, 16);
    rows[(1 * quarter_rem) + i][k + 3] = d2.data.range(31, 24);

    rows[(2 * quarter_rem) + i][k + 0] = d3.data.range(7, 0);
    rows[(2 * quarter_rem) + i][k + 1] = d3.data.range(15, 8);
    rows[(2 * quarter_rem) + i][k + 2] = d3.data.range(23, 16);
    rows[(2 * quarter_rem) + i][k + 3] = d3.data.range(31, 24);

    rows[(3 * quarter_rem) + i][k + 0] = d4.data.range(7, 0);
    rows[(3 * quarter_rem) + i][k + 1] = d4.data.range(15, 8);
    rows[(3 * quarter_rem) + i][k + 2] = d4.data.range(23, 16);
    rows[(3 * quarter_rem) + i][k + 3] = d4.data.range(31, 24);
}
}