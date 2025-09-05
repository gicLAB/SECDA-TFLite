#include <systemc.h>

void ACCNAME::ReadCols() {
    int quarter_rem = pM_rem / 4;
    for (k = 0; k < pK; k+=4) {

    #pragma HLS pipeline II = 1
    // // TODO: I think this should be read.data
    d1 = din1->read();
    d2 = din2->read();
    d3 = din3->read();
    d4 = din4->read();

    cols[(0 * quarter_rem) + i][k + 0] = d1.data.range(7, 0);
    cols[(0 * quarter_rem) + i][k + 1] = d1.data.range(15, 8);
    cols[(0 * quarter_rem) + i][k + 2] = d1.data.range(23, 16);
    cols[(0 * quarter_rem) + i][k + 3] = d1.data.range(31, 24);

    cols[(1 * quarter_rem) + i][k + 0] = d2.data.range(7, 0);
    cols[(1 * quarter_rem) + i][k + 1] = d2.data.range(15, 8);
    cols[(1 * quarter_rem) + i][k + 2] = d2.data.range(23, 16);
    cols[(1 * quarter_rem) + i][k + 3] = d2.data.range(31, 24);

    cols[(2 * quarter_rem) + i][k + 0] = d3.data.range(7, 0);
    cols[(2 * quarter_rem) + i][k + 1] = d3.data.range(15, 8);
    cols[(2 * quarter_rem) + i][k + 2] = d3.data.range(23, 16);
    cols[(2 * quarter_rem) + i][k + 3] = d3.data.range(31, 24);

    cols[(3 * quarter_rem) + i][k + 0] = d4.data.range(7, 0);
    cols[(3 * quarter_rem) + i][k + 1] = d4.data.range(15, 8);
    cols[(3 * quarter_rem) + i][k + 2] = d4.data.range(23, 16);
    cols[(3 * quarter_rem) + i][k + 3] = d4.data.range(31, 24);
}
}