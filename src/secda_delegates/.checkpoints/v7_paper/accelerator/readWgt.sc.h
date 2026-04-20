#include <systemc.h>

void ACCNAME::ReadWgt() {
    wgtReadReadyS.write(0);
    wait();
    DWAIT(2);

    while(1) {
        while(wgtReadReadyS.read() == 0) wait();
        DWAIT(1);

        // no_wgtR += 1;
        int pmr = pM_rem;
        readTimeWgtS.write(1);
        for (i = 0; i < pmr; i++) {
            DWAIT(1);
            for (k = 0; k < pK; k += 4) {
                ADATA d = din3->read();
                cols[i][k+0] = d.data.range(7, 0);
                cols[i][k+1] = d.data.range(15, 8);
                cols[i][k+2] = d.data.range(23, 16);
                cols[i][k+3] = d.data.range(31, 24);
                DWAIT(6);
            }
        }
        readTimeWgtS.write(0);
        wgtReadReadyS.write(0);
        wait();
    }
}