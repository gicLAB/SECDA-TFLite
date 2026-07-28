// HWC (ReadInp):
//     0 Idle
//     1 Reading

void ACCNAME::ReadInp() {
  wait();
  while (1) {

    HWC_SIG(ReadInp, 0);

    while (inpReadReadyS.read() == 0) wait();

    HWC_SIG(ReadInp, 1);


    readTimeInpS.write(1);

    int count = inpReadReadyS.read();

    DWAIT(2);

    for (int c = 0; c < count; c++) {
      DWAIT(1);
      for (int i = 0; i < pn_block; i++) {
        DWAIT(1);

        for (int k = 0; k < pK; k += 4) {
          ADATA d = din2->read();

          if (count == 1) { 
            for (int core = 0; core < NUM_CORES; core++) {
              // OPTIMIZATION: Write 32-bit chunk directly
              rows_packed[core][i][k/4] = d.data;
              DWAIT(3);
            }
          } else { 
            rows_packed[c][i][k/4] = d.data;
            DWAIT(4);
          }
        }
      }
    }
    DWAIT(1);
    inpReadReadyS.write(0);
    readTimeInpS.write(0);
    wait();
  }
}

