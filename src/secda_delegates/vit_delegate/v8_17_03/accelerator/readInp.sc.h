void ACCNAME::ReadInp() {
  wait();
  while (1) {
    while (inpReadReadyS.read() == 0) wait();

    readTimeInpS.write(1);

    int count = inpReadReadyS.read();

    DWAIT(2);

    for (int c = 0; c < count; c++) {
      DWAIT(1);
      for (int i = 0; i < pn_block; i++) {
        DWAIT(1);

        for (int k = 0; k < pK; k += 4) {
          ADATA d = din2->read();

          if (count == 1) { // Dense Broadcast
            for (int core = 0; core < NUM_CORES; core++) {
              rows[core][i][k + 0] = d.data.range(7, 0);
              rows[core][i][k + 1] = d.data.range(15, 8);
              rows[core][i][k + 2] = d.data.range(23, 16);
              rows[core][i][k + 3] = d.data.range(31, 24);
                  DWAIT(3);

            }
          } else { // Mobile Unique
            rows[c][i][k + 0] = d.data.range(7, 0);
            rows[c][i][k + 1] = d.data.range(15, 8);
            rows[c][i][k + 2] = d.data.range(23, 16);
            rows[c][i][k + 3] = d.data.range(31, 24);
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