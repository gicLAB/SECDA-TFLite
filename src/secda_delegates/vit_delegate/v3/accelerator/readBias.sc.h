void ACCNAME::ReadBias() {
  wait();
  while(1) {

    while(biasReadReadyS.read() == 0) { // Synchronisation
      wait();
    }

    biasReadDoneS.write(0);
    wait();

    DWAIT(2);

    // #pragma HLS pipeline II = 1
    for (int a = 0; a < pN_rem; a++) { // Loop 1.1.2.2
      DWAIT(1); 
      for (int b = 0; b < pM_rem; b++) { // Loop 1.1.2.2.1
        res[a][b] += din4->read().data;
        // biasReads += 1;
        // cout << "res[" << a << "][" << b << "] = " << res[a][b] << endl;
        // cout << "reads done: " << reads++ << endl;
        DWAIT(2);
      }
    }
    // cout << "Bias read"<<endl;
    biasReadDoneS.write(1);
    biasReadReadyS.write(0);
    wait();

}
}