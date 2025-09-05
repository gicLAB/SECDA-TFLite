void ACCNAME::ReadBias() {

  // #pragma HLS pipeline II = 1
  for (int a = 0; a < pN_rem; a++) {   // Loop 1.1.2.2
    for (int b = 0; b < pM_rem; b++) { // Loop 1.1.2.2.1
      res[a][b] = din4->read().data;
    }
  }
}
