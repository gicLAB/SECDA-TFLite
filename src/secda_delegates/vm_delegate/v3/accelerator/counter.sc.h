#ifndef __SYNTHESIS__
void ACCNAME::Read_Cycle_Counter() {
  while (1) {
    if (load_wgt.read() && load_data) load_wgts->value++;
    if (load_inp.read() && load_data) load_inps->value++;
    cycles->value++;
    wait();
  }
}

void ACCNAME::Writer_Cycle_Counter() {
  wait();
  while (1) {
    while (out_check) {
      compute->value++;
      int s1 = schS.read();
      shS->increment(s1);
      int s2 = s1*10;


      int w1 = vars.vars_0.computeS.read();
      int p1 = vars.vars_0.postS.read();
      if (w1 == 1) idle1->value++;
      if (w1 == 3) gemm1->value++;
      if (w1 == 4) wstall1->value++;
      if (p1 != 1) post1->value++;
      gmSA->increment(w1+s2);
      psSA->increment(p1);

      int w2 = vars.vars_1.computeS.read();
      int p2 = vars.vars_1.postS.read();
      if (w2 == 1) idle2->value++;
      if (w2 == 3) gemm2->value++;
      if (w2 == 4) wstall2->value++;
      if (p2 != 1) post2->value++;
      gmSB->increment(w2);
      psSB->increment(p2);

      int w3 = vars.vars_2.computeS.read();
      int p3 = vars.vars_2.postS.read();
      if (w3 == 1) idle3->value++;
      if (w3 == 3) gemm3->value++;
      if (w3 == 4) wstall3->value++;
      // gmSC->increment(w3);
      // psSC->increment(p3);

      int w4 = vars.vars_3.computeS.read();
      int p4 = vars.vars_3.postS.read();
      if (w4 == 1) idle4->value++;
      if (w4 == 3) gemm4->value++;
      if (w4 == 4) wstall4->value++;
      // gmSD->increment(w4);
      // psSD->increment(p4);

      DWAIT();
    }
    wait();
  }
}
#endif
