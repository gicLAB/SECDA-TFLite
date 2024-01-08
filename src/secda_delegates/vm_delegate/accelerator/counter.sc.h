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
      int w1 = vars.vars_0.computeS.read();
      int w2 = vars.vars_1.computeS.read();
      int w3 = vars.vars_2.computeS.read();
      int w4 = vars.vars_3.computeS.read();
      int p1 = vars.vars_0.postS.read();
      int p2 = vars.vars_1.postS.read();
      int p3 = vars.vars_2.postS.read();
      int p4 = vars.vars_3.postS.read();
      int s1 = schS.read();
      if (w1 == 1) idle1->value++;
      if (w2 == 1) idle2->value++;
      if (w3 == 1) idle3->value++;
      if (w4 == 1) idle4->value++;
      if (w1 == 3) gemm1->value++;
      if (w2 == 3) gemm2->value++;
      if (w3 == 3) gemm3->value++;
      if (w4 == 3) gemm4->value++;
      if (w1 == 4) wstall1->value++;
      if (w2 == 4) wstall2->value++;
      if (w3 == 4) wstall3->value++;
      if (w4 == 4) wstall4->value++;
      if (p1 != 1) post1->value++;
      if (p2 != 1) post2->value++;
      if (p3 != 1) post3->value++;
      if (p4 != 1) post4->value++;
      shS->increment(s1);
      DWAIT();
    }
    wait();
  }
}
#endif
