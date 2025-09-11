#ifndef __SYNTHESIS__
void ACCNAME::Read_Cycle_Counter() {
  while (1) {
    if (load_wgt_drv.read() && load_data) load_wgts->value++;
    if (load_inp_drv.read() && load_data) load_inps->value++;
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
      int s2 = s1 * 10;

      int w1 = vars.vars_0.computeS.read();
      int p1 = vars.vars_0.postS.read();
      if (w1 == 1) idle1->value++;
      if (w1 == 3) gemm1->value++;
      if (w1 == 4) wstall1->value++;
      if (p1 != 1) post1->value++;
      gmSA->increment(w1 + s2);
      psSA->increment(p1);

#if VMM_COUNT > 1
      int w2 = vars.vars_1.computeS.read();
      int p2 = vars.vars_1.postS.read();
      if (w2 == 1) idle2->value++;
      if (w2 == 3) gemm2->value++;
      if (w2 == 4) wstall2->value++;
      if (p2 != 1) post2->value++;
      gmSB->increment(w2);
      psSB->increment(p2);

#if VMM_COUNT > 2
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

#if VMM_COUNT > 4

      int w5 = vars.vars_4.computeS.read();
      int p5 = vars.vars_4.postS.read();
      if (w5 == 1) idle5->value++;
      if (w5 == 3) gemm5->value++;
      if (w5 == 4) wstall5->value++;
      if (p5 != 1) post5->value++;
      gmSA->increment(w5);
      psSA->increment(p5);

      int w6 = vars.vars_5.computeS.read();
      int p6 = vars.vars_5.postS.read();
      if (w6 == 1) idle6->value++;
      if (w6 == 3) gemm6->value++;
      if (w6 == 4) wstall6->value++;
      if (p6 != 1) post6->value++;
      gmSB->increment(w6);
      psSB->increment(p6);

#if VMM_COUNT > 6
      int w7 = vars.vars_6.computeS.read();
      int p7 = vars.vars_6.postS.read();
      if (w7 == 1) idle7->value++;
      if (w7 == 3) gemm7->value++;
      if (w7 == 4) wstall7->value++;
      if (p7 != 1) post7->value++;
      gmSC->increment(w7);
      psSC->increment(p7);

      int w8 = vars.vars_7.computeS.read();
      int p8 = vars.vars_7.postS.read();
      if (w8 == 1) idle8->value++;
      if (w8 == 3) gemm8->value++;
      if (w8 == 4) wstall8->value++;
      if (p8 != 1) post8->value++;
      gmSD->increment(w8);
      psSD->increment(p8);
#if VMM_COUNT > 8
      int w9 = vars.vars_8.computeS.read();
      int p9 = vars.vars_8.postS.read();
      if (w9 == 1) idle9->value++;
      if (w9 == 3) gemm9->value++;
      if (w9 == 4) wstall9->value++;
      if (p9 != 1) post9->value++;
      gmSA->increment(w9);
      psSA->increment(p9);

      int w10 = vars.vars_9.computeS.read();
      int p10 = vars.vars_9.postS.read();
      if (w10 == 1) idle10->value++;
      if (w10 == 3) gemm10->value++;
      if (w10 == 4) wstall10->value++;
      if (p10 != 1) post10->value++;
      gmSB->increment(w10);
      psSB->increment(p10);

      int w11 = vars.vars_10.computeS.read();
      int p11 = vars.vars_10.postS.read();
      if (w11 == 1) idle11->value++;
      if (w11 == 3) gemm11->value++;
      if (w11 == 4) wstall11->value++;
      if (p11 != 1) post11->value++;
      gmSC->increment(w11);
      psSC->increment(p11);

      int w12 = vars.vars_11.computeS.read();
      int p12 = vars.vars_11.postS.read();
      if (w12 == 1) idle12->value++;
      if (w12 == 3) gemm12->value++;
      if (w12 == 4) wstall12->value++;
      if (p12 != 1) post12->value++;
      gmSD->increment(w12);
      psSD->increment(p12);

      int w13 = vars.vars_12.computeS.read();
      int p13 = vars.vars_12.postS.read();
      if (w13 == 1) idle13->value++;
      if (w13 == 3) gemm13->value++;
      if (w13 == 4) wstall13->value++;
      if (p13 != 1) post13->value++;
      gmSA->increment(w13);
      psSA->increment(p13);

      int w14 = vars.vars_13.computeS.read();
      int p14 = vars.vars_13.postS.read();
      if (w14 == 1) idle14->value++;
      if (w14 == 3) gemm14->value++;
      if (w14 == 4) wstall14->value++;
      if (p14 != 1) post14->value++;
      gmSB->increment(w14);
      psSB->increment(p14);

      int w15 = vars.vars_14.computeS.read();
      int p15 = vars.vars_14.postS.read();
      if (w15 == 1) idle15->value++;
      if (w15 == 3) gemm15->value++;
      if (w15 == 4) wstall15->value++;
      if (p15 != 1) post15->value++;
      gmSC->increment(w15);
      psSC->increment(p15);

      int w16 = vars.vars_15.computeS.read();
      int p16 = vars.vars_15.postS.read();
      if (w16 == 1) idle16->value++;
      if (w16 == 3) gemm16->value++;
      if (w16 == 4) wstall16->value++;
      if (p16 != 1) post16->value++;
      gmSD->increment(w16);
      psSD->increment(p16);
#endif
#endif
#endif
#endif
#endif
      DWAIT();
    }
    wait();
  }
}
#endif