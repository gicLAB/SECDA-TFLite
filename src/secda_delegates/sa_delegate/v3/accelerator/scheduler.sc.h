void ACCNAME::schedule_gemm_unit(int unit_counter, int w_idx, int i_idx,
                                 int l, int r, int rb_over, int lb_over) {
  int d = (depth / 4);
  while (!gemm_unit_1_ready.read()) wait();
  gemm_unit_1_ready.write(0);

  schS.write(51);
  wait();

  for (int i = 0; i < 15; i++) {
    if (i <= 0) sIs2.write(0);
    if (i <= 1) sIs3.write(0);
    if (i <= 2) sIs4.write(0);
    if (i <= 3) sIs5.write(0);
    if (i <= 4) sIs6.write(0);
    if (i <= 5) sIs7.write(0);
    if (i <= 6) sIs8.write(0);
    if (i <= 7) sIs9.write(0);
    if (i <= 8) sIs10.write(0);
    if (i <= 9) sIs11.write(0);
    if (i <= 10) sIs12.write(0);
    if (i <= 11) sIs13.write(0);
    if (i <= 12) sIs14.write(0);
    if (i <= 13) sIs15.write(0);
    if (i <= 14) sIs16.write(0);
    if (i <= 0) sWs2.write(0);
    if (i <= 1) sWs3.write(0);
    if (i <= 2) sWs4.write(0);
    if (i <= 3) sWs5.write(0);
    if (i <= 4) sWs6.write(0);
    if (i <= 5) sWs7.write(0);
    if (i <= 6) sWs8.write(0);
    if (i <= 7) sWs9.write(0);
    if (i <= 8) sWs10.write(0);
    if (i <= 9) sWs11.write(0);
    if (i <= 10) sWs12.write(0);
    if (i <= 11) sWs13.write(0);
    if (i <= 12) sWs14.write(0);
    if (i <= 13) sWs15.write(0);
    if (i <= 14) sWs16.write(0);
    DWAIT();
  }

  schS.write(52);
  for (int rin = 0; rin < d; rin++) {
    ACC_DTYPE<32> i1 = wgt_data1a[w_idx];
    ACC_DTYPE<32> i2 = wgt_data2a[w_idx];
    ACC_DTYPE<32> i3 = wgt_data3a[w_idx];
    ACC_DTYPE<32> i4 = wgt_data4a[w_idx];
    ACC_DTYPE<32> i5 = wgt_data1a[w_idx + d];
    ACC_DTYPE<32> i6 = wgt_data2a[w_idx + d];
    ACC_DTYPE<32> i7 = wgt_data3a[w_idx + d];
    ACC_DTYPE<32> i8 = wgt_data4a[w_idx + d];
    ACC_DTYPE<32> i9 = wgt_data1a[w_idx + (2 * d)];
    ACC_DTYPE<32> i10 = wgt_data2a[w_idx + (2 * d)];
    ACC_DTYPE<32> i11 = wgt_data3a[w_idx + (2 * d)];
    ACC_DTYPE<32> i12 = wgt_data4a[w_idx + (2 * d)];
    ACC_DTYPE<32> i13 = wgt_data1a[w_idx + (3 * d)];
    ACC_DTYPE<32> i14 = wgt_data2a[w_idx + (3 * d)];
    ACC_DTYPE<32> i15 = wgt_data3a[w_idx + (3 * d)];
    ACC_DTYPE<32> i16 = wgt_data4a[w_idx + (3 * d)];

    ACC_DTYPE<32> w1 = inp_data1[i_idx];
    ACC_DTYPE<32> w2 = inp_data2[i_idx];
    ACC_DTYPE<32> w3 = inp_data3[i_idx];
    ACC_DTYPE<32> w4 = inp_data4[i_idx];
    ACC_DTYPE<32> w5 = inp_data1[i_idx + d];
    ACC_DTYPE<32> w6 = inp_data2[i_idx + d];
    ACC_DTYPE<32> w7 = inp_data3[i_idx + d];
    ACC_DTYPE<32> w8 = inp_data4[i_idx + d];
    ACC_DTYPE<32> w9 = inp_data1[i_idx + (2 * d)];
    ACC_DTYPE<32> w10 = inp_data2[i_idx + (2 * d)];
    ACC_DTYPE<32> w11 = inp_data3[i_idx + (2 * d)];
    ACC_DTYPE<32> w12 = inp_data4[i_idx + (2 * d)];
    ACC_DTYPE<32> w13 = inp_data1[i_idx + (3 * d)];
    ACC_DTYPE<32> w14 = inp_data2[i_idx + (3 * d)];
    ACC_DTYPE<32> w15 = inp_data3[i_idx + (3 * d)];
    ACC_DTYPE<32> w16 = inp_data4[i_idx + (3 * d)];

    sIs1.write(i1.range(7, 0));
    sIs1.write(i1.range(15, 8));
    sIs1.write(i1.range(23, 16));
    sIs1.write(i1.range(31, 24));
    sIs2.write(i2.range(7, 0));
    sIs2.write(i2.range(15, 8));
    sIs2.write(i2.range(23, 16));
    sIs2.write(i2.range(31, 24));
    sIs3.write(i3.range(7, 0));
    sIs3.write(i3.range(15, 8));
    sIs3.write(i3.range(23, 16));
    sIs3.write(i3.range(31, 24));
    sIs4.write(i4.range(7, 0));
    sIs4.write(i4.range(15, 8));
    sIs4.write(i4.range(23, 16));
    sIs4.write(i4.range(31, 24));
    sIs5.write(i5.range(7, 0));
    sIs5.write(i5.range(15, 8));
    sIs5.write(i5.range(23, 16));
    sIs5.write(i5.range(31, 24));
    sIs6.write(i6.range(7, 0));
    sIs6.write(i6.range(15, 8));
    sIs6.write(i6.range(23, 16));
    sIs6.write(i6.range(31, 24));
    sIs7.write(i7.range(7, 0));
    sIs7.write(i7.range(15, 8));
    sIs7.write(i7.range(23, 16));
    sIs7.write(i7.range(31, 24));
    sIs8.write(i8.range(7, 0));
    sIs8.write(i8.range(15, 8));
    sIs8.write(i8.range(23, 16));
    sIs8.write(i8.range(31, 24));
    sIs9.write(i9.range(7, 0));
    sIs9.write(i9.range(15, 8));
    sIs9.write(i9.range(23, 16));
    sIs9.write(i9.range(31, 24));
    sIs10.write(i10.range(7, 0));
    sIs10.write(i10.range(15, 8));
    sIs10.write(i10.range(23, 16));
    sIs10.write(i10.range(31, 24));
    sIs11.write(i11.range(7, 0));
    sIs11.write(i11.range(15, 8));
    sIs11.write(i11.range(23, 16));
    sIs11.write(i11.range(31, 24));
    sIs12.write(i12.range(7, 0));
    sIs12.write(i12.range(15, 8));
    sIs12.write(i12.range(23, 16));
    sIs12.write(i12.range(31, 24));
    sIs13.write(i13.range(7, 0));
    sIs13.write(i13.range(15, 8));
    sIs13.write(i13.range(23, 16));
    sIs13.write(i13.range(31, 24));
    sIs14.write(i14.range(7, 0));
    sIs14.write(i14.range(15, 8));
    sIs14.write(i14.range(23, 16));
    sIs14.write(i14.range(31, 24));
    sIs15.write(i15.range(7, 0));
    sIs15.write(i15.range(15, 8));
    sIs15.write(i15.range(23, 16));
    sIs15.write(i15.range(31, 24));
    sIs16.write(i16.range(7, 0));
    sIs16.write(i16.range(15, 8));
    sIs16.write(i16.range(23, 16));
    sIs16.write(i16.range(31, 24));

    sWs1.write(w1.range(7, 0));
    sWs1.write(w1.range(15, 8));
    sWs1.write(w1.range(23, 16));
    sWs1.write(w1.range(31, 24));
    sWs2.write(w2.range(7, 0));
    sWs2.write(w2.range(15, 8));
    sWs2.write(w2.range(23, 16));
    sWs2.write(w2.range(31, 24));
    sWs3.write(w3.range(7, 0));
    sWs3.write(w3.range(15, 8));
    sWs3.write(w3.range(23, 16));
    sWs3.write(w3.range(31, 24));
    sWs4.write(w4.range(7, 0));
    sWs4.write(w4.range(15, 8));
    sWs4.write(w4.range(23, 16));
    sWs4.write(w4.range(31, 24));
    sWs5.write(w5.range(7, 0));
    sWs5.write(w5.range(15, 8));
    sWs5.write(w5.range(23, 16));
    sWs5.write(w5.range(31, 24));
    sWs6.write(w6.range(7, 0));
    sWs6.write(w6.range(15, 8));
    sWs6.write(w6.range(23, 16));
    sWs6.write(w6.range(31, 24));
    sWs7.write(w7.range(7, 0));
    sWs7.write(w7.range(15, 8));
    sWs7.write(w7.range(23, 16));
    sWs7.write(w7.range(31, 24));
    sWs8.write(w8.range(7, 0));
    sWs8.write(w8.range(15, 8));
    sWs8.write(w8.range(23, 16));
    sWs8.write(w8.range(31, 24));
    sWs9.write(w9.range(7, 0));
    sWs9.write(w9.range(15, 8));
    sWs9.write(w9.range(23, 16));
    sWs9.write(w9.range(31, 24));
    sWs10.write(w10.range(7, 0));
    sWs10.write(w10.range(15, 8));
    sWs10.write(w10.range(23, 16));
    sWs10.write(w10.range(31, 24));
    sWs11.write(w11.range(7, 0));
    sWs11.write(w11.range(15, 8));
    sWs11.write(w11.range(23, 16));
    sWs11.write(w11.range(31, 24));
    sWs12.write(w12.range(7, 0));
    sWs12.write(w12.range(15, 8));
    sWs12.write(w12.range(23, 16));
    sWs12.write(w12.range(31, 24));
    sWs13.write(w13.range(7, 0));
    sWs13.write(w13.range(15, 8));
    sWs13.write(w13.range(23, 16));
    sWs13.write(w13.range(31, 24));
    sWs14.write(w14.range(7, 0));
    sWs14.write(w14.range(15, 8));
    sWs14.write(w14.range(23, 16));
    sWs14.write(w14.range(31, 24));
    sWs15.write(w15.range(7, 0));
    sWs15.write(w15.range(15, 8));
    sWs15.write(w15.range(23, 16));
    sWs15.write(w15.range(31, 24));
    sWs16.write(w16.range(7, 0));
    sWs16.write(w16.range(15, 8));
    sWs16.write(w16.range(23, 16));
    sWs16.write(w16.range(31, 24));
    w_idx++;
    i_idx++;
    DWAIT(8);
  }

  schS.write(53);
  for (int i = 0; i < 30; i++) {
    if (i <= 29) sIs1.write(0);
    if (i <= 28) sIs2.write(0);
    if (i <= 27) sIs3.write(0);
    if (i <= 26) sIs4.write(0);
    if (i <= 25) sIs5.write(0);
    if (i <= 24) sIs6.write(0);
    if (i <= 23) sIs7.write(0);
    if (i <= 22) sIs8.write(0);
    if (i <= 21) sIs9.write(0);
    if (i <= 20) sIs10.write(0);
    if (i <= 19) sIs11.write(0);
    if (i <= 18) sIs12.write(0);
    if (i <= 17) sIs13.write(0);
    if (i <= 16) sIs14.write(0);
    if (i <= 15) sIs15.write(0);
    if (i <= 14) sIs16.write(0);

    if (i <= 29) sWs1.write(0);
    if (i <= 28) sWs2.write(0);
    if (i <= 27) sWs3.write(0);
    if (i <= 26) sWs4.write(0);
    if (i <= 25) sWs5.write(0);
    if (i <= 24) sWs6.write(0);
    if (i <= 23) sWs7.write(0);
    if (i <= 22) sWs8.write(0);
    if (i <= 21) sWs9.write(0);
    if (i <= 20) sWs10.write(0);
    if (i <= 19) sWs11.write(0);
    if (i <= 18) sWs12.write(0);
    if (i <= 17) sWs13.write(0);
    if (i <= 16) sWs14.write(0);
    if (i <= 15) sWs15.write(0);
    if (i <= 14) sWs16.write(0);
    DWAIT();
  }
  WRQ3.write(crx[l]);
  WRQ3.write(crx[l + 1]);
  WRQ3.write(crx[l + 2]);
  WRQ3.write(crx[l + 3]);
  wait();
  DWAIT(6);

  schS.write(54);
  for (int i = 0; i < 4; i++) {
    WRQ1.write(inp_sum1[r + i]);
    WRQ1.write(inp_sum2[r + i]);
    WRQ1.write(inp_sum3[r + i]);
    WRQ1.write(inp_sum4[r + i]);

    WRQ2.write(wgt_sum1[l + i]);
    WRQ2.write(wgt_sum2[l + i]);
    WRQ2.write(wgt_sum3[l + i]);
    WRQ2.write(wgt_sum4[l + i]);

    WRQ3.write(crf1[l + i]);
    WRQ3.write(crf2[l + i]);
    WRQ3.write(crf3[l + i]);
    WRQ3.write(crf4[l + i]);
    DWAIT(7);
  }

  schS.write(55);
  WRQ1.write(rb_over);
  WRQ2.write(lb_over);
  schS.write(56);
  wait();
}

void ACCNAME::Scheduler() {
  int unit_counter = 0;
  gemm_unit_1_ready.write(1);
  schS.write(0);
  wait();
  while (1) {
    schS.write(10);
    while (!schedule.read()) wait();

    schS.write(1);
    int dm = depth / 4;
    for (int r = 0; r < inp_block; r += 16) {
      int r4 = r / 4;
      int i_idx = r4 * dm;
      schS.write(2);
      int rb_over = ((r + 16) - inp_block);
      DWAIT(7);
      schS.write(4);
      for (int l = 0; l < wgt_block; l += 16) {
        schS.write(5);
        int l4 = l / 4;
        int w_idx = l4 * dm;
        int lb_over = ((l + 16) - wgt_block);
        DWAIT(8);
        schedule_gemm_unit(unit_counter, w_idx, i_idx, l4, r4, rb_over,
                           lb_over);
        schS.write(6);
        wait();
        DWAIT(2);
      }
    }

    schS.write(7);
    schedule.write(0);
    schS.write(8);
    wait();
  }
}
