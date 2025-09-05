#ifndef __SYNTHESIS__
int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1);
  sc_int<64> result = x_64 * quantized_multiplier_64 + round;
  result = result >> total_shift;
  int nresult = result;
  if (result > MAX32) result = MAX32;
  if (result < MIN32) result = MIN32;
  sc_int<32> result_32 = result;
  return result_32;
}
#else
int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
  sc_int<64> val = x * pl;
  if (val > MAX32) val = MAX32;
  if (val < MIN32) val = MIN32;
  sc_int<64> val_2 = val * qm;
  sc_int<32> temp_1;
  temp_1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
  sc_int<32> val_3 = temp_1;
  val_3 = val_3 >> pr;
  sc_int<32> temp_2 = temp_1 & msk;
  sc_int<32> temp_3 = (temp_1 < 0) & 1;
  sc_int<32> temp_4 = sm + temp_3;
  sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
  sc_int<32> result_32 = val_3 + temp_5;
  return result_32;
}
#endif

sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) {
  sc_int<32> c;
#pragma HLS RESOURCE variable = c core = Mul
  c = a * b;
  return c;
}

ACC_DTYPE<32> ACCNAME::Clamp_Combine(int i1, int i2, int i3, int i4, int qa_max,
                                     int qa_min) {
  // if (i1 < qa_min) i1 = qa_min;
  // else if (i1 > qa_max) i1 = qa_max;
  // if (i2 < qa_min) i2 = qa_min;
  // else if (i2 > qa_max) i2 = qa_max;
  // if (i3 < qa_min) i3 = qa_min;
  // else if (i3 > qa_max) i3 = qa_max;
  // if (i4 < qa_min) i4 = qa_min;
  // else if (i4 > qa_max) i4 = qa_max;
  ACC_DTYPE<32> d;
  d.range(7, 0) = i1;
  d.range(15, 8) = i2;
  d.range(23, 16) = i3;
  d.range(31, 24) = i4;
  return d;
}

void ACCNAME::Compute() {
#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = out_sig
  // Profiling signals
  // readTimeTotalS.write(0);
  // readTimeParamsS.write(0);
  // readTimeRowsS.write(0);
  // readTimeColsS.write(0);
  // readTimeBiasS.write(0);

  // peTotalS.write(0);
  // peMultiplyS.write(0);
  // peAccumulateS.write(0);

  // pePostTotalS.write(0);
  // pePostProcessS.write(0);
  // pePostWriteS.write(0);


  // For HLX
  out_sig.write(0);
  wait();
  DATA a;
  dout2.write(a);
  dout3.write(a);
  dout4.write(a);

  DWAIT(2);

  while (1) {
    // readTimeTotalS.write(1);
    // readTimeParamsS.write(1);

    out_sig.write(1);
    // readS.write(1);
    // PES.write(0);
    pN = din1->read().data;
    pM = din1->read().data;
    pK = din1->read().data;


    crx = din1->read().data;
    crf = din1->read().data;
    ra = din1->read().data;

    no_rows = din1->read().data;
    no_cols = din1->read().data;
    // readTimeParamsS.write(0);

    if (crx > 0) {
      pl = crx;
      pr = 0;
      msk = 0;
      sm = 0;
    } else {
      pl = 1;
      pr = -crx;
      msk = (1 << -crx) - 1;
      sm = msk >> 1;
    }
    DWAIT(12);
    // int tpk = pK;
    for (int n = 0; n < pN; n += no_rows) { // Loop 1.1
      pN_rem = din2->read().data;
      DWAIT(1);

      rowReadReadyS.write(1);
      // wait();
      for (int m = 0; m < pM; m += no_cols) { // Loop 1.1.2
        pM_rem = din3->read().data;

        DWAIT(1);

        colReadReadyS.write(1);
        while (pePostReadyS.read() == 1) wait();

        // wait();
        // ReadBias();
        biasReadReadyS.write(1);

        DWAIT();
        while (rowReadReadyS.read() == 1 || colReadReadyS.read() == 1) {
          wait();
        }
        while(biasReadReadyS.read() == 1) wait();

        // wait();
        // readTimeTotalS.write(0);
        PE();
        // wait();
        // PE_Post();
        pePostReadyS.write(1); 
        // wait();
        // readTimeTotalS.write(1);
        // wait();
      }
      // wait();
    }
    // wait();
    DWAIT();
  }
}