int ACCNAME::SHR(int value, int shift) { return value >> shift; }

sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) {
  sc_int<32> c;
#pragma HLS RESOURCE variable = c core = Mul
  c = a * b;
  return c;
}

void ACCNAME::func_HalfAdder(sc_uint<1> a, sc_uint<1> b, sc_uint<1> *sum,
                             sc_uint<1> *Cout) {
  *sum = a ^ b;
  *Cout = (a & b);
}

void ACCNAME::func_FullAdder(sc_uint<1> a, sc_uint<1> b, sc_uint<1> Cin,
                             sc_uint<1> *sum, sc_uint<1> *Cout) {
  sc_uint<1> w1, w2, w3;

  w1 = a ^ b;
  *sum = w1 ^ Cin;

  w2 = a & b;
  w3 = w1 & Cin;
  *Cout = w3 | w2;
}

void ACCNAME::func_ppu(sc_uint<1> Ci, sc_uint<1> Di, sc_uint<1> Cin,
                       sc_uint<1> Sin, sc_uint<1> *Co, sc_uint<1> *Do,
                       sc_uint<1> *Cout, sc_uint<1> *Sout)

{
  sc_uint<1> m;

  *Co = Ci;
  *Do = Di;
  m = Ci & Di;
  func_FullAdder(m, Sin, Cin, Sout, Cout);
}

void ACCNAME::func_cdppu(sc_uint<1> ai, sc_uint<1> bi, sc_uint<1> Sin,
                         sc_uint<1> *ao, sc_uint<1> *bo, sc_uint<1> *Sout) {
  sc_uint<1> m;
  *ao = ai;
  *bo = bi;

  m = ai & bi;
  *Sout = Sin ^ m;
}

void ACCNAME::func_ppuh(sc_uint<1> ai, sc_uint<1> bi, sc_uint<1> Sin,
                        sc_uint<1> *ao, sc_uint<1> *bo, sc_uint<1> *Cout,
                        sc_uint<1> *Sout) {
  sc_uint<1> m;
  *ao = ai;
  *bo = bi;

  m = ai & bi;

  func_HalfAdder(Sin, m, Sout, Cout);
}

void ACCNAME::func_ppuf(sc_uint<1> ai, sc_uint<1> bi, sc_uint<1> aj,
                        sc_uint<1> bj, sc_uint<1> Sin, sc_uint<1> *ao,
                        sc_uint<1> *bo, sc_uint<1> *ajo, sc_uint<1> *bjo,
                        sc_uint<1> *Cout, sc_uint<1> *Sout) {
  sc_uint<1> m1, m2;
  *ao = ai;
  *bo = bi;
  *ajo = aj;
  *bjo = bj;

  m1 = ai & bi;
  m2 = aj & bj;
  func_FullAdder(Sin, m1, m2, Sout, Cout);
}

void ACCNAME::func_sppu(sc_uint<1> Ci, sc_uint<1> Di, sc_uint<1> Cin,
                        sc_uint<1> Sin, sc_uint<1> *Co, sc_uint<1> *Do,
                        sc_uint<1> *Cout, sc_uint<1> *Sout)

{
  sc_uint<1> m;

  *Co = Ci;
  *Do = Di;
  m = ~(Ci & Di);
  func_FullAdder(m, Sin, Cin, Sout, Cout);
}

sc_uint<12> ACCNAME::func_sgroupAX_4(sc_uint<8> c, sc_uint<4> d) {
  sc_uint<1> result[5];
  sc_uint<12> resultX;
  sc_uint<1> w0[8], w11[8], w12[8], w13[8], w14[8], w21[8], w22[8], w23[8],
      w24[8];
  sc_uint<1> w31[8], w32[8], w33[8], w34[8];

  // row0
  w0[0] = c[0] & d[0];
  w0[1] = c[1] & d[0];
  w0[2] = c[2] & d[0];
  w0[3] = c[3] & d[0];
  w0[4] = c[4] & d[0];
  w0[5] = c[5] & d[0];
  w0[6] = c[6] & d[0];
  w0[7] = ~(c[7] & d[0]);

  resultX[0] = w0[0];

  func_cdppu(c.range(0, 0), d.range(1, 1), w0[1], &w13[0], &w11[0], &result[1]);
  func_cdppu(c.range(1, 1), w11[0], w0[2], &w13[1], &w11[1], &w14[1]);
  func_cdppu(c.range(2, 2), w11[1], w0[3], &w13[2], &w11[2], &w14[2]);

  func_cdppu(w13[0], d.range(2, 2), w14[1], &w23[0], &w21[0], &result[2]);
  func_cdppu(w13[1], w21[0], w14[2], &w23[1], &w21[1], &w24[1]);

  func_ppuf(c.range(3, 3), w11[2], w13[2], w21[1], w0[4], &w13[3], &w11[3],
            &w23[2], &w21[2], &w22[2], &w24[2]);

  func_cdppu(w23[0], d.range(3, 3), w24[1], &w33[0], &w31[0], &result[3]);
  func_ppuh(w23[1], w31[0], w24[2], &w33[1], &w31[1], &w32[1], &result[4]);

  func_ppuh(c.range(4, 4), w11[3], w0[5], &w13[4], &w11[4], &w12[4], &w14[4]);

  func_ppu(c.range(5, 5), w11[4], w12[4], w0[6], &w13[5], &w11[5], &w12[5],
           &w14[5]);
  func_ppu(c.range(6, 6), w11[5], w12[5], w0[7], &w13[6], &w11[6], &w12[6],
           &w14[6]);
  func_sppu(c.range(7, 7), w11[6], w12[6], 1, &w13[7], &w11[7], &w12[7],
            &w14[7]);

  func_ppu(w13[3], w21[2], w22[2], w14[4], &w23[3], &w21[3], &w22[3], &w24[3]);
  func_ppu(w13[4], w21[3], w22[3], w14[5], &w23[4], &w21[4], &w22[4], &w24[4]);
  func_ppu(w13[5], w21[4], w22[4], w14[6], &w23[5], &w21[5], &w22[5], &w24[5]);
  func_ppu(w13[6], w21[5], w22[5], w14[7], &w23[6], &w21[6], &w22[6], &w24[6]);
  func_sppu(w13[7], w21[6], w22[6], w12[7], &w23[7], &w21[7], &w22[7], &w24[7]);

  func_ppu(w23[2], w31[1], w32[1], w24[3], &w33[2], &w31[2], &w32[2], &w34[2]);
  func_ppu(w23[3], w31[2], w32[2], w24[4], &w33[3], &w31[3], &w32[3], &w34[3]);
  func_ppu(w23[4], w31[3], w32[3], w24[5], &w33[4], &w31[4], &w32[4], &w34[4]);
  func_ppu(w23[5], w31[4], w32[4], w24[6], &w33[5], &w31[5], &w32[5], &w34[5]);
  func_ppu(w23[6], w31[5], w32[5], w24[7], &w33[6], &w31[6], &w32[6], &w34[6]);
  func_sppu(w23[7], w31[6], w32[6], w22[7], &w33[7], &w31[7], &w32[7], &w34[7]);

  resultX[1] = result[1];
  resultX[2] = result[2];
  resultX[3] = result[3];
  resultX[4] = result[4];

  resultX[5] = w34[2];
  resultX[6] = w34[3];
  resultX[7] = w34[4];
  resultX[8] = w34[5];
  resultX[9] = w34[6];
  resultX[10] = w34[7];
  resultX[11] = w32[7];

  return resultX;
}

sc_uint<12> ACCNAME::func_groupB(sc_uint<8> c, sc_uint<4> d) {
  sc_uint<12> result;
  sc_uint<1> w0[8], w11[8], w12[8], w13[8], w14[8], w21[8], w22[8], w23[8],
      w24[8];
  sc_uint<1> w31[8], w32[8], w33[8], w34[8], test[2];
  // row0
  w0[0] = c[0] & d[0];
  w0[1] = c[1] & d[0];
  w0[2] = c[2] & d[0];
  w0[3] = c[3] & d[0];
  w0[4] = c[4] & d[0];
  w0[5] = c[5] & d[0];
  w0[6] = c[6] & d[0];
  w0[7] = ~(c[7] & d[0]);

  result[0] = w0[0];

  // ppu of row1
  func_ppu(c.range(0, 0), d.range(1, 1), 0, w0[1], &w13[0], &w11[0], &w12[0],
           &w14[0]);
  func_ppu(c.range(1, 1), w11[0], w12[0], w0[2], &w13[1], &w11[1], &w12[1],
           &w14[1]);
  func_ppu(c.range(2, 2), w11[1], w12[1], w0[3], &w13[2], &w11[2], &w12[2],
           &w14[2]);
  func_ppu(c.range(3, 3), w11[2], w12[2], w0[4], &w13[3], &w11[3], &w12[3],
           &w14[3]);
  func_ppu(c.range(4, 4), w11[3], w12[3], w0[5], &w13[4], &w11[4], &w12[4],
           &w14[4]);
  func_ppu(c.range(5, 5), w11[4], w12[4], w0[6], &w13[5], &w11[5], &w12[5],
           &w14[5]);
  func_ppu(c.range(6, 6), w11[5], w12[5], w0[7], &w13[6], &w11[6], &w12[6],
           &w14[6]);
  func_sppu(c.range(7, 7), w11[6], w12[6], 0, &w13[7], &w11[7], &w12[7],
            &w14[7]);

  result[1] = w14[0];

  // ppu of row2
  func_ppu(w13[0], d.range(2, 2), 0, w14[1], &w23[0], &w21[0], &w22[0],
           &w24[0]);
  func_ppu(w13[1], w21[0], w22[0], w14[2], &w23[1], &w21[1], &w22[1], &w24[1]);
  func_ppu(w13[2], w21[1], w22[1], w14[3], &w23[2], &w21[2], &w22[2], &w24[2]);
  func_ppu(w13[3], w21[2], w22[2], w14[4], &w23[3], &w21[3], &w22[3], &w24[3]);
  func_ppu(w13[4], w21[3], w22[3], w14[5], &w23[4], &w21[4], &w22[4], &w24[4]);
  func_ppu(w13[5], w21[4], w22[4], w14[6], &w23[5], &w21[5], &w22[5], &w24[5]);
  func_ppu(w13[6], w21[5], w22[5], w14[7], &w23[6], &w21[6], &w22[6], &w24[6]);
  func_sppu(w13[7], w21[6], w22[6], w12[7], &w23[7], &w21[7], &w22[7], &w24[7]);

  result[2] = w24[0];

  // ppu of row3
  func_sppu(w23[0], d.range(3, 3), 0, w24[1], &w33[0], &w31[0], &w32[0],
            &w34[0]);
  func_sppu(w23[1], w31[0], w32[0], w24[2], &w33[1], &w31[1], &w32[1], &w34[1]);
  func_sppu(w23[2], w31[1], w32[1], w24[3], &w33[2], &w31[2], &w32[2], &w34[2]);
  func_sppu(w23[3], w31[2], w32[2], w24[4], &w33[3], &w31[3], &w32[3], &w34[3]);
  func_sppu(w23[4], w31[3], w32[3], w24[5], &w33[4], &w31[4], &w32[4], &w34[4]);
  func_sppu(w23[5], w31[4], w32[4], w24[6], &w33[5], &w31[5], &w32[5], &w34[5]);
  func_sppu(w23[6], w31[5], w32[5], w24[7], &w33[6], &w31[6], &w32[6], &w34[6]);
  func_ppu(w23[7], w31[6], w32[6], w22[7], &w33[7], &w31[7], &w32[7], &w34[7]);

  func_HalfAdder(1, w32[7], &test[0], &test[1]);

  result[3] = w34[0];
  result[4] = w34[1];
  result[5] = w34[2];
  result[6] = w34[3];
  result[7] = w34[4];
  result[8] = w34[5];
  result[9] = w34[6];
  result[10] = w34[7];
  result[11] = test[0];

  return result;
}

sc_uint<12> ACCNAME::func_cla(sc_uint<12> a, sc_uint<12> b) {
  sc_uint<12> result;
  sc_uint<1> p[12], g[12], c[12];

  p[0] = a[0] ^ b[0];
  g[0] = a[0] & b[0];

  p[1] = a[1] ^ b[1];
  g[1] = a[1] & b[1];

  p[2] = a[2] ^ b[2];
  g[2] = a[2] & b[2];

  p[3] = a[3] ^ b[3];
  g[3] = a[3] & b[3];

  p[4] = a[4] ^ b[4];
  g[4] = a[4] & b[4];

  p[5] = a[5] ^ b[5];
  g[5] = a[5] & b[5];

  p[6] = a[6] ^ b[6];
  g[6] = a[6] & b[6];

  p[7] = a[7] ^ b[7];
  g[7] = a[7] & b[7];

  p[8] = a[8] ^ b[8];
  g[8] = a[8] & b[8];

  p[9] = a[9] ^ b[9];
  g[9] = a[9] & b[9];

  p[10] = a[10] ^ b[10];
  g[10] = a[10] & b[10];

  p[11] = a[11] ^ b[11];
  g[11] = a[11] & b[11];

  c[1] = g[0];
  c[2] = g[1] || (p[1] && g[0]);
  c[3] = g[2] || (p[2] && g[1]) || (p[1] && p[2] && g[0]);
  c[4] = g[3] || (p[3] && g[2]) || (p[2] && p[3] && g[1]) ||
         (p[1] && p[2] && p[3] && g[0]);
  c[5] = g[4] || (p[4] && g[3]) || (p[3] && p[4] && g[2]) ||
         (p[2] && p[3] && p[4] && g[1]) ||
         (p[1] && p[2] && p[3] && p[4] && g[0]);
  c[6] = g[5] || (p[5] && g[4]) || (p[4] && p[5] && g[3]) ||
         (p[3] && p[4] && p[5] && g[2]) ||
         (p[2] && p[3] && p[4] && p[5] && g[1]) ||
         (p[1] && p[2] && p[3] && p[4] && p[5] && g[0]);
  c[7] = g[6] || (p[6] && g[5]) || (p[5] && p[6] && g[4]) ||
         (p[4] && p[5] && p[6] && g[3]) ||
         (p[3] && p[4] && p[5] && p[6] && g[2]) ||
         (p[2] && p[3] && p[4] && p[5] && p[6] && g[1]) ||
         (p[1] && p[2] && p[3] && p[4] && p[5] && p[6] && g[0]);
  c[8] = g[7] || (p[7] && g[6]) || (p[6] && p[7] && g[5]) ||
         (p[5] && p[6] && p[7] && g[4]) ||
         (p[4] && p[5] && p[6] && p[7] && g[3]) ||
         (p[3] && p[4] && p[5] && p[6] && p[7] && g[2]) ||
         (p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && g[1]) ||
         (p[1] && p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && g[0]);
  c[9] = g[8] || (p[8] && g[7]) || (p[7] && p[8] && g[6]) ||
         (p[6] && p[7] && p[8] && g[5]) ||
         (p[5] && p[6] && p[7] && p[8] && g[4]) ||
         (p[4] && p[5] && p[6] && p[7] && p[8] && g[3]) ||
         (p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && g[2]) ||
         (p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && g[1]) ||
         (p[1] && p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && g[0]);
  c[10] =
      g[9] || (p[9] && g[8]) || (p[8] && p[9] && g[7]) ||
      (p[7] && p[8] && p[9] && g[6]) ||
      (p[6] && p[7] && p[8] && p[9] && g[5]) ||
      (p[5] && p[6] && p[7] && p[8] && p[9] && g[4]) ||
      (p[4] && p[5] && p[6] && p[7] && p[8] && p[9] && g[3]) ||
      (p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && p[9] && g[2]) ||
      (p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && p[9] && g[1]) ||
      (p[1] && p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && p[9] &&
       g[0]);
  c[11] =
      g[10] || (p[10] && g[9]) || (p[9] && p[10] && g[8]) ||
      (p[8] && p[9] && p[10] && g[7]) ||
      (p[7] && p[8] && p[9] && p[10] && g[6]) ||
      (p[6] && p[7] && p[8] && p[9] && p[10] && g[5]) ||
      (p[5] && p[6] && p[7] && p[8] && p[9] && p[10] && g[4]) ||
      (p[4] && p[5] && p[6] && p[7] && p[8] && p[9] && p[10] && g[3]) ||
      (p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && p[9] && p[10] && g[2]) ||
      (p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && p[9] && p[10] &&
       g[1]) ||
      (p[1] && p[2] && p[3] && p[4] && p[5] && p[6] && p[7] && p[8] && p[9] &&
       p[10] && g[0]);

  result[0] = p[0];
  result[1] = p[1] ^ c[1];
  result[2] = p[2] ^ c[2];
  result[3] = p[3] ^ c[3];
  result[4] = p[4] ^ c[4];
  result[5] = p[5] ^ c[5];
  result[6] = p[6] ^ c[6];
  result[7] = p[7] ^ c[7];
  result[8] = p[8] ^ c[8];
  result[9] = p[9] ^ c[9];
  result[10] = p[10] ^ c[10];
  result[11] = p[11] ^ c[11];

  return result;
}

// sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) {
//   sc_int<32> out_c;
//   sc_uint<8> c;
//   sc_uint<8> d;
//   sc_uint<12> rA, rA_cus, rB;
//   sc_uint<16> resultX;
//   c = b;
//   d = a;
//   rA = func_sgroupAX_4(c, d.range(3, 0));
//   rB = func_groupB(c, d.range(7, 4));
//   resultX.range(3, 0) = rA.range(3, 0);
//   rA_cus.range(11, 8) = 0;
//   rA_cus.range(7, 0) = rA.range(11, 4);
//   resultX.range(15, 4) = func_cla(rA_cus, rB);
//   out_c = resultX;
//   return out_c;
// }

void ACCNAME::Output_Handler() {
  bool ready = false;
  bool resetted = true;
  DATA last = {5000, 1};
  wait();
  while (1) {
    while (out_check.read() && !ready && resetted) {
      bool w1 = w1S.read() == 10;
      bool w2 = w2S.read() == 10;
      bool w3 = w3S.read() == 10;
      bool w4 = w4S.read() == 10;

      bool wr1 = !write1.read();
      bool wr2 = !write2.read();
      bool wr3 = !write3.read();
      bool wr4 = !write4.read();

      bool block_done = !schedule.read();

      ready = block_done && w1 && w2 && w3 && w4 && wr1 && wr2 && wr3 && wr4;

      if (ready) {
        dout1.write(last);
        dout2.write(last);
        dout3.write(last);
        dout4.write(last);
        out_check.write(0);
        resetted = false;
      }
      wait();
      DWAIT(4);
    }

    if (!out_check.read()) {
      resetted = true;
      ready = false;
    }
    wait();
    DWAIT();
  }
}
