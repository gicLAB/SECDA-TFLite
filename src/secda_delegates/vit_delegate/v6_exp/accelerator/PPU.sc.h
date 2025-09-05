// ppu_v6.cpp
// V6: The PPU logic is unchanged. It is simply triggered at the correct time
// by the new scheduler. It correctly processes the fully accumulated 'res'
// buffer and then clears it for the next (n, m) tile.
void ACCNAME::PPU() {
  ppuReadyS.write(0);
  pePostTotalS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (ppuReadyS.read() == 0) wait();

    wait(); // a
    cout << "PPU START" << endl;

    pePostTotalS.write(1);

    int ppu_pnr = pN_rem;
    int ppu_pmr = pM_rem;
    int cur_crf = crf;
    int cur_crx = crx;
    int cur_ra = ra;

    wait(); // a

    DWAIT(3);

    for (int i = 0; i < ppu_pnr; i++) {
      DWAIT(1);
      for (int j = 0; j < ppu_pmr; j += 4) {

        res[i][j + 0] += prec[i][j + 0];
        res[i][j + 1] += prec[i][j + 1];
        res[i][j + 2] += prec[i][j + 2];
        res[i][j + 3] += prec[i][j + 3];

        value1 = Quantised_Multiplier(res[i][j + 0], cur_crf, cur_crx);
        value2 = Quantised_Multiplier(res[i][j + 1], cur_crf, cur_crx);
        value3 = Quantised_Multiplier(res[i][j + 2], cur_crf, cur_crx);
        value4 = Quantised_Multiplier(res[i][j + 3], cur_crf, cur_crx);

        svalue1 = value1 + cur_ra;
        svalue2 = value2 + cur_ra;
        svalue3 = value3 + cur_ra;
        svalue4 = value4 + cur_ra;

        if (svalue1 > MAX8) svalue1 = MAX8;
        else if (svalue1 < MIN8) svalue1 = MIN8;
        if (svalue2 > MAX8) svalue2 = MAX8;
        else if (svalue2 < MIN8) svalue2 = MIN8;
        if (svalue3 > MAX8) svalue3 = MAX8;
        else if (svalue3 < MIN8) svalue3 = MIN8;
        if (svalue4 > MAX8) svalue4 = MAX8;
        else if (svalue4 < MIN8) svalue4 = MIN8;

        dout_1 = svalue1.range(7, 0);
        dout_2 = svalue2.range(7, 0);
        dout_3 = svalue3.range(7, 0);
        dout_4 = svalue4.range(7, 0);

        // V6: This clearing action is still correct. It resets the accumulator
        // for the *next* (n, m) tile after the current one is fully processed.
        res[i][j + 0] = 0;
        res[i][j + 1] = 0;
        res[i][j + 2] = 0;
        res[i][j + 3] = 0;

        d_array[j / 4].data =
            Clamp_Combine(dout_1, dout_2, dout_3, dout_4, MAX8, MIN8);
        DWAIT(32);
      }
      wait(); // a

      for (int j = 0; j < ppu_pmr / 4; j++) {
        if (i == (ppu_pnr - 1) && j == (ppu_pmr / 4) - 1) {
          d_array[j].tlast = true;
        } else d_array[j].tlast = false;
        dout1.write(d_array[j]);

        DWAIT(2);
      }
      wait(); // a
    }
    cout << "PPU END" << endl;
    DWAIT(1);
    ppuReadyS.write(0);
    pePostTotalS.write(0);
    wait();
  }
}
