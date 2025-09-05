void ACCNAME::PE_Post(int i, int j, int crf, int crx, int ra) {

    value1 = Quantised_Multiplier(res[i][j + 0], crf, crx);
    value2 = Quantised_Multiplier(res[i][j + 1], crf, crx);
    value3 = Quantised_Multiplier(res[i][j + 2], crf, crx);
    value4 = Quantised_Multiplier(res[i][j + 3], crf, crx);

    svalue1 = value1 + ra;
    svalue2 = value2 + ra;
    svalue3 = value3 + ra;
    svalue4 = value4 + ra;

    if (svalue1 > MAX8) svalue1 = MAX8;
    if (svalue2 > MAX8) svalue2 = MAX8;
    if (svalue3 > MAX8) svalue3 = MAX8;
    if (svalue4 > MAX8) svalue4 = MAX8;
    if (svalue1 < MIN8) svalue1 = MIN8;
    if (svalue2 < MIN8) svalue2 = MIN8;
    if (svalue3 < MIN8) svalue3 = MIN8;
    if (svalue4 < MIN8) svalue4 = MIN8;

    cur_outs[0] = svalue1.range(7, 0);
    cur_outs[1] = svalue2.range(7, 0);
    cur_outs[2] = svalue3.range(7, 0);
    cur_outs[3] = svalue4.range(7, 0);

    d_array[j / 4].data = Clamp_Combine(cur_outs[0], cur_outs[1], cur_outs[2],
                                                cur_outs[3], MAX8, MIN8);


}