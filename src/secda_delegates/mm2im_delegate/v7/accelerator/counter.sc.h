#ifndef __SYNTHESIS__

void ACCNAME::In_Counter() {
  wait();
  while (1) {
    int ins = inS.read();
    int sched = scheduleS.read();
    int load = data_loadS.read();
    int pd = pdS.read();

    int com_1 = vars.X1.computeS.read();
    int com_2 = vars.X2.computeS.read();
    int com_3 = vars.X3.computeS.read();
    int com_4 = vars.X4.computeS.read();
    int com_5 = vars.X5.computeS.read();
    int com_6 = vars.X6.computeS.read();
    int com_7 = vars.X7.computeS.read();
    int com_8 = vars.X8.computeS.read();



    int out_1 = vars.X1.sendS.read();
    int out_2 = vars.X2.sendS.read();
    int out_3 = vars.X3.sendS.read();
    int out_4 = vars.X4.sendS.read();
    int out_5 = vars.X5.sendS.read();
    int out_6 = vars.X6.sendS.read();
    int out_7 = vars.X7.sendS.read();
    int out_8 = vars.X8.sendS.read();

    T_in->increment(ins);
    T_sh->increment(sched);
    T_ld->increment(load);
    T_pd->increment(pd);
    T_com_1->increment(com_1);
    T_com_2->increment(com_2);
    T_com_3->increment(com_3);
    T_com_4->increment(com_4);
    T_com_5->increment(com_5);
    T_com_6->increment(com_6);
    T_com_7->increment(com_7);
    T_com_8->increment(com_8);
    T_out_1->increment(out_1);
    T_out_2->increment(out_2);
    T_out_3->increment(out_3);
    T_out_4->increment(out_4);
    T_out_5->increment(out_5);
    T_out_6->increment(out_6);
    T_out_7->increment(out_7);
    T_out_8->increment(out_8);

    if (ins == 4) schedule_cycles->value++;
    if (sched == 6) process_cycles->value++;
    if (sched == 71) store_cycles->value++;
    if (sched == 31) update_wgt_cycles->value++;
    if (sched == 5) update_inp_cycles->value++;
    if (com_1 == 6) compute_cycles->value++;
    if (out_1 == 3) send_cycles->value++;
    if (out_1 == 4) out_cycles->value++;
    if (load == 2) load_wgt_cycles->value++;
    if (load == 3) load_inp_cycles->value++;
    if (load == 4) load_col_map_cycles->value++;
    wait();
  }
}

#endif