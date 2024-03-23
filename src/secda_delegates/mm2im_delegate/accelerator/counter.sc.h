#ifndef __SYNTHESIS__

void ACCNAME::In_Counter() {
  wait();
  while (1) {
    int ins = inS.read();
    int sched = scheduleS.read();
    int compute = vars.X1.computeS.read();
    int send = vars.X1.sendS.read();
    int load = data_loadS.read();
    int pd = pdS.read();

    T_in->increment(ins);
    T_sh->increment(sched);
    T_ld->increment(load);
    T_com->increment(compute);
    T_sd->increment(send);
    T_pd->increment(pd);

    if (ins == 4) schedule_cycles->value++;
    if (sched == 6) process_cycles->value++;
    if (sched == 71) store_cycles->value++;
    if (sched == 31) update_wgt_cycles->value++;
    if (sched == 5) update_inp_cycles->value++;
    if (compute == 6) compute_cycles->value++;
    if (send == 3) send_cycles->value++;
    if (send == 4) out_cycles->value++;
    if (load == 2) load_wgt_cycles->value++;
    if (load == 3) load_inp_cycles->value++;
    if (load == 4) load_col_map_cycles->value++;
    wait();
  }
}

#endif