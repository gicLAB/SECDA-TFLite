void ACCNAME::Counter() {
    wait();
    while(1) {
        per_batch_cycles->value++;
        if(computeS.read() == 1) active_cycles->value++;
        if(readS.read() == 1) read_cycles->value++;
        wait();
    }
}