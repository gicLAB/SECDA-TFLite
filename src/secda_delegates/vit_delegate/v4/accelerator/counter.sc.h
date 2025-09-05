#ifndef __SYNTHESIS__

void ACCNAME::Counter() {
    wait();
    while(1) {
        // per_batch_cycles->value++;
        if(computeS.read() == 1) { 
            active_cycles->value++;
            // cout << "cycle working" << endl;
        }

        if(readS.read() == 1) {
            read_cycles->value++;
            // cout << "currently reading" << endl;
        }
        
        if(PES.read() == 1) {
            PE_cycles->value++;
            // cout << "PE working" << endl;
        }
        wait();
    }
}

#endif