#ifndef __SYNTHESIS__

void ACCNAME::Counter() {
    wait();
    while(1) {
        // if(readTimeTotalS.read() == 1) readTimeTotal->value++;
        if(accTotalS.read() == 1) accTotal->value++;
        if(readTimeParamS.read() == 1) readTimeParam->value++;
        if(readTimeInpS.read() == 1) readTimeInp->value++;
        if(readTimeWgtS.read() == 1) readTimeWgt->value++;
        if(readTimeBiasS.read() == 1) readTimeBias->value++;
        if(peTotalS.read() == 1) peTotal->value++;
        // if(peMultiplyS.read() == 1) peMultiply->value++;
        // if(peAccumulateS.read() == 1) peAccumulate->value++;
        if(pePostTotalS.read() == 1) pePostTotal->value++;
        // if(pePostProcessS.read() == 1) pePostProcess->value++;
        // if(pePostWriteS.read() == 1) pePostWrite->value++;

        wait();
    }
}

#endif