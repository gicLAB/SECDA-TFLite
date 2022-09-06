#include "acc.h"
void ACCNAME::Data_In1(){
int llength=0;
int rlength=0;

wait();
  while(1){
    while(!read_inputs.read())wait();
    llength = llen.read();
    rlength = rlen.read();
	int la=0;
    int lb=0;
	int ra=0;
	int rb=0;

    DWAIT();
    if(ltake.read()){
    	DWAIT(2);
    	for (int i = 0; i <llength/4; i++){
			ACC_DTYPE<32> data=din1.read().data;
    		lb++;
            lhsdata1a[la] =data;  //FIFO
			la=lb;
#ifndef __SYNTHESIS__
			input_max_index=la;
			DWAIT();
#endif
    	}
		for (int i = 0; i <rlength; i++){
			ACC_DTYPE<32> wsums=din1.read().data;
			ACC_DTYPE<32> rfs=din1.read().data;
			ACC_DTYPE<32> exs=din1.read().data;
    		rb++;
			lhs_sum1[ra] =wsums;
			crf1[ra] =rfs;
			crx[ra] =exs;
			ra=rb;
            DWAIT(3);
    	}
    }

    DWAIT();
    if(rtake.read()){
    	for (int i = 0; i <rlength/4; i++){
			ACC_DTYPE<32> data=din1.read().data;
    		rb++;
			rhsdata1[ra] =data;  //FIFO
			ra=rb;
#ifndef __SYNTHESIS__
			weight_max_index=ra;
			DWAIT();
#endif
    	}
		for (int i = 0; i <llength; i++){
			ACC_DTYPE<32> isums=din1.read().data;
    		lb++;
			rhs_sum1[la] =isums;  //FIFO
			la=lb;
            DWAIT();
    	}
    }
    d_in1.write(0);
    while(read_inputs.read())wait();
  }
}
