#include "acc.h"
void ACCNAME::Data_In4(){
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
			ACC_DTYPE<32> data=din4.read().data;
    		lb++;
            lhsdata4a[la] =data;  //FIFO
			la=lb;
            DWAIT();
    	}
        for (int i = 0; i <rlength; i++){
			ACC_DTYPE<32> wsums=din4.read().data;
			ACC_DTYPE<32> rfs=din4.read().data;
    		rb++;
			lhs_sum4[ra] =wsums;
			crf4[ra] =rfs;
			ra=rb;
            DWAIT(2);
    	}
    }

    DWAIT();
    if(rtake.read()){
    	for (int i = 0; i <rlength/4; i++){
			ACC_DTYPE<32> data=din4.read().data;
    		rb++;
			rhsdata4[ra] =data;  //FIFO
			ra=rb;
            DWAIT();
    	}
		for (int i = 0; i <llength; i++){
			ACC_DTYPE<32> isums=din4.read().data;
    		lb++;
			rhs_sum4[la] =isums;  //FIFO
			la=lb;
            DWAIT();
    	}
    }
    d_in4.write(0);
    while(read_inputs.read())wait();
  }
}
