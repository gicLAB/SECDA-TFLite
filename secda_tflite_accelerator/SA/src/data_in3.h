#include "acc.h"
void ACCNAME::Data_In3(){
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
			ACC_DTYPE<32> data=din3.read().data;
    		lb++;
            lhsdata3a[la] =data;  //FIFO
			la=lb;
            DWAIT();
    	}
        for (int i = 0; i <rlength; i++){
			ACC_DTYPE<32> wsums=din3.read().data;
			ACC_DTYPE<32> rfs=din3.read().data;
    		rb++;
			lhs_sum3[ra] =wsums;
			crf3[ra] =rfs;
			ra=rb;
            DWAIT(2);
    	}
    }

    DWAIT();
    if(rtake.read()){
    	for (int i = 0; i <rlength/4; i++){
			ACC_DTYPE<32> data=din3.read().data;
    		rb++;
			rhsdata3[ra] =data;  //FIFO
			ra=rb;
            DWAIT();
    	}
		for (int i = 0; i <llength; i++){
			ACC_DTYPE<32> isums=din3.read().data;
    		lb++;
			rhs_sum3[la] =isums;  //FIFO
			la=lb;
            DWAIT();
    	}
    }
    d_in3.write(0);
    while(read_inputs.read())wait();
  }
}
