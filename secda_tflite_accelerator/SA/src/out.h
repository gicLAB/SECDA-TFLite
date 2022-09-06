#include "acc.h"

int ACCNAME::SHR(int value ,int shift){
	return  value >> shift;
}


void ACCNAME::Output_Handler(){

bool ready = false;
bool resetted = true;
DATA last ={5000,1};
wait();
	while(1){
		while(out_check.read() && !ready && resetted){
			bool w1 = w1S.read()==10;
			bool wr1 = !write1.read();
			bool block_done = !schedule.read();
			ready= block_done && w1 && wr1;

			if(ready){
				dout1.write(last);
				dout2.write(last);
				dout3.write(last);
				dout4.write(last);
				out_check.write(0);
				resetted=false;
			}
			wait();
			DWAIT(4);
		}

		if(!out_check.read()){
			resetted=true;
			ready=false;
		}
		wait();
		DWAIT();
	}
}


