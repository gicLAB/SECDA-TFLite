#include "acc.h"


void ACCNAME::WSync1(){
wait();
	while(true){
		while(write1_1.read() || write1_2.read() || write1_3.read() || write1_4.read()){

			bool w1 = write1_1.read();
			bool w2 = write1_2.read();
			bool w3 = write1_3.read();
			bool w4 = write1_4.read();

			wait();
		}
		write1_1.write(1);
		write1_2.write(1);
		write1_3.write(1);
		write1_4.write(1);
		arrange1.write(0);
		DWAIT();
	}
}


void ACCNAME::WSync2(){
wait();
	while(true){
		while(write2_1.read() || write2_2.read() || write2_3.read() || write2_4.read())wait();
		write2_1.write(1);
		write2_2.write(1);
		write2_3.write(1);
		write2_4.write(1);
		arrange2.write(0);
		DWAIT();
	}
}



void ACCNAME::WSync3(){
wait();
	while(true){
		while(write3_1.read() || write3_2.read() || write3_3.read() || write3_4.read())wait();
		write3_1.write(1);
		write3_2.write(1);
		write3_3.write(1);
		write3_4.write(1);
		arrange3.write(0);
		DWAIT();
	}
}




void ACCNAME::WSync4(){
wait();
	while(true){
		while(write4_1.read() || write4_2.read() || write4_3.read() || write4_4.read())wait();
		write4_1.write(1);
		write4_2.write(1);
		write4_3.write(1);
		write4_4.write(1);
		arrange4.write(0);
		DWAIT();
	}
}
