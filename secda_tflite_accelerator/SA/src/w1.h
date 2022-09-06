#include "acc.h"

void ACCNAME::Worker1(){
	ACC_DTYPE<32> od[256];
	ACC_DTYPE<8> in[16][16];
	ACC_DTYPE<8> we[16][16];

	ACC_DTYPE<32> od1a;
	ACC_DTYPE<32> od2a;
	ACC_DTYPE<32> od3a;
	ACC_DTYPE<32> od4a;
	ACC_DTYPE<32> od5a;
	ACC_DTYPE<32> od6a;
	ACC_DTYPE<32> od7a;
	ACC_DTYPE<32> od8a;
	ACC_DTYPE<32> od9a;
	ACC_DTYPE<32> od10a;
	ACC_DTYPE<32> od11a;
	ACC_DTYPE<32> od12a;
	ACC_DTYPE<32> od13a;
	ACC_DTYPE<32> od14a;
	ACC_DTYPE<32> od15a;
	ACC_DTYPE<32> od16a;

	ACC_DTYPE<32> od1b;
	ACC_DTYPE<32> od2b;
	ACC_DTYPE<32> od3b;
	ACC_DTYPE<32> od4b;
	ACC_DTYPE<32> od5b;
	ACC_DTYPE<32> od6b;
	ACC_DTYPE<32> od7b;
	ACC_DTYPE<32> od8b;
	ACC_DTYPE<32> od9b;
	ACC_DTYPE<32> od10b;
	ACC_DTYPE<32> od11b;
	ACC_DTYPE<32> od12b;
	ACC_DTYPE<32> od13b;
	ACC_DTYPE<32> od14b;
	ACC_DTYPE<32> od15b;
	ACC_DTYPE<32> od16b;

	ACC_DTYPE<32> od1c;
	ACC_DTYPE<32> od2c;
	ACC_DTYPE<32> od3c;
	ACC_DTYPE<32> od4c;
	ACC_DTYPE<32> od5c;
	ACC_DTYPE<32> od6c;
	ACC_DTYPE<32> od7c;
	ACC_DTYPE<32> od8c;
	ACC_DTYPE<32> od9c;
	ACC_DTYPE<32> od10c;
	ACC_DTYPE<32> od11c;
	ACC_DTYPE<32> od12c;
	ACC_DTYPE<32> od13c;
	ACC_DTYPE<32> od14c;
	ACC_DTYPE<32> od15c;
	ACC_DTYPE<32> od16c;

	ACC_DTYPE<32> od1d;
	ACC_DTYPE<32> od2d;
	ACC_DTYPE<32> od3d;
	ACC_DTYPE<32> od4d;
	ACC_DTYPE<32> od5d;
	ACC_DTYPE<32> od6d;
	ACC_DTYPE<32> od7d;
	ACC_DTYPE<32> od8d;
	ACC_DTYPE<32> od9d;
	ACC_DTYPE<32> od10d;
	ACC_DTYPE<32> od11d;
	ACC_DTYPE<32> od12d;
	ACC_DTYPE<32> od13d;
	ACC_DTYPE<32> od14d;
	ACC_DTYPE<32> od15d;
	ACC_DTYPE<32> od16d;

	ACC_DTYPE<32> od1e;
	ACC_DTYPE<32> od2e;
	ACC_DTYPE<32> od3e;
	ACC_DTYPE<32> od4e;
	ACC_DTYPE<32> od5e;
	ACC_DTYPE<32> od6e;
	ACC_DTYPE<32> od7e;
	ACC_DTYPE<32> od8e;
	ACC_DTYPE<32> od9e;
	ACC_DTYPE<32> od10e;
	ACC_DTYPE<32> od11e;
	ACC_DTYPE<32> od12e;
	ACC_DTYPE<32> od13e;
	ACC_DTYPE<32> od14e;
	ACC_DTYPE<32> od15e;
	ACC_DTYPE<32> od16e;

	ACC_DTYPE<32> od1f;
	ACC_DTYPE<32> od2f;
	ACC_DTYPE<32> od3f;
	ACC_DTYPE<32> od4f;
	ACC_DTYPE<32> od5f;
	ACC_DTYPE<32> od6f;
	ACC_DTYPE<32> od7f;
	ACC_DTYPE<32> od8f;
	ACC_DTYPE<32> od9f;
	ACC_DTYPE<32> od10f;
	ACC_DTYPE<32> od11f;
	ACC_DTYPE<32> od12f;
	ACC_DTYPE<32> od13f;
	ACC_DTYPE<32> od14f;
	ACC_DTYPE<32> od15f;
	ACC_DTYPE<32> od16f;

	ACC_DTYPE<32> od1g;
	ACC_DTYPE<32> od2g;
	ACC_DTYPE<32> od3g;
	ACC_DTYPE<32> od4g;
	ACC_DTYPE<32> od5g;
	ACC_DTYPE<32> od6g;
	ACC_DTYPE<32> od7g;
	ACC_DTYPE<32> od8g;
	ACC_DTYPE<32> od9g;
	ACC_DTYPE<32> od10g;
	ACC_DTYPE<32> od11g;
	ACC_DTYPE<32> od12g;
	ACC_DTYPE<32> od13g;
	ACC_DTYPE<32> od14g;
	ACC_DTYPE<32> od15g;
	ACC_DTYPE<32> od16g;


#pragma HLS array_partition variable=od complete dim=0
#pragma HLS array_partition variable=odi complete dim=0
#pragma HLS array_partition variable=in complete dim=0
#pragma HLS array_partition variable=we complete dim=0

#pragma HLS RESOURCE variable=od1a core=Mul
#pragma HLS RESOURCE variable=od2a core=Mul
#pragma HLS RESOURCE variable=od3a core=Mul
#pragma HLS RESOURCE variable=od4a core=Mul
#pragma HLS RESOURCE variable=od5a core=Mul
#pragma HLS RESOURCE variable=od6a core=Mul
#pragma HLS RESOURCE variable=od7a core=Mul
#pragma HLS RESOURCE variable=od8a core=Mul
#pragma HLS RESOURCE variable=od9a core=Mul
#pragma HLS RESOURCE variable=od10a core=Mul
#pragma HLS RESOURCE variable=od11a core=Mul
#pragma HLS RESOURCE variable=od12a core=Mul
#pragma HLS RESOURCE variable=od13a core=Mul
#pragma HLS RESOURCE variable=od14a core=Mul
#pragma HLS RESOURCE variable=od15a core=Mul
#pragma HLS RESOURCE variable=od16a core=Mul

#pragma HLS RESOURCE variable=od1b core=Mul
#pragma HLS RESOURCE variable=od2b core=Mul
#pragma HLS RESOURCE variable=od3b core=Mul
#pragma HLS RESOURCE variable=od4b core=Mul
#pragma HLS RESOURCE variable=od5b core=Mul
#pragma HLS RESOURCE variable=od6b core=Mul
#pragma HLS RESOURCE variable=od7b core=Mul
#pragma HLS RESOURCE variable=od8b core=Mul
#pragma HLS RESOURCE variable=od9b core=Mul
#pragma HLS RESOURCE variable=od10b core=Mul
#pragma HLS RESOURCE variable=od11b core=Mul
#pragma HLS RESOURCE variable=od12b core=Mul
#pragma HLS RESOURCE variable=od13b core=Mul
#pragma HLS RESOURCE variable=od14b core=Mul
#pragma HLS RESOURCE variable=od15b core=Mul
#pragma HLS RESOURCE variable=od16b core=Mul


#pragma HLS RESOURCE variable=od1c core=Mul
#pragma HLS RESOURCE variable=od2c core=Mul
#pragma HLS RESOURCE variable=od3c core=Mul
#pragma HLS RESOURCE variable=od4c core=Mul
#pragma HLS RESOURCE variable=od5c core=Mul
#pragma HLS RESOURCE variable=od6c core=Mul
#pragma HLS RESOURCE variable=od7c core=Mul
#pragma HLS RESOURCE variable=od8c core=Mul
#pragma HLS RESOURCE variable=od9c core=Mul
#pragma HLS RESOURCE variable=od10c core=Mul
#pragma HLS RESOURCE variable=od11c core=Mul
#pragma HLS RESOURCE variable=od12c core=Mul
#pragma HLS RESOURCE variable=od13c core=Mul
#pragma HLS RESOURCE variable=od14c core=Mul
#pragma HLS RESOURCE variable=od15c core=Mul
#pragma HLS RESOURCE variable=od16c core=Mul

#pragma HLS RESOURCE variable=od1d core=Mul
#pragma HLS RESOURCE variable=od2d core=Mul
#pragma HLS RESOURCE variable=od3d core=Mul
#pragma HLS RESOURCE variable=od4d core=Mul
#pragma HLS RESOURCE variable=od5d core=Mul
#pragma HLS RESOURCE variable=od6d core=Mul
#pragma HLS RESOURCE variable=od7d core=Mul
#pragma HLS RESOURCE variable=od8d core=Mul
#pragma HLS RESOURCE variable=od9d core=Mul
#pragma HLS RESOURCE variable=od10d core=Mul
#pragma HLS RESOURCE variable=od11d core=Mul
#pragma HLS RESOURCE variable=od12d core=Mul
#pragma HLS RESOURCE variable=od13d core=Mul
#pragma HLS RESOURCE variable=od14d core=Mul
#pragma HLS RESOURCE variable=od15d core=Mul
#pragma HLS RESOURCE variable=od16d core=Mul

#pragma HLS RESOURCE variable=od1e core=Mul
#pragma HLS RESOURCE variable=od2e core=Mul
#pragma HLS RESOURCE variable=od3e core=Mul
#pragma HLS RESOURCE variable=od4e core=Mul
#pragma HLS RESOURCE variable=od5e core=Mul
#pragma HLS RESOURCE variable=od6e core=Mul
#pragma HLS RESOURCE variable=od7e core=Mul
#pragma HLS RESOURCE variable=od8e core=Mul
#pragma HLS RESOURCE variable=od9e core=Mul
#pragma HLS RESOURCE variable=od10e core=Mul
#pragma HLS RESOURCE variable=od11e core=Mul
#pragma HLS RESOURCE variable=od12e core=Mul
#pragma HLS RESOURCE variable=od13e core=Mul
#pragma HLS RESOURCE variable=od14e core=Mul
#pragma HLS RESOURCE variable=od15e core=Mul
#pragma HLS RESOURCE variable=od16e core=Mul


#pragma HLS RESOURCE variable=od1f core=Mul
#pragma HLS RESOURCE variable=od2f core=Mul
#pragma HLS RESOURCE variable=od3f core=Mul
#pragma HLS RESOURCE variable=od4f core=Mul
#pragma HLS RESOURCE variable=od5f core=Mul
#pragma HLS RESOURCE variable=od6f core=Mul
#pragma HLS RESOURCE variable=od7f core=Mul
#pragma HLS RESOURCE variable=od8f core=Mul
#pragma HLS RESOURCE variable=od9f core=Mul
#pragma HLS RESOURCE variable=od10f core=Mul
#pragma HLS RESOURCE variable=od11f core=Mul
#pragma HLS RESOURCE variable=od12f core=Mul
#pragma HLS RESOURCE variable=od13f core=Mul
#pragma HLS RESOURCE variable=od14f core=Mul
#pragma HLS RESOURCE variable=od15f core=Mul
#pragma HLS RESOURCE variable=od16f core=Mul


//#pragma HLS RESOURCE variable=od1g core=Mul
//#pragma HLS RESOURCE variable=od2g core=Mul
//#pragma HLS RESOURCE variable=od3g core=Mul
//#pragma HLS RESOURCE variable=od4g core=Mul
//#pragma HLS RESOURCE variable=od5g core=Mul
//#pragma HLS RESOURCE variable=od6g core=Mul
//#pragma HLS RESOURCE variable=od7g core=Mul
//#pragma HLS RESOURCE variable=od8g core=Mul
//#pragma HLS RESOURCE variable=od9g core=Mul
//#pragma HLS RESOURCE variable=od10g core=Mul
//#pragma HLS RESOURCE variable=od11g core=Mul
//#pragma HLS RESOURCE variable=od12g core=Mul
//#pragma HLS RESOURCE variable=od13g core=Mul
//#pragma HLS RESOURCE variable=od14g core=Mul
//#pragma HLS RESOURCE variable=od15g core=Mul
//#pragma HLS RESOURCE variable=od16g core=Mul





	w1S.write(0);
	wait();
	while(1){

		while(gemm_unit_1_ready.read()) {
			w1S.write(10);
			DWAIT();
		}

		int d = depth+30;
		w1S.write(1);
		wait();

		for(int i=0;i<256;i++){
#pragma HLS unroll
			od[i]=0;
		}


		w1S.write(3);
		DWAIT();
		for(int i=0;i<d;i++){
#pragma HLS pipeline II=1

			for(int i=15;i>0;i--){
#pragma HLS unroll
				for(int j=0;j<16;j++){
#pragma HLS unroll
					in[j][i] = in[j][i-1];
					we[i][j] = we[i-1][j];

				}
			}

			in[0][0] = sIs1.read();
			in[1][0] = sIs2.read();
			in[2][0] = sIs3.read();
			in[3][0] = sIs4.read();
			in[4][0] = sIs5.read();
			in[5][0] = sIs6.read();
			in[6][0] = sIs7.read();
			in[7][0] = sIs8.read();
			in[8][0] = sIs9.read();
			in[9][0] = sIs10.read();
			in[10][0] = sIs11.read();
			in[11][0] = sIs12.read();
			in[12][0] = sIs13.read();
			in[13][0] = sIs14.read();
			in[14][0] = sIs15.read();
			in[15][0] = sIs16.read();


			we[0][0] = sWs1.read();
			we[0][1] = sWs2.read();
			we[0][2] = sWs3.read();
			we[0][3] = sWs4.read();
			we[0][4] = sWs5.read();
			we[0][5] = sWs6.read();
			we[0][6] = sWs7.read();
			we[0][7] = sWs8.read();
			we[0][8] = sWs9.read();
			we[0][9] = sWs10.read();
			we[0][10] = sWs11.read();
			we[0][11] = sWs12.read();
			we[0][12] = sWs13.read();
			we[0][13] = sWs14.read();
			we[0][14] = sWs15.read();
			we[0][15] = sWs16.read();


			for(int i=0;i<9;i++){
#pragma HLS unroll
				for(int j=0;j<16;j++){
#pragma HLS unroll
					od[(i*16) + j] += in[j][i] * we[j][i];
				}
			}

			od1a = in[0][10] * we[0][10];
			od2a = in[1][10] * we[1][10];
			od3a = in[2][10] * we[2][10];
			od4a = in[3][10] * we[3][10];
			od5a = in[4][10] * we[4][10];
			od6a = in[5][10] * we[5][10];
			od7a = in[6][10] * we[6][10];
			od8a = in[7][10] * we[7][10];
			od9a = in[8][10] * we[8][10];
			od10a = in[9][10] * we[9][10];
			od11a = in[10][10] * we[10][10];
			od12a = in[11][10] * we[11][10];
			od13a = in[12][10] * we[12][10];
			od14a = in[13][10] * we[13][10];
			od15a = in[14][10] * we[14][10];
			od16a = in[15][10] * we[15][10];

			od1b = in[0][11] * we[0][11];
			od2b = in[1][11] * we[1][11];
			od3b = in[2][11] * we[2][11];
			od4b = in[3][11] * we[3][11];
			od5b = in[4][11] * we[4][11];
			od6b = in[5][11] * we[5][11];
			od7b = in[6][11] * we[6][11];
			od8b = in[7][11] * we[7][11];
			od9b = in[8][11] * we[8][11];
			od10b = in[9][11] * we[9][11];
			od11b = in[10][11] * we[10][11];
			od12b = in[11][11] * we[11][11];
			od13b = in[12][11] * we[12][11];
			od14b = in[13][11] * we[13][11];
			od15b = in[14][11] * we[14][11];
			od16b = in[15][11] * we[15][11];

			od1c = in[0][12] * we[0][12];
			od2c = in[1][12] * we[1][12];
			od3c = in[2][12] * we[2][12];
			od4c = in[3][12] * we[3][12];
			od5c = in[4][12] * we[4][12];
			od6c = in[5][12] * we[5][12];
			od7c = in[6][12] * we[6][12];
			od8c = in[7][12] * we[7][12];
			od9c = in[8][12] * we[8][12];
			od10c = in[9][12] * we[9][12];
			od11c = in[10][12] * we[10][12];
			od12c = in[11][12] * we[11][12];
			od13c = in[12][12] * we[12][12];
			od14c = in[13][12] * we[13][12];
			od15c = in[14][12] * we[14][12];
			od16c = in[15][12] * we[15][12];

			od1d = in[0][13] * we[0][13];
			od2d = in[1][13] * we[1][13];
			od3d = in[2][13] * we[2][13];
			od4d = in[3][13] * we[3][13];
			od5d = in[4][13] * we[4][13];
			od6d = in[5][13] * we[5][13];
			od7d = in[6][13] * we[6][13];
			od8d = in[7][13] * we[7][13];
			od9d = in[8][13] * we[8][13];
			od10d = in[9][13] * we[9][13];
			od11d = in[10][13] * we[10][13];
			od12d = in[11][13] * we[11][13];
			od13d = in[12][13] * we[12][13];
			od14d = in[13][13] * we[13][13];
			od15d = in[14][13] * we[14][13];
			od16d = in[15][13] * we[15][13];

			od1e = in[0][14] * we[0][14];
			od2e = in[1][14] * we[1][14];
			od3e = in[2][14] * we[2][14];
			od4e = in[3][14] * we[3][14];
			od5e = in[4][14] * we[4][14];
			od6e = in[5][14] * we[5][14];
			od7e = in[6][14] * we[6][14];
			od8e = in[7][14] * we[7][14];
			od9e = in[8][14] * we[8][14];
			od10e = in[9][14] * we[9][14];
			od11e = in[10][14] * we[10][14];
			od12e = in[11][14] * we[11][14];
			od13e = in[12][14] * we[12][14];
			od14e = in[13][14] * we[13][14];
			od15e = in[14][14] * we[14][14];
			od16e = in[15][14] * we[15][14];

			od1f = in[0][15] * we[0][15];
			od2f = in[1][15] * we[1][15];
			od3f = in[2][15] * we[2][15];
			od4f = in[3][15] * we[3][15];
			od5f = in[4][15] * we[4][15];
			od6f = in[5][15] * we[5][15];
			od7f = in[6][15] * we[6][15];
			od8f = in[7][15] * we[7][15];
			od9f = in[8][15] * we[8][15];
			od10f = in[9][15] * we[9][15];
			od11f = in[10][15] * we[10][15];
			od12f = in[11][15] * we[11][15];
			od13f = in[12][15] * we[12][15];
			od14f = in[13][15] * we[13][15];
			od15f = in[14][15] * we[14][15];
			od16f = in[15][15] * we[15][15];

			od1g = in[0][9] * we[0][9];
			od2g = in[1][9] * we[1][9];
			od3g = in[2][9] * we[2][9];
			od4g = in[3][9] * we[3][9];
			od5g = in[4][9] * we[4][9];
			od6g = in[5][9] * we[5][9];
			od7g = in[6][9] * we[6][9];
			od8g = in[7][9] * we[7][9];
			od9g = in[8][9] * we[8][9];
			od10g = in[9][9] * we[9][9];
			od11g = in[10][9] * we[10][9];
			od12g = in[11][9] * we[11][9];
			od13g = in[12][9] * we[12][9];
			od14g = in[13][9] * we[13][9];
			od15g = in[14][9] * we[14][9];
			od16g = in[15][9] * we[15][9];

			od[(9*16) + 0] += od1g;
			od[(9*16) + 1] += od2g;
			od[(9*16) + 2] += od3g;
			od[(9*16) + 3] += od4g;
			od[(9*16) + 4] += od5g;
			od[(9*16) + 5] += od6g;
			od[(9*16) + 6] += od7g;
			od[(9*16) + 7] += od8g;
			od[(9*16) + 8] += od9g;
			od[(9*16) + 9] += od10g;
			od[(9*16) + 10] += od11g;
			od[(9*16) + 11] += od12g;
			od[(9*16) + 12] += od13g;
			od[(9*16) + 13] += od14g;
			od[(9*16) + 14] += od15g;
			od[(9*16) + 15] += od16g;

			od[(10*16) + 0] += od1a;
			od[(10*16) + 1] += od2a;
			od[(10*16) + 2] += od3a;
			od[(10*16) + 3] += od4a;
			od[(10*16) + 4] += od5a;
			od[(10*16) + 5] += od6a;
			od[(10*16) + 6] += od7a;
			od[(10*16) + 7] += od8a;
			od[(10*16) + 8] += od9a;
			od[(10*16) + 9] += od10a;
			od[(10*16) + 10] += od11a;
			od[(10*16) + 11] += od12a;
			od[(10*16) + 12] += od13a;
			od[(10*16) + 13] += od14a;
			od[(10*16) + 14] += od15a;
			od[(10*16) + 15] += od16a;

			od[(11*16) + 0] += od1b;
			od[(11*16) + 1] += od2b;
			od[(11*16) + 2] += od3b;
			od[(11*16) + 3] += od4b;
			od[(11*16) + 4] += od5b;
			od[(11*16) + 5] += od6b;
			od[(11*16) + 6] += od7b;
			od[(11*16) + 7] += od8b;
			od[(11*16) + 8] += od9b;
			od[(11*16) + 9] += od10b;
			od[(11*16) + 10] += od11b;
			od[(11*16) + 11] += od12b;
			od[(11*16) + 12] += od13b;
			od[(11*16) + 13] += od14b;
			od[(11*16) + 14] += od15b;
			od[(11*16) + 15] += od16b;

			od[(12*16) + 0] += od1c;
			od[(12*16) + 1] += od2c;
			od[(12*16) + 2] += od3c;
			od[(12*16) + 3] += od4c;
			od[(12*16) + 4] += od5c;
			od[(12*16) + 5] += od6c;
			od[(12*16) + 6] += od7c;
			od[(12*16) + 7] += od8c;
			od[(12*16) + 8] += od9c;
			od[(12*16) + 9] += od10c;
			od[(12*16) + 10] += od11c;
			od[(12*16) + 11] += od12c;
			od[(12*16) + 12] += od13c;
			od[(12*16) + 13] += od14c;
			od[(12*16) + 14] += od15c;
			od[(12*16) + 15] += od16c;

			od[(13*16) + 0] += od1d;
			od[(13*16) + 1] += od2d;
			od[(13*16) + 2] += od3d;
			od[(13*16) + 3] += od4d;
			od[(13*16) + 4] += od5d;
			od[(13*16) + 5] += od6d;
			od[(13*16) + 6] += od7d;
			od[(13*16) + 7] += od8d;
			od[(13*16) + 8] += od9d;
			od[(13*16) + 9] += od10d;
			od[(13*16) + 10] += od11d;
			od[(13*16) + 11] += od12d;
			od[(13*16) + 12] += od13d;
			od[(13*16) + 13] += od14d;
			od[(13*16) + 14] += od15d;
			od[(13*16) + 15] += od16d;

			od[(14*16) + 0] += od1e;
			od[(14*16) + 1] += od2e;
			od[(14*16) + 2] += od3e;
			od[(14*16) + 3] += od4e;
			od[(14*16) + 4] += od5e;
			od[(14*16) + 5] += od6e;
			od[(14*16) + 6] += od7e;
			od[(14*16) + 7] += od8e;
			od[(14*16) + 8] += od9e;
			od[(14*16) + 9] += od10e;
			od[(14*16) + 10] += od11e;
			od[(14*16) + 11] += od12e;
			od[(14*16) + 12] += od13e;
			od[(14*16) + 13] += od14e;
			od[(14*16) + 14] += od15e;
			od[(14*16) + 15] += od16e;

			od[(15*16) + 0] += od1f;
			od[(15*16) + 1] += od2f;
			od[(15*16) + 2] += od3f;
			od[(15*16) + 3] += od4f;
			od[(15*16) + 4] += od5f;
			od[(15*16) + 5] += od6f;
			od[(15*16) + 6] += od7f;
			od[(15*16) + 7] += od8f;
			od[(15*16) + 8] += od9f;
			od[(15*16) + 9] += od10f;
			od[(15*16) + 10] += od11f;
			od[(15*16) + 11] += od12f;
			od[(15*16) + 12] += od13f;
			od[(15*16) + 13] += od14f;
			od[(15*16) + 14] += od15f;
			od[(15*16) + 15] += od16f;
			DWAIT(3);
		}
		w1S.write(4);

		while(write1.read()){
			w1S.write(9);
			DWAIT();
		}

		for(int i=0;i<256;i++){
#pragma HLS unroll
			g1[i]=od[i];
		}

		wait();
		write1.write(1);
		w1S.write(5);
		gemm_unit_1_ready.write(1);
		wait();

#ifndef __SYNTHESIS__
		g1_out_count+=64;
#endif
	}
}
