#include "acc.h"

void ACCNAME::Worker1(){
//#pragma HLS RESOURCE variable=od2kt core=Mul
//#pragma HLS RESOURCE variable=od3kt core=Mul
//#pragma HLS RESOURCE variable=od4kt core=Mul
//#pragma HLS RESOURCE variable=od5kt core=Mul
//#pragma HLS RESOURCE variable=od6kt core=Mul
//#pragma HLS RESOURCE variable=od7kt core=Mul
#pragma HLS RESOURCE variable=od8kt core=Mul
#pragma HLS RESOURCE variable=od9kt core=Mul
#pragma HLS RESOURCE variable=od10kt core=Mul
#pragma HLS RESOURCE variable=od11kt core=Mul
#pragma HLS RESOURCE variable=od12kt core=Mul
#pragma HLS RESOURCE variable=od13kt core=Mul
#pragma HLS RESOURCE variable=od14kt core=Mul
#pragma HLS RESOURCE variable=od15kt core=Mul
#pragma HLS RESOURCE variable=od16kt core=Mul
#pragma HLS RESOURCE variable=od17kt core=Mul

//#pragma HLS RESOURCE variable=od2at core=Mul
//#pragma HLS RESOURCE variable=od3at core=Mul
//#pragma HLS RESOURCE variable=od4at core=Mul
//#pragma HLS RESOURCE variable=od5at core=Mul
//#pragma HLS RESOURCE variable=od6at core=Mul
//#pragma HLS RESOURCE variable=od7at core=Mul
#pragma HLS RESOURCE variable=od8at core=Mul
#pragma HLS RESOURCE variable=od9at core=Mul
#pragma HLS RESOURCE variable=od10at core=Mul
#pragma HLS RESOURCE variable=od11at core=Mul
#pragma HLS RESOURCE variable=od12at core=Mul
#pragma HLS RESOURCE variable=od13at core=Mul
#pragma HLS RESOURCE variable=od14at core=Mul
#pragma HLS RESOURCE variable=od15at core=Mul
#pragma HLS RESOURCE variable=od16at core=Mul
#pragma HLS RESOURCE variable=od17at core=Mul

//#pragma HLS RESOURCE variable=od2bt core=Mul
//#pragma HLS RESOURCE variable=od3bt core=Mul
//#pragma HLS RESOURCE variable=od4bt core=Mul
//#pragma HLS RESOURCE variable=od5bt core=Mul
//#pragma HLS RESOURCE variable=od6bt core=Mul
//#pragma HLS RESOURCE variable=od7bt core=Mul
#pragma HLS RESOURCE variable=od8bt core=Mul
#pragma HLS RESOURCE variable=od9bt core=Mul
#pragma HLS RESOURCE variable=od10bt core=Mul
#pragma HLS RESOURCE variable=od11bt core=Mul
#pragma HLS RESOURCE variable=od12bt core=Mul
#pragma HLS RESOURCE variable=od13bt core=Mul
#pragma HLS RESOURCE variable=od14bt core=Mul
#pragma HLS RESOURCE variable=od15bt core=Mul
#pragma HLS RESOURCE variable=od16bt core=Mul
#pragma HLS RESOURCE variable=od17bt core=Mul

//#pragma HLS RESOURCE variable=od2ct core=Mul
//#pragma HLS RESOURCE variable=od3ct core=Mul
//#pragma HLS RESOURCE variable=od4ct core=Mul
//#pragma HLS RESOURCE variable=od5ct core=Mul
//#pragma HLS RESOURCE variable=od6ct core=Mul
//#pragma HLS RESOURCE variable=od7ct core=Mul
#pragma HLS RESOURCE variable=od8ct core=Mul
#pragma HLS RESOURCE variable=od9ct core=Mul
#pragma HLS RESOURCE variable=od10ct core=Mul
#pragma HLS RESOURCE variable=od11ct core=Mul
#pragma HLS RESOURCE variable=od12ct core=Mul
#pragma HLS RESOURCE variable=od13ct core=Mul
#pragma HLS RESOURCE variable=od14ct core=Mul
#pragma HLS RESOURCE variable=od15ct core=Mul
#pragma HLS RESOURCE variable=od16ct core=Mul
#pragma HLS RESOURCE variable=od17ct core=Mul

	ACC_DTYPE<32> od0;
	ACC_DTYPE<32> od1;
	ACC_DTYPE<32> od2;
	ACC_DTYPE<32> od3;
	ACC_DTYPE<32> od4;
	ACC_DTYPE<32> od5;
	ACC_DTYPE<32> od6;
	ACC_DTYPE<32> od7;
	ACC_DTYPE<32> od8;
	ACC_DTYPE<32> od9;
	ACC_DTYPE<32> od10;
	ACC_DTYPE<32> od11;
	ACC_DTYPE<32> od12;
	ACC_DTYPE<32> od13;
	ACC_DTYPE<32> od14;
	ACC_DTYPE<32> od15;
	ACC_DTYPE<32> od16;
	ACC_DTYPE<32> od17;

	ACC_DTYPE<32> od2kt;
	ACC_DTYPE<32> od3kt;
	ACC_DTYPE<32> od4kt;
	ACC_DTYPE<32> od5kt;
	ACC_DTYPE<32> od6kt;
	ACC_DTYPE<32> od7kt;
	ACC_DTYPE<32> od8kt;
	ACC_DTYPE<32> od9kt;
	ACC_DTYPE<32> od10kt;
	ACC_DTYPE<32> od11kt;
	ACC_DTYPE<32> od12kt;
	ACC_DTYPE<32> od13kt;
	ACC_DTYPE<32> od14kt;
	ACC_DTYPE<32> od15kt;
	ACC_DTYPE<32> od16kt;
	ACC_DTYPE<32> od17kt;

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
	ACC_DTYPE<32> od17a;

	ACC_DTYPE<32> od2at;
	ACC_DTYPE<32> od3at;
	ACC_DTYPE<32> od4at;
	ACC_DTYPE<32> od5at;
	ACC_DTYPE<32> od6at;
	ACC_DTYPE<32> od7at;
	ACC_DTYPE<32> od8at;
	ACC_DTYPE<32> od9at;
	ACC_DTYPE<32> od10at;
	ACC_DTYPE<32> od11at;
	ACC_DTYPE<32> od12at;
	ACC_DTYPE<32> od13at;
	ACC_DTYPE<32> od14at;
	ACC_DTYPE<32> od15at;
	ACC_DTYPE<32> od16at;
	ACC_DTYPE<32> od17at;


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
	ACC_DTYPE<32> od17b;

	ACC_DTYPE<32> od2bt;
	ACC_DTYPE<32> od3bt;
	ACC_DTYPE<32> od4bt;
	ACC_DTYPE<32> od5bt;
	ACC_DTYPE<32> od6bt;
	ACC_DTYPE<32> od7bt;
	ACC_DTYPE<32> od8bt;
	ACC_DTYPE<32> od9bt;
	ACC_DTYPE<32> od10bt;
	ACC_DTYPE<32> od11bt;
	ACC_DTYPE<32> od12bt;
	ACC_DTYPE<32> od13bt;
	ACC_DTYPE<32> od14bt;
	ACC_DTYPE<32> od15bt;
	ACC_DTYPE<32> od16bt;
	ACC_DTYPE<32> od17bt;


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
	ACC_DTYPE<32> od17c;

	ACC_DTYPE<32> od2ct;
	ACC_DTYPE<32> od3ct;
	ACC_DTYPE<32> od4ct;
	ACC_DTYPE<32> od5ct;
	ACC_DTYPE<32> od6ct;
	ACC_DTYPE<32> od7ct;
	ACC_DTYPE<32> od8ct;
	ACC_DTYPE<32> od9ct;
	ACC_DTYPE<32> od10ct;
	ACC_DTYPE<32> od11ct;
	ACC_DTYPE<32> od12ct;
	ACC_DTYPE<32> od13ct;
	ACC_DTYPE<32> od14ct;
	ACC_DTYPE<32> od15ct;
	ACC_DTYPE<32> od16ct;
	ACC_DTYPE<32> od17ct;

	ACC_DTYPE<8> s0;
	ACC_DTYPE<8> s1;
	ACC_DTYPE<8> s2;
	ACC_DTYPE<8> s3;
	ACC_DTYPE<8> s4;
	ACC_DTYPE<8> s5;
	ACC_DTYPE<8> s6;
	ACC_DTYPE<8> s7;

	ACC_DTYPE<8> ss0;
	ACC_DTYPE<8> ss1;
	ACC_DTYPE<8> ss2;
	ACC_DTYPE<8> ss3;
	ACC_DTYPE<8> ss4;
	ACC_DTYPE<8> ss5;
	ACC_DTYPE<8> ss6;
	ACC_DTYPE<8> ss7;


	ACC_DTYPE<8> f0;
	ACC_DTYPE<8> f1;
	ACC_DTYPE<8> f2;
	ACC_DTYPE<8> f3;

	ACC_DTYPE<8> f4;
	ACC_DTYPE<8> f5;
	ACC_DTYPE<8> f6;
	ACC_DTYPE<8> f7;

	ACC_DTYPE<8> sf0;
	ACC_DTYPE<8> sf1;
	ACC_DTYPE<8> sf2;
	ACC_DTYPE<8> sf3;

	ACC_DTYPE<8> sf4;
	ACC_DTYPE<8> sf5;
	ACC_DTYPE<8> sf6;
	ACC_DTYPE<8> sf7;

	w1S.write(0);
	wait();
	while(1){

		while(gemm_unit_1_ready.read()) {
			w1S.write(10);
			DWAIT();
		}
		int d = (depth/4) + 1;
		int l_pointer=gemm_unit_1_l_pointer.read();
		gemm_unit_1_iwuse.write(1);
		gemm_unit_1_ready.write(1);
		w1S.write(1);
		wait();

		od2=0;
		od3=0;
		od4=0;
		od5=0;
		od6=0;
		od7=0;
		od8=0;
		od9=0;
		od10=0;
		od11=0;
		od12=0;
		od13=0;
		od14=0;
		od15=0;
		od16=0;
		od17=0;

		od2a=0;
		od3a=0;
		od4a=0;
		od5a=0;
		od6a=0;
		od7a=0;
		od8a=0;
		od9a=0;
		od10a=0;
		od11a=0;
		od12a=0;
		od13a=0;
		od14a=0;
		od15a=0;
		od16a=0;
		od17a=0;

		od2b=0;
		od3b=0;
		od4b=0;
		od5b=0;
		od6b=0;
		od7b=0;
		od8b=0;
		od9b=0;
		od10b=0;
		od11b=0;
		od12b=0;
		od13b=0;
		od14b=0;
		od15b=0;
		od16b=0;
		od17b=0;

		od2c=0;
		od3c=0;
		od4c=0;
		od5c=0;
		od6c=0;
		od7c=0;
		od8c=0;
		od9c=0;
		od10c=0;
		od11c=0;
		od12c=0;
		od13c=0;
		od14c=0;
		od15c=0;
		od16c=0;
		od17c=0;

		ACC_DTYPE<32> i1 = lhsdata1a[l_pointer];
		ACC_DTYPE<32> i2 = lhsdata2a[l_pointer];
		ACC_DTYPE<32> i3 = lhsdata3a[l_pointer];
		ACC_DTYPE<32> i4 = lhsdata4a[l_pointer];

		ACC_DTYPE<32> w1 = rhs1a_1[0];
		ACC_DTYPE<32> w2 = rhs1b_1[0];
		ACC_DTYPE<32> w3 = rhs1c_1[0];
		ACC_DTYPE<32> w4 = rhs1d_1[0];

		f0 =i1.range(7,0);
		f1 =i2.range(7,0);
		f2 =i3.range(7,0);
		f3 =i4.range(7,0);
		f4 =i1.range(15,8);
		f5 =i2.range(15,8);
		f6 =i3.range(15,8);
		f7 =i4.range(15,8);
		sf0 =i1.range(23,16);
		sf1 =i2.range(23,16);
		sf2 =i3.range(23,16);
		sf3 =i4.range(23,16);
		sf4 =i1.range(31,24);
		sf5 =i2.range(31,24);
		sf6 =i3.range(31,24);
		sf7 =i4.range(31,24);

		s0 =w1.range(7,0);
		s1 =w2.range(7,0);
		s2 =w3.range(7,0);
		s3 =w4.range(7,0);
		s4 =w1.range(15,8);
		s5 =w2.range(15,8);
		s6 =w3.range(15,8);
		s7 =w4.range(15,8);
		ss0 =w1.range(23,16);
		ss1 =w2.range(23,16);
		ss2 =w3.range(23,16);
		ss3 =w4.range(23,16);
		ss4 =w1.range(31,24);
		ss5 =w2.range(31,24);
		ss6 =w3.range(31,24);
		ss7 =w4.range(31,24);
		w1S.write(2);

		DWAIT(5);
		for (int rin = 1; rin < d ; rin++){
			  od2kt= f0 * s0;
			  od3kt= f0 * s1;
			  od4kt= f0 * s2;
			  od5kt= f0 * s3;
			  od6kt= f1 * s0;
			  od7kt= f1 * s1;
			  od8kt= f1 * s2;
			  od9kt= f1 * s3;
			  od10kt= f2 * s0;
			  od11kt= f2 * s1;
			  od12kt= f2 * s2;
			  od13kt= f2 * s3;
			  od14kt= f3 * s0;
			  od15kt= f3 * s1;
			  od16kt= f3 * s2;
			  od17kt= f3 * s3;

			  od2at= f4 * s4;
			  od3at= f4 * s5;
			  od4at= f4 * s6;
			  od5at= f4 * s7;
			  od6at= f5 * s4;
			  od7at= f5 * s5;
			  od8at= f5 * s6;
			  od9at= f5 * s7;
			  od10at= f6 * s4;
			  od11at= f6 * s5;
			  od12at= f6 * s6;
			  od13at= f6 * s7;
			  od14at= f7 * s4;
			  od15at= f7 * s5;
			  od16at= f7 * s6;
			  od17at= f7 * s7;

			  od2bt= sf0 * ss0;
			  od3bt= sf0 * ss1;
			  od4bt= sf0 * ss2;
			  od5bt= sf0 * ss3;
			  od6bt= sf1 * ss0;
			  od7bt= sf1 * ss1;
			  od8bt= sf1 * ss2;
			  od9bt= sf1 * ss3;
			  od10bt= sf2 * ss0;
			  od11bt= sf2 * ss1;
			  od12bt= sf2 * ss2;
			  od13bt= sf2 * ss3;
			  od14bt= sf3 * ss0;
			  od15bt= sf3 * ss1;
			  od16bt= sf3 * ss2;
			  od17bt= sf3 * ss3;

			  od2ct= sf4 * ss4;
			  od3ct= sf4 * ss5;
			  od4ct= sf4 * ss6;
			  od5ct= sf4 * ss7;
			  od6ct= sf5 * ss4;
			  od7ct= sf5 * ss5;
			  od8ct= sf5 * ss6;
			  od9ct= sf5 * ss7;
			  od10ct= sf6 * ss4;
			  od11ct= sf6 * ss5;
			  od12ct= sf6 * ss6;
			  od13ct= sf6 * ss7;
			  od14ct= sf7 * ss4;
			  od15ct= sf7 * ss5;
			  od16ct= sf7 * ss6;
			  od17ct= sf7 * ss7;

			//    cout << f0 <<  " * " <<  s0 << " = " << od2kt << endl;
			//    cout << f4 <<  " * " <<  s4 << " = " << od2at << endl;
			//    cout << sf0 <<  " * " <<  ss0 << " = " << od2bt << endl;
			//    cout << sf4 <<  " * " <<  ss4 << " = " << od2ct << endl;

				cout << f3 <<  " * " <<  s0 << " = " << od14kt << endl;
				cout << f7 <<  " * " <<  s4 << " = " << od14at << endl;
				cout << sf3 <<  " * " <<  ss0 << " = " << od14bt << endl;
				cout << sf7 <<  " * " <<  ss4 << " = " << od14ct << endl;

			  od2+= od2kt;
			  od3+= od3kt;
			  od4+= od4kt;
			  od5+= od5kt;
			  od6+= od6kt;
			  od7+= od7kt;
			  od8+= od8kt;
			  od9+= od9kt;
			  od10+= od10kt;
			  od11+= od11kt;
			  od12+= od12kt;
			  od13+= od13kt;
			  od14+= od14kt;
			  od15+= od15kt;
			  od16+= od16kt;
			  od17+= od17kt;

			  od2a+= od2at;
			  od3a+= od3at;
			  od4a+= od4at;
			  od5a+= od5at;
			  od6a+= od6at;
			  od7a+= od7at;
			  od8a+= od8at;
			  od9a+= od9at;
			  od10a+= od10at;
			  od11a+= od11at;
			  od12a+= od12at;
			  od13a+= od13at;
			  od14a+= od14at;
			  od15a+= od15at;
			  od16a+= od16at;
			  od17a+= od17at;

			  od2b+= od2bt;
			  od3b+= od3bt;
			  od4b+= od4bt;
			  od5b+= od5bt;
			  od6b+= od6bt;
			  od7b+= od7bt;
			  od8b+= od8bt;
			  od9b+= od9bt;
			  od10b+= od10bt;
			  od11b+= od11bt;
			  od12b+= od12bt;
			  od13b+= od13bt;
			  od14b+= od14bt;
			  od15b+= od15bt;
			  od16b+= od16bt;
			  od17b+= od17bt;

			  od2c+= od2ct;
			  od3c+= od3ct;
			  od4c+= od4ct;
			  od5c+= od5ct;
			  od6c+= od6ct;
			  od7c+= od7ct;
			  od8c+= od8ct;
			  od9c+= od9ct;
			  od10c+= od10ct;
			  od11c+= od11ct;
			  od12c+= od12ct;
			  od13c+= od13ct;
			  od14c+= od14ct;
			  od15c+= od15ct;
			  od16c+= od16ct;
			  od17c+= od17ct;

			  i1 = lhsdata1a[rin+l_pointer];
			  i2 = lhsdata2a[rin+l_pointer];
			  i3 = lhsdata3a[rin+l_pointer];
			  i4 = lhsdata4a[rin+l_pointer];

			  w1 = rhs1a_1[rin];
			  w2 = rhs1b_1[rin];
			  w3 = rhs1c_1[rin];
			  w4 = rhs1d_1[rin];

			  f0 =i1.range(7,0);
			  f1 =i2.range(7,0);
			  f2 =i3.range(7,0);
			  f3 =i4.range(7,0);
			  f4 =i1.range(15,8);
			  f5 =i2.range(15,8);
			  f6 =i3.range(15,8);
			  f7 =i4.range(15,8);
			  sf0 =i1.range(23,16);
			  sf1 =i2.range(23,16);
			  sf2 =i3.range(23,16);
			  sf3 =i4.range(23,16);
			  sf4 =i1.range(31,24);
			  sf5 =i2.range(31,24);
			  sf6 =i3.range(31,24);
			  sf7 =i4.range(31,24);

			  s0 =w1.range(7,0);
			  s1 =w2.range(7,0);
			  s2 =w3.range(7,0);
			  s3 =w4.range(7,0);
			  s4 =w1.range(15,8);
			  s5 =w2.range(15,8);
			  s6 =w3.range(15,8);
			  s7 =w4.range(15,8);
			  ss0 =w1.range(23,16);
			  ss1 =w2.range(23,16);
			  ss2 =w3.range(23,16);
			  ss3 =w4.range(23,16);
			  ss4 =w1.range(31,24);
			  ss5 =w2.range(31,24);
			  ss6 =w3.range(31,24);
			  ss7 =w4.range(31,24);
			  w1S.write(3);

#ifndef __SYNTHESIS__
			  g1_macs+=64;
			  DWAIT(3);
#endif
		}
		w1S.write(4);
		gemm_unit_1_iwuse.write(0);
		od2+= od2a + od2b + od2c;
		od3+= od3a + od3b + od3c;
		od4+= od4a + od4b + od4c;
		od5+= od5a + od5b + od5c;
		od6+= od6a + od6b + od6c;
		od7+= od7a + od7b + od7c;
		od8+= od8a + od8b + od8c;
		od9+= od9a + od9b + od9c;
		od10+= od10a + od10b + od10c;
		od11+= od11a + od11b + od11c;
		od12+= od12a + od12b + od12c;
		od13+= od13a + od13b + od13c;
		od14+= od14a + od14b + od14c;
		od15+= od15a + od15b + od15c;
		od16+= od16a + od16b + od16c;
		od17+= od17a + od17b + od17c;


		while(write1.read()){
			w1S.write(9);
			DWAIT();
		}

#ifdef SYSC_ACC_DEBUG
		if (true){
		 cout << "===============" << endl;
		 cout << "G1" << endl;
		 cout << od2 << " , " << od6 << " , " << od10 << " , " << od14 << endl;
		 cout << od3 << " , " << od7 << " , " << od11 << " , " << od15 << endl;
		 cout << od4 << " , " << od8 << " , " << od12 << " , " << od16 << endl;
		 cout << od5 << " , " << od9 << " , " << od13 << " , " << od17 << endl;
		 cout << "===============" << endl;
		}
#endif

		g1.d2 = od2;
		g1.d3 = od3;
		g1.d4 = od4;
		g1.d5 = od5;
		g1.d6 = od6;
		g1.d7 = od7;
		g1.d8 = od8;
		g1.d9 = od9;
		g1.d10 = od10;
		g1.d11 = od11;
		g1.d12 = od12;
		g1.d13 = od13;
		g1.d14 = od14;
		g1.d15 = od15;
		g1.d16 = od16;
		g1.d17 = od17;
		write1.write(1);
		w1S.write(5);
		wait();

#ifndef __SYNTHESIS__
		g1_out_count+=16;
		DWAIT(2);
#endif
	}
}
