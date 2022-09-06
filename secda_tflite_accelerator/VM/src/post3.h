#include "acc.h"

void ACCNAME::Post3(){
ACC_DTYPE<32> pram[16];
ACC_DTYPE<32> pcrf[4];
ACC_DTYPE<8> pex[4];
ACC_DTYPE<64> pls[4];
ACC_DTYPE<32> prs[4];
ACC_DTYPE<32> msks[4];
ACC_DTYPE<32> sms[4];

wait();
	while(true){
		while(!write3.read())wait();
		int y1=WRQ3.read();
		int y2=WRQ3.read();
		int y3=WRQ3.read();
		int y4=WRQ3.read();

		int x1=WRQ3.read();
		int x2=WRQ3.read();
		int x3=WRQ3.read();
		int x4=WRQ3.read();

		int z1=WRQ3.read();
		int z2=WRQ3.read();
		int z3=WRQ3.read();
		int z4=WRQ3.read();

		ACC_DTYPE<32> ex=WRQ3.read();
		pex[0]=ex.range(7,0);
		pex[1]=ex.range(15,8);
		pex[2]=ex.range(23,16);
		pex[3]=ex.range(31,24);

		for(int i=0;i<4;i++){
			if(pex[i]>0){
				pls[i]=  (1 << pex[i]);
				prs[i]=0;
				msks[i]= 0;
				sms[i]= 0;
			}else{
				pls[i]= 1;
				prs[i]=-pex[i];
				msks[i]= (1 << -pex[i])-1;
				sms[i]= ((1 << -pex[i])-1) >> 1;
			}
		}

		pcrf[0]=z1;
		pcrf[1]=z2;
		pcrf[2]=z3;
		pcrf[3]=z4;

		int p1 = y1+x1;
		int p2 = y1+x2;
		int p3 = y1+x3;
		int p4 = y1+x4;

		int p5 = y2+x1;
		int p6 = y2+x2;
		int p7 = y2+x3;
		int p8 = y2+x4;

		int p9 = y3+x1;
		int p10 = y3+x2;
		int p11 = y3+x3;
		int p12 = y3+x4;

		int p13 = y4+x1;
		int p14 = y4+x2;
		int p15 = y4+x3;
		int p16 = y4+x4;

		pram[0]= (g3.d2+p1);
		pram[1]= (g3.d6+p2);
		pram[2]= (g3.d10+p3);
		pram[3]= (g3.d14+p4);

		pram[4]= (g3.d3+p5);
		pram[5]= (g3.d7+p6);
		pram[6]= (g3.d11+p7);
		pram[7]= (g3.d15+p8);

		pram[8]= (g3.d4+p9);
		pram[9]= (g3.d8+p10);
		pram[10]= (g3.d12+p11);
		pram[11]= (g3.d16+p12);

		pram[12]= (g3.d5+p13);
		pram[13]= (g3.d9+p14);
		pram[14]= (g3.d13+p15);
		pram[15]= (g3.d17+p16);

		wait();
		for(int i=0;i<16;i+=2){
#pragma HLS pipeline II=1

			int mi1 =  i%4;
			int mi2 =  (i+1)%4;
			int aa1 = pram[i];
			int aa2 = pram[i+1];

			sc_int<64> loff1 = pls[mi1];
			sc_int<64> loff2 = pls[mi2];
			sc_int<32> rs1 = prs[mi1];
			sc_int<32> rs2 = prs[mi2];
			sc_int<32> ms1 = msks[mi1];
			sc_int<32> ms2 = msks[mi2];
			sc_int<32> sm1 = sms[mi1];
			sc_int<32> sm2 = sms[mi2];
			sc_int<64> rf1 = pcrf[mi1];
			sc_int<64> rf2 = pcrf[mi2];
			sc_int<64> a1 = (aa1)*loff1;
			sc_int<64> a2 = (aa2)*loff2;

			if(a1>MAX)a1 = MAX;
			if(a2>MAX)a2 = MAX;

			if(a1<MIN)a1 = MIN;
			if(a2<MIN)a2 = MIN;

			sc_int<64> r_a1 = a1 * rf1;
			sc_int<64> r_a2 = a2 * rf2;

			sc_int<32> bf_a1;
			sc_int<32> bf_a2;

			bf_a1 = (r_a1+POS)/DIVMAX;
			bf_a2 = (r_a2+POS)/DIVMAX;

			if(r_a1<0)bf_a1 = (r_a1+NEG)/DIVMAX;
			if(r_a2<0)bf_a2 = (r_a2+NEG)/DIVMAX;

			sc_int<32> f_a1 =(bf_a1);
			sc_int<32> f_a2 =(bf_a2);

			f_a1 = SHR(f_a1,rs1);
			f_a2 = SHR(f_a2,rs2);

			sc_int<32> rf_a1 = bf_a1 & ms1;
			sc_int<32> rf_a2 = bf_a2 & ms2;

			sc_int<32> lf_a1 = (bf_a1 < 0) & 1;
			sc_int<32> lf_a2 = (bf_a2 < 0) & 1;

			sc_int<32> tf_a1 = sm1 + lf_a1;
			sc_int<32> tf_a2 = sm2 + lf_a2;

			sc_int<32> af_a1 = ((rf_a1 > tf_a1) & 1) + ra;
			sc_int<32> af_a2 = ((rf_a2 > tf_a2) & 1) + ra;

			f_a1+= af_a1;
			f_a2+= af_a2;

			if(f_a1>MAX8)f_a1 = MAX8;
			else if (f_a1<MIN8) f_a1 = MIN8;
			if(f_a2>MAX8)f_a2 = MAX8;
			else if (f_a2<MIN8) f_a2 = MIN8;

#ifndef __SYNTHESIS__
			int32_t kaa1 = pram[i];
			int32_t kaa2 = pram[i+1];
			int64_t krf1 = pcrf[i];
			int64_t krf2 = pcrf[i+1];
			int64_t ka1 = (kaa1)*loff1;
			int64_t ka2 = (kaa2)*loff2;
			if(ka1>MAX)ka1 = MAX;
			if(ka2>MAX)ka2 = MAX;
			if(ka1<MIN)ka1 = MIN;
			if(ka2<MIN)ka2 = MIN;
			int64_t kr_a1 = ka1 * krf1;
			int64_t kr_a2 = ka2 * krf2;
			int32_t kbf_a1;
			int32_t kbf_a2;
			kbf_a1 = (kr_a1+POS)/DIVMAX;
			kbf_a2 = (kr_a2+POS)/DIVMAX;
			if(kr_a1<0)kbf_a1 = (kr_a1+NEG)/DIVMAX;
			if(kr_a2<0)kbf_a2 = (kr_a2+NEG)/DIVMAX;
			int32_t kf_a1 =(kbf_a1);
			int32_t kf_a2 =(kbf_a2);
			kf_a1 = SHR(kf_a1,rs1);
			kf_a2 = SHR(kf_a2,rs2);
			int32_t krf_a1 = kbf_a1 & ms1;
			int32_t krf_a2 = kbf_a2 & ms2;
			int32_t klf_a1 = (kbf_a1 < 0) & 1;
			int32_t klf_a2 = (kbf_a2 < 0) & 1;
			int32_t ktf_a1 = sm1 + klf_a1;
			int32_t ktf_a2 = sm2 + klf_a2;
			int32_t kaf_a1 = ((krf_a1 > ktf_a1) & 1) + ra;
			int32_t kaf_a2 = ((krf_a2 > ktf_a2) & 1) + ra;
			kf_a1+= kaf_a1;
			kf_a2+= kaf_a2;
			if(kf_a1>MAX8)kf_a1 = MAX8;
			else if (kf_a1<MIN8) kf_a1 = MIN8;
			if(kf_a2>MAX8)kf_a2 = MAX8;
			else if (kf_a2<MIN8) kf_a2 = MIN8;
			int wait_here = 4;
#endif
			pram[i] = f_a1;
			pram[i+1] = f_a2;
		}
		wait();

#ifdef SYSC_ACC_DEBUG2
		if (print_po){
		cout << "===============" << endl;
		cout << "P3" << endl;
		cout << pram[0] << " , " << pram[1] << " , " << pram[2] << " , " << pram[3] << endl;
		cout << pram[4] << " , " << pram[5] << " , " << pram[6] << " , " << pram[7] << endl;
		cout << pram[8] << " , " << pram[9] << " , " << pram[10] << " , " << pram[11] << endl;
		cout << pram[12] << " , " << pram[13] << " , " << pram[14] << " , " << pram[15] << endl;
		cout << "===============" << endl;
		}
#endif

		r3.d2=pram[0];
		r3.d6=pram[1];
		r3.d10=pram[2];
		r3.d14=pram[3];
		r3.d3=pram[4];
		r3.d7=pram[5];
		r3.d11=pram[6];
		r3.d15=pram[7];
		r3.d4=pram[8];
		r3.d8=pram[9];
		r3.d12=pram[10];
		r3.d16=pram[11];
		r3.d5=pram[12];
		r3.d9=pram[13];
		r3.d13=pram[14];
		r3.d17=pram[15];

		DWAIT(65);
		arrange3.write(1);wait();
		while(arrange3.read())wait();
		write3.write(0);
		wait();
	}
}


void ACCNAME::Arranger3(){
DATA d;
d.tlast=false;
wait();
	while(true){
		while(!arrange1.read())wait();
		d.data.range(7,0) = r1.d4.range(7,0);
		d.data.range(15,8) = r1.d8.range(7,0);
		d.data.range(23,16) = r1.d12.range(7,0);
		d.data.range(31,24) = r1.d16.range(7,0);
		dout3.write(d);
		DWAIT();
		write1_3.write(0);

		while(!arrange2.read())wait();
		d.data.range(7,0) = r2.d4.range(7,0);
		d.data.range(15,8) = r2.d8.range(7,0);
		d.data.range(23,16) = r2.d12.range(7,0);
		d.data.range(31,24) = r2.d16.range(7,0);
		dout3.write(d);
		DWAIT();
		write2_3.write(0);

		while(!arrange3.read())wait();
		d.data.range(7,0) = r3.d4.range(7,0);
		d.data.range(15,8) = r3.d8.range(7,0);
		d.data.range(23,16) = r3.d12.range(7,0);
		d.data.range(31,24) = r3.d16.range(7,0);
		dout3.write(d);
		DWAIT();
		write3_3.write(0);

		while(!arrange4.read())wait();
		d.data.range(7,0) = r4.d4.range(7,0);
		d.data.range(15,8) = r4.d8.range(7,0);
		d.data.range(23,16) = r4.d12.range(7,0);
		d.data.range(31,24) = r4.d16.range(7,0);
		dout3.write(d);
		DWAIT();
		write4_3.write(0);
	}
}
