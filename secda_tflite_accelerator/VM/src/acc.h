


#ifndef ACC_H
#define ACC_H

#include <systemc.h>

// #define __SYNTHESIS__
#define SYSC_ACC_DEBUG
//#define SYSC_ACC_DEBUG2


#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif




#define ACCNAME VM_INT8_V1_0
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int



#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648

#define MAX8 127
#define MIN8 -128



typedef struct _DATA{
  ACC_DTYPE<32> data;
  bool        tlast;

  inline friend ostream& operator << (ostream& os, const _DATA &v){
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;

typedef struct _HDATA{
  bool rhs_take;
  bool lhs_take;
  sc_uint<8>  rhs_count;
  sc_uint<8>  lhs_count;
  sc_uint<16> dst_addr;

} HDATA;




typedef struct _ADATA{
  ACC_DTYPE<32> d2;
  ACC_DTYPE<32> d3;
  ACC_DTYPE<32> d4;
  ACC_DTYPE<32> d5;
  ACC_DTYPE<32> d6;
  ACC_DTYPE<32> d7;
  ACC_DTYPE<32> d8;
  ACC_DTYPE<32> d9;
  ACC_DTYPE<32> d10;
  ACC_DTYPE<32> d11;
  ACC_DTYPE<32> d12;
  ACC_DTYPE<32> d13;
  ACC_DTYPE<32> d14;
  ACC_DTYPE<32> d15;
  ACC_DTYPE<32> d16;
  ACC_DTYPE<32> d17;

  inline friend ostream& operator << (ostream& os, const _ADATA &v){
    return os;
  }
} ADATA;


SC_MODULE(ACCNAME) {

	//debug vars
	bool print_po = false;
	bool print_wo = false;



	int simplecounter=0;
	// int rows=0;
	ACC_DTYPE<14> depth;
	sc_in<bool> clock;
	sc_in <bool>  reset;


	sc_fifo_in<DATA> din1;
	sc_fifo_in<DATA> din2;
	sc_fifo_in<DATA> din3;
	sc_fifo_in<DATA> din4;

	sc_fifo_out<DATA> dout1;
	sc_fifo_out<DATA> dout2;
	sc_fifo_out<DATA> dout3;
	sc_fifo_out<DATA> dout4;


	sc_signal<bool> read_inputs;
	sc_signal<bool> rtake;
	sc_signal<bool> ltake;
	sc_signal<int> llen;
	sc_signal<int> rlen;
    sc_signal<int> lhs_block_max;
    sc_signal<int> rhs_block_max;

#ifndef __SYNTHESIS__
	sc_signal<bool,SC_MANY_WRITERS> d_in1;
	sc_signal<bool,SC_MANY_WRITERS> d_in2;
	sc_signal<bool,SC_MANY_WRITERS> d_in3;
	sc_signal<bool,SC_MANY_WRITERS> d_in4;
	sc_signal<bool,SC_MANY_WRITERS> schedule;
	sc_signal<bool,SC_MANY_WRITERS> out_check;
	sc_signal<bool,SC_MANY_WRITERS> gemm_unit_1_ready;
	sc_signal<bool,SC_MANY_WRITERS> gemm_unit_2_ready;
	sc_signal<bool,SC_MANY_WRITERS> gemm_unit_3_ready;
	sc_signal<bool,SC_MANY_WRITERS> gemm_unit_4_ready;
	sc_signal<bool,SC_MANY_WRITERS> write1;
	sc_signal<bool,SC_MANY_WRITERS> write2;
	sc_signal<bool,SC_MANY_WRITERS> write3;
	sc_signal<bool,SC_MANY_WRITERS> write4;

	sc_signal<bool,SC_MANY_WRITERS> arrange1;
	sc_signal<bool,SC_MANY_WRITERS> arrange2;
	sc_signal<bool,SC_MANY_WRITERS> arrange3;
	sc_signal<bool,SC_MANY_WRITERS> arrange4;
	sc_signal<bool,SC_MANY_WRITERS> write1_1;
	sc_signal<bool,SC_MANY_WRITERS> write1_2;
	sc_signal<bool,SC_MANY_WRITERS> write1_3;
	sc_signal<bool,SC_MANY_WRITERS> write1_4;
	sc_signal<bool,SC_MANY_WRITERS> write2_1;
	sc_signal<bool,SC_MANY_WRITERS> write2_2;
	sc_signal<bool,SC_MANY_WRITERS> write2_3;
	sc_signal<bool,SC_MANY_WRITERS> write2_4;
	sc_signal<bool,SC_MANY_WRITERS> write3_1;
	sc_signal<bool,SC_MANY_WRITERS> write3_2;
	sc_signal<bool,SC_MANY_WRITERS> write3_3;
	sc_signal<bool,SC_MANY_WRITERS> write3_4;
	sc_signal<bool,SC_MANY_WRITERS> write4_1;
	sc_signal<bool,SC_MANY_WRITERS> write4_2;
	sc_signal<bool,SC_MANY_WRITERS> write4_3;
	sc_signal<bool,SC_MANY_WRITERS> write4_4;

#else
	sc_signal<bool> d_in1;
	sc_signal<bool> d_in2;
	sc_signal<bool> d_in3;
	sc_signal<bool> d_in4;
    sc_signal<bool> schedule;
    sc_signal<bool> out_check;
	sc_signal<bool> gemm_unit_1_ready;
	sc_signal<bool> gemm_unit_2_ready;
	sc_signal<bool> gemm_unit_3_ready;
	sc_signal<bool> gemm_unit_4_ready;
    sc_signal<bool> write1;
    sc_signal<bool> write2;
    sc_signal<bool> write3;
    sc_signal<bool> write4;

	sc_signal<bool> arrange1;
	sc_signal<bool> arrange2;
	sc_signal<bool> arrange3;
	sc_signal<bool> arrange4;
	sc_signal<bool> write1_1;
	sc_signal<bool> write1_2;
	sc_signal<bool> write1_3;
	sc_signal<bool> write1_4;
	sc_signal<bool> write2_1;
	sc_signal<bool> write2_2;
	sc_signal<bool> write2_3;
	sc_signal<bool> write2_4;
	sc_signal<bool> write3_1;
	sc_signal<bool> write3_2;
	sc_signal<bool> write3_3;
	sc_signal<bool> write3_4;
	sc_signal<bool> write4_1;
	sc_signal<bool> write4_2;
	sc_signal<bool> write4_3;
	sc_signal<bool> write4_4;

#endif


    sc_signal<int> gemm_unit_1_l_pointer;
    sc_signal<int> gemm_unit_2_l_pointer;
    sc_signal<int> gemm_unit_3_l_pointer;
    sc_signal<int> gemm_unit_4_l_pointer;

    sc_signal<bool> gemm_unit_1_iwuse;
    sc_signal<bool> gemm_unit_2_iwuse;
    sc_signal<bool> gemm_unit_3_iwuse;
    sc_signal<bool> gemm_unit_4_iwuse;

    ADATA g1;
    ADATA g2;
    ADATA g3;
    ADATA g4;

    ADATA r1;
    ADATA r2;
    ADATA r3;
    ADATA r4;

	//GEMM 1 Inputs
    ACC_DTYPE<32> lhsdata1a[2048];
	ACC_DTYPE<32> lhsdata2a[2048];
	ACC_DTYPE<32> lhsdata3a[2048];
	ACC_DTYPE<32> lhsdata4a[2048];

	//GEMM 2 Inputs
	ACC_DTYPE<32> lhsdata1b[2048];
	ACC_DTYPE<32> lhsdata2b[2048];
	ACC_DTYPE<32> lhsdata3b[2048];
	ACC_DTYPE<32> lhsdata4b[2048];

	//GEMM 3 Inputs
	ACC_DTYPE<32> lhsdata1c[2048];
	ACC_DTYPE<32> lhsdata2c[2048];
	ACC_DTYPE<32> lhsdata3c[2048];
	ACC_DTYPE<32> lhsdata4c[2048];

	//GEMM 4 Inputs
	ACC_DTYPE<32> lhsdata1d[2048];
	ACC_DTYPE<32> lhsdata2d[2048];
	ACC_DTYPE<32> lhsdata3d[2048];
	ACC_DTYPE<32> lhsdata4d[2048];


//	//GEMM 1 Inputs
//    ACC_DTYPE<32> lhsdata1a[4096];
//	ACC_DTYPE<32> lhsdata2a[4096];
//	ACC_DTYPE<32> lhsdata3a[4096];
//	ACC_DTYPE<32> lhsdata4a[4096];
//
//	//GEMM 2 Inputs
//	ACC_DTYPE<32> lhsdata1b[4096];
//	ACC_DTYPE<32> lhsdata2b[4096];
//	ACC_DTYPE<32> lhsdata3b[4096];
//	ACC_DTYPE<32> lhsdata4b[4096];
//
//	//GEMM 3 Inputs
//	ACC_DTYPE<32> lhsdata1c[4096];
//	ACC_DTYPE<32> lhsdata2c[4096];
//	ACC_DTYPE<32> lhsdata3c[4096];
//	ACC_DTYPE<32> lhsdata4c[4096];
//
//	//GEMM 4 Inputs
//	ACC_DTYPE<32> lhsdata1d[4096];
//	ACC_DTYPE<32> lhsdata2d[4096];
//	ACC_DTYPE<32> lhsdata3d[4096];
//	ACC_DTYPE<32> lhsdata4d[4096];


	//Global Weights
	ACC_DTYPE<32> rhsdata1[8192];
	ACC_DTYPE<32> rhsdata2[8192];
	ACC_DTYPE<32> rhsdata3[8192];
	ACC_DTYPE<32> rhsdata4[8192];

    //First Set (A)
	ACC_DTYPE<32> rhs1a_1[2048];
	ACC_DTYPE<32> rhs1b_1[2048];
	ACC_DTYPE<32> rhs1c_1[2048];
	ACC_DTYPE<32> rhs1d_1[2048];

	ACC_DTYPE<32> rhs2a_1[2048];
	ACC_DTYPE<32> rhs2b_1[2048];
	ACC_DTYPE<32> rhs2c_1[2048];
	ACC_DTYPE<32> rhs2d_1[2048];

	ACC_DTYPE<32> rhs3a_1[2048];
	ACC_DTYPE<32> rhs3b_1[2048];
	ACC_DTYPE<32> rhs3c_1[2048];
	ACC_DTYPE<32> rhs3d_1[2048];

	ACC_DTYPE<32> rhs4a_1[2048];
	ACC_DTYPE<32> rhs4b_1[2048];
	ACC_DTYPE<32> rhs4c_1[2048];
	ACC_DTYPE<32> rhs4d_1[2048];

//    //First Set (A)
//	ACC_DTYPE<32> rhs1a_1[1024];
//	ACC_DTYPE<32> rhs1b_1[1024];
//	ACC_DTYPE<32> rhs1c_1[1024];
//	ACC_DTYPE<32> rhs1d_1[1024];
//
//	ACC_DTYPE<32> rhs2a_1[1024];
//	ACC_DTYPE<32> rhs2b_1[1024];
//	ACC_DTYPE<32> rhs2c_1[1024];
//	ACC_DTYPE<32> rhs2d_1[1024];
//
//	ACC_DTYPE<32> rhs3a_1[1024];
//	ACC_DTYPE<32> rhs3b_1[1024];
//	ACC_DTYPE<32> rhs3c_1[1024];
//	ACC_DTYPE<32> rhs3d_1[1024];
//
//	ACC_DTYPE<32> rhs4a_1[1024];
//	ACC_DTYPE<32> rhs4b_1[1024];
//	ACC_DTYPE<32> rhs4c_1[1024];
//	ACC_DTYPE<32> rhs4d_1[1024];


	//new sums bram
	ACC_DTYPE<32> lhs_sum1[512];
	ACC_DTYPE<32> lhs_sum2[512];
	ACC_DTYPE<32> lhs_sum3[512];
	ACC_DTYPE<32> lhs_sum4[512];

	ACC_DTYPE<32> rhs_sum1[512];
	ACC_DTYPE<32> rhs_sum2[512];
	ACC_DTYPE<32> rhs_sum3[512];
	ACC_DTYPE<32> rhs_sum4[512];

	//crf & crx
	ACC_DTYPE<32> crf1[512];
	ACC_DTYPE<32> crf2[512];
	ACC_DTYPE<32> crf3[512];
	ACC_DTYPE<32> crf4[512];
	ACC_DTYPE<32> crx[512];

	int ra=0;

    sc_fifo<int> WRQ1;
    sc_fifo<int> WRQ2;
    sc_fifo<int> WRQ3;
    sc_fifo<int> WRQ4;

    sc_signal<int>  w1S;
    sc_signal<int>  w2S;
    sc_signal<int>  w3S;
    sc_signal<int>  w4S;



#ifndef __SYNTHESIS__
    int weight_max_index=0;
    int input_max_index=0;
    int local_weight_max_index=0;

    int g1_macs=0;
    int g2_macs=0;
    int g3_macs=0;
    int g4_macs=0;

    int g1_out_count=0;
    int g2_out_count=0;
    int g3_out_count=0;
    int g4_out_count=0;

#endif

	sc_out<int>  inS;
	sc_out<int>  read_cycle_count;
	sc_out<int> process_cycle_count;
	sc_out<int> gemm_1_idle;
	sc_out<int> gemm_2_idle;
	sc_out<int> gemm_3_idle;
	sc_out<int> gemm_4_idle;
	sc_out<int> gemm_1_write;
	sc_out<int> gemm_2_write;
	sc_out<int> gemm_3_write;
	sc_out<int> gemm_4_write;
	sc_out<int> gemm_1;
	sc_out<int> gemm_2;
	sc_out<int> gemm_3;
	sc_out<int> gemm_4;
	sc_out<int> wstall_1;
	sc_out<int> wstall_2;
	sc_out<int> wstall_3;
	sc_out<int> wstall_4;

	sc_out<int> rmax;
	sc_out<int> lmax;

	sc_out<int>  outS;
	sc_out<int>  w1SS;
	sc_out<int>  w2SS;
	sc_out<int>  w3SS;
	sc_out<int>  w4SS;
	sc_out<int>  schS;
	sc_out<int>  p1S;


	void Input_Handler();

	void Output_Handler();

	void Worker1();

	void Worker2();

	void Worker3();

	void Worker4();

	void Data_In1();

	void Data_In2();

	void Data_In3();

	void Data_In4();

	void Tracker();

	void Scheduler();

	void Post1();

	void Post2();

	void Post3();

	void Post4();

	void Arranger1();

	void Arranger2();

	void Arranger3();

	void Arranger4();

	void WSync1();

	void WSync2();

	void WSync3();

	void WSync4();

	void load_weights(int,int);

	void schedule_gemm_unit(int, int, int, int);

	int SHR(int,int);

	void overwrite_weights_check();

	void Read_Cycle_Counter();

	void Process_Cycle_Counter();

	void Writer_Cycle_Counter();

	SC_HAS_PROCESS(ACCNAME);

  // Parameters for the DUT
  ACCNAME(sc_module_name name_) :sc_module(name_),WRQ1(512), WRQ2(512), WRQ3(512), WRQ4(512){

    SC_CTHREAD(Input_Handler,clock.pos());
    reset_signal_is(reset,true);

    SC_CTHREAD(Worker1,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Worker2,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Worker3,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Worker4,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Output_Handler,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Data_In1,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Data_In2,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Data_In3,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Data_In4,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Scheduler,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Post1,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Post2,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Post3,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Post4,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(WSync1,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(WSync2,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(WSync3,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(WSync4,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Arranger1,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Arranger2,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Arranger3,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Arranger4,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Read_Cycle_Counter,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Process_Cycle_Counter,clock);
    reset_signal_is(reset,true);

    SC_CTHREAD(Writer_Cycle_Counter,clock);
    reset_signal_is(reset,true);

#pragma HLS RESOURCE variable=din1 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA1" port_map={{din1_0 TDATA} {din1_1 TLAST}}
#pragma HLS RESOURCE variable=din2 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA2" port_map={{din2_0 TDATA} {din2_1 TLAST}}
#pragma HLS RESOURCE variable=din3 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA3" port_map={{din3_0 TDATA} {din3_1 TLAST}}
#pragma HLS RESOURCE variable=din4 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA4" port_map={{din4_0 TDATA} {din4_1 TLAST}}
#pragma HLS RESOURCE variable=dout1 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA1" port_map={{dout1_0 TDATA} {dout1_1 TLAST}}
#pragma HLS RESOURCE variable=dout2 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA2" port_map={{dout2_0 TDATA} {dout2_1 TLAST}}
#pragma HLS RESOURCE variable=dout3 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA3" port_map={{dout3_0 TDATA} {dout3_1 TLAST}}
#pragma HLS RESOURCE variable=dout4 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA4" port_map={{dout4_0 TDATA} {dout4_1 TLAST}}


#pragma HLS RESET variable=reset
  }
};
#endif /* ACC_H */
