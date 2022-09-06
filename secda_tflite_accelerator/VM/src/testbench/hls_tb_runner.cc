

//#ifdef __RTL_SIMULATION__
//#include "MGemm_rtl_wrapper.h"
//#define MGemm MGemm_rtl_wrapper
//#else
//#include "acc.h"
//#endif

#include "testbench.h"

#include <iostream>
#include <fstream>
#include<stdio.h>
#include<stdlib.h>
using namespace std;

void load_start(GemmDriver *gd, string name){
	string input;
	ifstream wrin0;
	ifstream wrin1;
	ifstream wrin2;
	ifstream wrin3;
	wrin0.open ("sample_data/"+name+"_0.txt");
	wrin1.open ("sample_data/"+name+"_1.txt");
	wrin2.open ("sample_data/"+name+"_2.txt");
	wrin3.open ("sample_data/"+name+"_3.txt");


	gd->inl0=0;
	gd->inl1=0;
	gd->inl2=0;
	gd->inl3=0;
	while (getline (wrin0, input)) {
		int d = atoi(input.c_str());
		gd->in0[gd->inl0++] = d;
	}
	while (getline (wrin1, input)) {
		int d = atoi(input.c_str());
		gd->in1[gd->inl1++] = d;
	}
	while (getline (wrin2, input)) {
		int d = atoi(input.c_str());
		gd->in2[gd->inl2++] = d;
	}
	while (getline (wrin3, input)) {
		int d = atoi(input.c_str());
		gd->in3[gd->inl3++] = d;
	}
	wrin0.close();
	wrin1.close();
	wrin2.close();
	wrin3.close();
	sc_start();

}

void read_out(GemmDriver *gd, string name){
  int expected_out = (gd->r_max/4) * (gd->l_max/4);

  ofstream wrin0;
  ofstream wrin1;
  ofstream wrin2;
  ofstream wrin3;
  wrin0.open ("sample_data/"+name+"_outputs_0.txt");
  wrin1.open ("sample_data/"+name+"_outputs_1.txt");
  wrin2.open ("sample_data/"+name+"_outputs_2.txt");
  wrin3.open ("sample_data/"+name+"_outputs_3.txt");

  int8_t* bo0= reinterpret_cast<int8_t*> (gd->out0);
  int8_t* bo1= reinterpret_cast<int8_t*> (gd->out1);
  int8_t* bo2= reinterpret_cast<int8_t*> (gd->out2);
  int8_t* bo3= reinterpret_cast<int8_t*> (gd->out3);

  int bc0 =0;
  int bc1 =0;
  int bc2 =0;
  int bc3 =0;

  for(int i=0;i<expected_out+1;i++){
    wrin0 << (int) bo0[bc0++] << "\n";
    wrin0 << (int) bo0[bc0++] << "\n";
    wrin0 << (int) bo0[bc0++] << "\n";
    wrin0 << (int) bo0[bc0++] << "\n";

    wrin1 << (int) bo1[bc1++] << "\n";
    wrin1 << (int) bo1[bc1++] << "\n";
    wrin1 << (int) bo1[bc1++] << "\n";
    wrin1 << (int) bo1[bc1++] << "\n";

    wrin2 << (int) bo2[bc2++] << "\n";
    wrin2 << (int) bo2[bc2++] << "\n";
    wrin2 << (int) bo2[bc2++] << "\n";
    wrin2 << (int) bo2[bc2++] << "\n";

    wrin3 << (int) bo3[bc3++] << "\n";
    wrin3 << (int) bo3[bc3++] << "\n";
    wrin3 << (int) bo3[bc3++] << "\n";
    wrin3 << (int) bo3[bc3++] << "\n";
  }
  wrin0.close();
  wrin1.close();
  wrin2.close();
  wrin3.close();
}

int sc_main(int argc, char* argv[]) {

  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions( SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions( SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);
  sc_report_handler::set_actions( SC_ID_OBJECT_EXISTS_, SC_LOG);


  struct systemC_sigs scs(1);
  ACCNAME mg("DUT");
  GemmDriver gd("Driver");
  SysC_Assign<int>(&mg,&gd,&scs);


  cout << "INFO: Simulating " << endl;

  load_start(&gd,"resnet18_l0_w0");
  load_start(&gd,"resnet18_l0_w0_i0");
  read_out(&gd,"a");
  cout << "Sim: Done " << endl;


//  sc_start();

#ifndef __SYNTHESIS__
  int additional_cycles = gd.r_max*gd.l_max*2/4;

  float weight_global_usage = (float) mg.weight_max_index/ (float)8192 *100;
  float input_global_usage = (float) mg.input_max_index/ (float)4096 *100;
  float weight_local_usage = (float) (mg.local_weight_max_index+1)/ (float)512 *100;


  int total_batch_macs = mg.g1_macs+mg.g2_macs+mg.g3_macs+mg.g4_macs;
  int total_out_count = mg.g1_out_count+mg.g2_out_count+mg.g3_out_count+mg.g4_out_count;

  float macs_to_out_ratio = (float) total_batch_macs/ (float) total_out_count;

  float gemm_cycle_percentage = (float) gd.g1_gemm/ (float) gd.p_cycles  *100;
  float write_cycle_percentage = (float) gd.g1_write/ (float) gd.p_cycles  *100;
  float wstall_cycle_percentage = (float) gd.wstall1/ (float) gd.p_cycles  *100;
  float idle_cycle_percentage = (float) gd.g1_idle/ (float) gd.p_cycles  *100;



  ofstream otu00;
  otu00.open ("m2_l1_profile_sim.txt");
  otu00 << "RMAX : "<< gd.r_max << endl;
  otu00 << "LMAX : "<< gd.l_max << endl;
  otu00 << "Loading Cycles: "<< gd.loading << endl;
  otu00 << "Processing Cycles: "<< gd.p_cycles << endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "G1 Idle Cycles: "<< gd.g1_idle << endl;
  otu00 << "G2 Idle Cycles: "<< gd.g2_idle << endl;
  otu00 << "G3 Idle Cycles: "<< gd.g3_idle << endl;
  otu00 << "G4 Idle Cycles: "<< gd.g4_idle << endl;
  otu00 << "Idle Cycle: "<< idle_cycle_percentage << "%" <<  endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "G1 Write Cycles: "<< gd.g1_write << endl;
  otu00 << "G2 Write Cycles: "<< gd.g2_write << endl;
  otu00 << "G3 Write Cycles: "<< gd.g3_write << endl;
  otu00 << "G4 Write Cycles: "<< gd.g4_write << endl;
  otu00 << "Write Cycle: "<< write_cycle_percentage << "%" <<  endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "G1 WStall Cycles: "<< gd.wstall1 << endl;
  otu00 << "G2 WStall Cycles: "<< gd.wstall2 << endl;
  otu00 << "G3 WStall Cycles: "<< gd.wstall3 << endl;
  otu00 << "G4 WStall Cycles: "<< gd.wstall4 << endl;
  otu00 << "WStall Cycle: "<< wstall_cycle_percentage << "%" <<  endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "G1 GEMM Cycles: "<< gd.g1_gemm << endl;
  otu00 << "G2 GEMM Cycles: "<< gd.g2_gemm << endl;
  otu00 << "G3 GEMM Cycles: "<< gd.g3_gemm << endl;
  otu00 << "G4 GEMM Cycles: "<< gd.g4_gemm << endl;
  otu00 << "GEMM Cycle: "<< gemm_cycle_percentage << "%" <<  endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "Global Weight Buffer Usage: "<< weight_global_usage << "%" <<  endl;
  otu00 << "Global Input Buffer Usage: "<< input_global_usage << "%" <<  endl;
  otu00 << "Local Weight Buffer Usage: "<< weight_local_usage << "%" <<  endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "Total MAC count: "<< total_batch_macs << endl;
  otu00 << "G1 MAC count: "<< mg.g1_macs << endl;
  otu00 << "G2 MAC count: "<< mg.g2_macs << endl;
  otu00 << "G3 MAC count: "<< mg.g3_macs << endl;
  otu00 << "G4 MAC count: "<< mg.g4_macs << endl;
  otu00 << "==================================================" <<  endl;
  otu00 << "Total Output count: "<< total_out_count << endl;
  otu00 << "G1 Output count: "<< mg.g1_out_count << endl;
  otu00 << "G2 Output count: "<< mg.g2_out_count << endl;
  otu00 << "G3 Output count: "<< mg.g3_out_count << endl;
  otu00 << "G4 Output count: "<< mg.g4_out_count << endl;
  otu00 << "MACs per Output: "<< macs_to_out_ratio << endl;
  otu00 << "==================================================" <<  endl;



  cout << "RMAX : "<< gd.r_max << endl;
  cout << "LMAX : "<< gd.l_max << endl;

  cout << "Loading Cycles: "<< gd.loading << endl;
  cout << "Processing Cycles: "<< gd.p_cycles << endl;

  cout << "==================================================" <<  endl;

  cout << "G1 Idle Cycles: "<< gd.g1_idle << endl;
  cout << "G2 Idle Cycles: "<< gd.g2_idle << endl;
  cout << "G3 Idle Cycles: "<< gd.g3_idle << endl;
  cout << "G4 Idle Cycles: "<< gd.g4_idle << endl;
  cout << "Idle Cycle: "<< idle_cycle_percentage << "%" <<  endl;

  cout << "==================================================" <<  endl;

  cout << "G1 Write Cycles: "<< gd.g1_write << endl;
  cout << "G2 Write Cycles: "<< gd.g2_write << endl;
  cout << "G3 Write Cycles: "<< gd.g3_write << endl;
  cout << "G4 Write Cycles: "<< gd.g4_write << endl;
  cout << "Write Cycle: "<< write_cycle_percentage << "%" <<  endl;


  cout << "==================================================" <<  endl;

  cout << "G1 WStall Cycles: "<< gd.wstall1 << endl;
  cout << "G2 WStall Cycles: "<< gd.wstall2 << endl;
  cout << "G3 WStall Cycles: "<< gd.wstall3 << endl;
  cout << "G4 WStall Cycles: "<< gd.wstall4 << endl;
  cout << "WStall Cycle: "<< wstall_cycle_percentage << "%" <<  endl;


  cout << "==================================================" <<  endl;

  cout << "G1 GEMM Cycles: "<< gd.g1_gemm << endl;
  cout << "G2 GEMM Cycles: "<< gd.g2_gemm << endl;
  cout << "G3 GEMM Cycles: "<< gd.g3_gemm << endl;
  cout << "G4 GEMM Cycles: "<< gd.g4_gemm << endl;
  cout << "GEMM Cycle: "<< gemm_cycle_percentage << "%" <<  endl;

  cout << "==================================================" <<  endl;



  cout << "Global Weight Buffer Usage: "<< weight_global_usage << "%" <<  endl;
  cout << "Global Input Buffer Usage: "<< input_global_usage << "%" <<  endl;
  cout << "Local Weight Buffer Usage: "<< weight_local_usage << "%" <<  endl;

  cout << "==================================================" <<  endl;

  cout << "Total MAC count: "<< total_batch_macs << endl;
  cout << "G1 MAC count: "<< mg.g1_macs << endl;
  cout << "G2 MAC count: "<< mg.g2_macs << endl;
  cout << "G3 MAC count: "<< mg.g3_macs << endl;
  cout << "G4 MAC count: "<< mg.g4_macs << endl;

  cout << "==================================================" <<  endl;

  cout << "Total Output count: "<< total_out_count << endl;
  cout << "G1 Output count: "<< mg.g1_out_count << endl;
  cout << "G2 Output count: "<< mg.g2_out_count << endl;
  cout << "G3 Output count: "<< mg.g3_out_count << endl;
  cout << "G4 Output count: "<< mg.g4_out_count << endl;
  cout << "MACs per Output: "<< macs_to_out_ratio << endl;

  cout << "==================================================" <<  endl;
#endif


  return 0;
}

