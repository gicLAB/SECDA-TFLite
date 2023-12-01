#include "vta.h"

#include <bitset>

void ACCNAME::vta_on_exit(){
  // Ensure that the tokens are empty
  sc_biguint<128> tmp_tok;
  int l2g_count = 0;
  int s2g_count = 0;
  int g2l_count = 0;
  int g2s_count = 0;
  while (l2g_dep_queue.nb_read(tmp_tok)) {
    l2g_count++;
  }
  while (s2g_dep_queue.nb_read(tmp_tok)) {
    s2g_count++;
  }
  while (g2l_dep_queue.nb_read(tmp_tok)) {
    g2l_count++;
  }
  while (g2s_dep_queue.nb_read(tmp_tok)) {
    g2s_count++;
  }
}

void ACCNAME::vta_main() {
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vtaS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=insn_count

#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=ins_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=uops_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=input_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=weight_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=bias_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=output_addr


// Instantiate memories
#ifndef __SYNTHESIS__
  bus_T* wgt_mem = new bus_T[VTA_WGT_BUFF_DEPTH * WGT_MAT_AXI_RATIO];
  bus_T* inp_mem = new bus_T[VTA_INP_BUFF_DEPTH * INP_MAT_AXI_RATIO];
  bus_T* out_mem = new bus_T[VTA_ACC_BUFF_DEPTH * OUT_MAT_AXI_RATIO];
  bus_T* acc_mem = new bus_T[VTA_ACC_BUFF_DEPTH * ACC_MAT_AXI_RATIO];
  uop_T* uop_mem = new uop_T[VTA_UOP_BUFF_DEPTH];

  crf_type* crf_mem = new crf_type[VTA_ACC_BUFF_DEPTH];
  crx_type* crx_mem = new crx_type[VTA_ACC_BUFF_DEPTH];


#else
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO]; // [512][32]
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO]; // [2048][2]
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]; // [2048][2]
  uop_T uop_mem[VTA_UOP_BUFF_DEPTH];
  bus_T acc_mem[VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO]; // [2048] [8]

  crf_type crf_mem[VTA_ACC_BUFF_DEPTH];
  crx_type crx_mem[VTA_ACC_BUFF_DEPTH];
#endif

  vtaS.write(0);

  // Temporary instructions
  insn_T tmp_load;
  insn_T tmp_gemv;
  insn_T tmp_store;

  // Peeking status
  bool tmp_load_popped = false;
  bool tmp_gemm_popped = false;
  bool tmp_store_popped = false;
  int exit_counter = 0;

  bool fetched = false; // NEW
  bool var_break = false; // NEW
  unsigned long long temp_insn[2];  // NEW

  vtaS.write(0);
  while (1) {

  	if(!fetched){
  		fetch();
  		fetched = true;
  	}

    // First execute as many load instructions as possible
  	int  que = tmp_load_queue.num_available();
    while (tmp_load_queue.num_available()!=0 || tmp_load_popped == true) { // NEW changed for sc_fifo method
      // Pop the load instruction
      if (!tmp_load_popped) {
        tmp_load_queue.read(tmp_load);
        tmp_load_popped = true;
      }
      // Check dependences and invoke the load stage
  		temp_insn[0] = tmp_load.range(63,0).to_uint64(); // NEW
  		temp_insn[1] = tmp_load.range(127,64).to_uint64(); // NEW

      VTAInsn insn;
      insn.generic = *((VTAGenericInsn *) &temp_insn);
      que  = g2l_dep_queue.num_available();
      if ((insn.generic.pop_next_dep && g2l_dep_queue.num_available()!=0 ) || // NEW changed for sc_fifo method
          !insn.generic.pop_next_dep) {
        // Push the instruction in the load queue
        load_queue.write(tmp_load);
        tmp_load_popped = false;
        load( inp_mem, wgt_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }


    // Next execute as many gemm instructions as possible
    while (tmp_gemm_queue.num_available()!=0 || tmp_gemm_popped == true) {
      // Pop the gemm instruction
      if (!tmp_gemm_popped) {
        tmp_gemm_queue.read(tmp_gemv);
        tmp_gemm_popped = true;
      }
      // Check dependences and invoke the load stage
  		temp_insn[0] = tmp_gemv.range(63,0).to_uint64(); // NEW
  		temp_insn[1] = tmp_gemv.range(127,64).to_uint64(); // NEW
      VTAInsn insn;
      insn.generic = *((VTAGenericInsn *) &temp_insn);
      if (
        (insn.generic.pop_prev_dep && l2g_dep_queue.num_available()!=0 &&
         insn.generic.pop_next_dep && s2g_dep_queue.num_available()!=0 ) ||
        (!insn.generic.pop_prev_dep && insn.generic.pop_next_dep &&
         s2g_dep_queue.num_available()!=0) ||
        (insn.generic.pop_prev_dep && l2g_dep_queue.num_available()!=0 &&
        !insn.generic.pop_next_dep) ||
        (!insn.generic.pop_prev_dep && !insn.generic.pop_next_dep)
      ) {
        // Push the instruction in the load queue
        gemm_queue.write(tmp_gemv);
        tmp_gemm_popped = false;
        compute(crf_mem,crx_mem,uop_mem,inp_mem, wgt_mem, out_mem,acc_mem);
      } else {
        // Execution of load stage pending on completion of other stages,
        // so break here...
        break;
      }
    }


    // Finally execute as many store instructions as possible
    while (tmp_store_queue.num_available()!=0 || tmp_store_popped == true) {
      // Pop the load instruction
      if (!tmp_store_popped) {
        tmp_store_queue.read(tmp_store);
        tmp_store_popped = true;
      }
      // Check dependences and invoke the load stage
  		temp_insn[0] = tmp_store.range(63,0).to_uint64(); // NEW
  		temp_insn[1] = tmp_store.range(127,64).to_uint64(); // NEW
      VTAInsn insn;
      insn.generic = *((VTAGenericInsn *) &temp_insn);

      if ((insn.generic.pop_prev_dep && g2s_dep_queue.num_available()!=0) ||
          !insn.generic.pop_prev_dep) {
        // Push the instruction in the load queue
        store_queue.write(tmp_store);
        tmp_store_popped = false;
        store(crf_mem,crx_mem, acc_mem,out_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }

    // Check if we get a signal that we are done
    if (done) {
    	vta_on_exit();
      fetched=false;
      var_break=true;
    }
    exit_counter++;
    if (exit_counter > 1000) {
      if (tmp_load_popped) {
        if (g2l_dep_queue.num_available()==0) {
//          printf("waiting on g2l\n");
        }
      }
      if (tmp_gemm_popped) {
    		temp_insn[0] = tmp_gemv.range(63,0).to_uint64(); // NEW
    		temp_insn[1] = tmp_gemv.range(127,64).to_uint64(); // NEW
        VTAGenericInsn insn = *((VTAGenericInsn *) &temp_insn);
        if (l2g_dep_queue.num_available()==0 && insn.pop_prev_dep) {
//          printf("waiting on l2g\n");
        }
        if (s2g_dep_queue.num_available()==0 && insn.pop_next_dep) {
//          printf("waiting on s2g\n");
        }
      }
      if (tmp_store_popped) {
        if (g2s_dep_queue.num_available()==0) {
//          printf("waiting on g2s\n");
        }
      }
      vta_on_exit();
      fetched=false;
      var_break=true;
    }

    while(var_break){
      exit_counter=0;

      bus_T acc_data = acc_mem[0];
      sc_uint<64> sc_accum = acc_data;
      int accum = sc_accum.range(31,0);

    	sc_pause();
    	wait();
    	wait();
    	var_break=false;
    }
    wait();

  }
}

