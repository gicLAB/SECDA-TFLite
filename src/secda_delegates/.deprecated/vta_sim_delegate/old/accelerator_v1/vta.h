#ifndef VTA_H
#define VTA_H

#include <systemc.h>
#include "AXI4_if.h"
#include "hw_spec.h"


#ifndef __SYNTHESIS__
//#define VLOG(X) cout X
#define VLOG(X)
//#define VLOG2(X) cout X
#define VLOG2(X)
#define VLOG3(X) cout X
#else
#define VLOG(X)
#define VLOG2(X)
#endif

#define STREAM_IN_DEPTH 512

/* \typedef bus_T memory bus datatype*/
//typedef sc_uint<VTA_BUS_WIDTH> bus_T;
typedef unsigned long long bus_T;

/* \typedef uop_T Micro-op datatype*/
//typedef sc_uint<VTA_UOP_WIDTH> uop_T;
typedef unsigned int uop_T;

/* \typedef inp_T Input datatype*/
typedef sc_int<VTA_INP_WIDTH> inp_T;

/* \typedef wgt_T Weight datatype*/
typedef sc_int<VTA_WGT_WIDTH> wgt_T;

/* \typedef out_T Output datatype*/
typedef sc_int<VTA_OUT_WIDTH> out_T;

/* \typedef acc_T Accumulator datatype*/
typedef sc_int<VTA_ACC_WIDTH> acc_T;

/* \typedef mul_T Multiplier output datatype*/
typedef sc_int<VTA_WGT_WIDTH+VTA_INP_WIDTH+1> mul_T;

/* \typedef sum_T GEMM accumulator datatype*/
typedef sc_int<VTA_WGT_WIDTH+VTA_INP_WIDTH+VTA_LOG_BLOCK_IN+1> sum_T;

/* \typedef uop_idx_T Micro-op SRAM index datatype*/
typedef sc_uint<VTA_LOG_UOP_BUFF_DEPTH+1> uop_idx_T;

/* \typedef inp_idx_T Input SRAM index datatype*/
typedef sc_uint<VTA_LOG_INP_BUFF_DEPTH+1> inp_idx_T;

/* \typedef wgt_idx_T Weight SRAM index datatype*/
typedef sc_uint<VTA_LOG_WGT_BUFF_DEPTH+1> wgt_idx_T;

/* \typedef acc_idx_T Accumulator SRAM index datatype*/
typedef sc_uint<VTA_LOG_ACC_BUFF_DEPTH+1> acc_idx_T;

/* \typedef opcode_T Opcode datatype*/
typedef sc_uint<VTA_OPCODE_BIT_WIDTH> opcode_T;

/* \typedef insn_T Instruction datatype*/
typedef sc_biguint<VTA_INS_WIDTH> insn_T;

/* \typedef loop_T Loop bound datatype*/
typedef sc_uint<VTA_LOOP_ITER_WIDTH> loop_T;

/* \typedef memop_id_T Memory operation ID datatype*/
typedef sc_uint<VTA_MEMOP_ID_BIT_WIDTH> memop_id_T;

/* \typedef memop_sram_T Memory operation SRAM index datatype*/
typedef sc_uint<VTA_MEMOP_SRAM_ADDR_BIT_WIDTH> memop_sram_T;

/* \typedef memop_dram_T Memory operation DRAM index datatype*/
typedef sc_uint<VTA_MEMOP_DRAM_ADDR_BIT_WIDTH> memop_dram_T;

/* \typedef memop_size_T Memory operation range datatype*/
typedef sc_uint<VTA_MEMOP_SIZE_BIT_WIDTH> memop_size_T;

/* \typedef memop_stride_T Memory operation stride datatype*/
typedef sc_uint<VTA_MEMOP_STRIDE_BIT_WIDTH> memop_stride_T;

/* \typedef memop_pad_T Memory operation pad width datatype*/
typedef sc_uint<VTA_MEMOP_PAD_BIT_WIDTH> memop_pad_T;

/* \typedef aluop_opcode_T ALU operation opcode datatype*/
typedef sc_uint<VTA_ALU_OPCODE_BIT_WIDTH> aluop_opcode_T;

/* \typedef aluop_imm_T ALU operation immediate datatype*/
typedef sc_int<VTA_ALUOP_IMM_BIT_WIDTH> aluop_imm_T;

/* \typedef aluop_shr_arg_T ALU operation shift right immediate datatype*/
typedef sc_int<VTA_SHR_ARG_BIT_WIDTH> aluop_shr_arg_T;

/* \typedef aluop_mul_arg_T ALU operation multiply datatype*/
typedef sc_int<VTA_MUL_ARG_BIT_WIDTH> aluop_mul_arg_T;



typedef int crf_type;
typedef int8_t crx_type;



#define SC_INP_ELEM_BYTES_RATIO 4
#define MAX8 127
#define MIN8 -128
#define MAX32 2147483647
#define MIN32 -2147483648

#define DIVMAX 2147483648
#define POS 1073741824
#define NEG -1073741823


#define ACCNAME VTA_SYSC
SC_MODULE(ACCNAME) {

  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_out<unsigned int> vtaS;


  AXI4M_bus_port<unsigned long long> insns; // only 64 bits should be 128, 1 << VTA_LOG_INS_WIDTH
  AXI4M_bus_port<unsigned int> uops; // (1 << VTA_LOG_UOP_WIDTH)
  AXI4M_bus_port<unsigned long long> data; // (1 << VTA_LOG_BUS_WIDTH)

  sc_in<unsigned int> insn_count;
  sc_in<unsigned int> ins_addr;
  sc_in<unsigned int> uops_addr;
  sc_in<unsigned int> input_addr;
  sc_in<unsigned int> weight_addr;
  sc_in<unsigned int> bias_addr;
  sc_in<unsigned int> output_addr;

  sc_in<unsigned int> crf_addr;
  sc_in<unsigned int> crx_addr;
  sc_in<unsigned int> ra_sig;

  // Instantiate temporary instruction queues (used for peeking)
  sc_fifo<insn_T> tmp_load_queue;
  sc_fifo<insn_T> tmp_gemm_queue;
  sc_fifo<insn_T> tmp_store_queue;


  // Instatiate physical instruction queues
  sc_fifo<insn_T> load_queue;
  sc_fifo<insn_T> gemm_queue;
  sc_fifo<insn_T> store_queue;

  // Dependence queues
  sc_fifo<insn_T> l2g_dep_queue;
  sc_fifo<insn_T> s2g_dep_queue;
  sc_fifo<insn_T> g2l_dep_queue;
  sc_fifo<insn_T> g2s_dep_queue;




  sc_signal<unsigned int> done;
  int ra =0;
  int layer=0;


  void vta_on_exit();

  void vta_main();

  void fetch();


#ifndef __SYNTHESIS__

  void load(bus_T *,bus_T *);

  void compute(
    crf_type *,
    crx_type *,
		uop_T *,
  	bus_T *,
    bus_T *,
		bus_T *,
		bus_T *
						);

  void gemm(
  	  insn_T insn_raw,
  	  uop_T [VTA_UOP_BUFF_DEPTH],
			bus_T *,
			bus_T *,
  	  bus_T *,
  	  bus_T *);

  void alu(
    crf_type *,
    crx_type *,
    insn_T insn_raw,
    uop_T [VTA_UOP_BUFF_DEPTH],
		bus_T *,
		bus_T *,
	  bus_T *,
	  bus_T *);

  void store(
    crf_type *,
    crx_type *,
    bus_T *,
    bus_T *out_mem);


  int Quantised_Multiplier(int, int, sc_int<8>);

  template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
  void print_mem(IDX_T,WIDE_T *);

  template <typename DATA_T, int MAT_AXI_RATIO>
  void reset_mem(memop_sram_T&,memop_sram_T,DATA_T *);

  template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
  void load_pad_2d(AXI4M_bus_port<unsigned long long> &, sc_in<unsigned int> &,
  		DATA_T *,
  	  memop_sram_T,memop_dram_T,
  	  memop_size_T, memop_size_T,
  	  memop_stride_T,
  	  memop_pad_T,memop_pad_T,
  	  memop_sram_T,memop_sram_T);

  template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
  void load_2d(AXI4M_bus_port<unsigned long long> &, sc_in<unsigned int> &,
  		DATA_T *,
			memop_sram_T,memop_dram_T,
			memop_size_T,memop_size_T,
			memop_stride_T);


  template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
  void load_1ds(AXI4M_bus_port<unsigned long long> &, 
      sc_in<unsigned int> &,
  		DATA_T *,
			memop_sram_T,memop_dram_T,
			memop_size_T,memop_size_T,
			memop_stride_T);

  template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
  void read_tensor(
    IDX_T ,
    WIDE_T *,
    NARROW_T [Y_DIM][X_DIM]);


  template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
  void write_tensor(
    IDX_T ,
    NARROW_T [Y_DIM][X_DIM],
    WIDE_T *);




#else

  void load(bus_T [VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
			bus_T [VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO]);

  void compute(
		uop_T [VTA_UOP_BUFF_DEPTH],
    bus_T [VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
    bus_T [VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
    bus_T [VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO],
		bus_T [VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO]
		);


  void gemm(
  	  insn_T insn_raw,
  	  uop_T [VTA_UOP_BUFF_DEPTH],
  	  bus_T [VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO],
  	  bus_T [VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  	  bus_T [VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
  	  bus_T [VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]);

  void alu(
    insn_T insn_raw,
    uop_T [VTA_UOP_BUFF_DEPTH],
    bus_T [VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO],
    bus_T [VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
    bus_T [VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
    bus_T [VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]);


  void store(
    bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]);


  template <typename DATA_T, int MAT_AXI_RATIO, int TEMP_BUFF_DEPTH>
  void reset_mem(memop_sram_T&,memop_sram_T,DATA_T [TEMP_BUFF_DEPTH][MAT_AXI_RATIO]);

  template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES, int TEMP_BUFF_DEPTH>
  void load_pad_2d(AXI4M_bus_port<unsigned long long> &, sc_in<unsigned int> &,
  		DATA_T [TEMP_BUFF_DEPTH][MAT_AXI_RATIO],
  	  memop_sram_T,memop_dram_T,
  	  memop_size_T, memop_size_T,
  	  memop_stride_T,
  	  memop_pad_T,memop_pad_T,
  	  memop_sram_T,memop_sram_T);

  template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES,int TEMP_BUFF_DEPTH>
  void load_2d(AXI4M_bus_port<unsigned long long> &, sc_in<unsigned int> &,
  		DATA_T [TEMP_BUFF_DEPTH][MAT_AXI_RATIO],
			memop_sram_T,memop_dram_T,
			memop_size_T,memop_size_T,
			memop_stride_T);

  template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
  void read_tensor(
    IDX_T ,
    WIDE_T [][NARROW_W * Y_DIM * X_DIM / WIDE_W],
    NARROW_T [Y_DIM][X_DIM]);


  template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
  void write_tensor(
    IDX_T ,
    NARROW_T [Y_DIM][X_DIM],
    WIDE_T [][NARROW_W * Y_DIM * X_DIM / WIDE_W]);


#endif










  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_),
  		tmp_load_queue(STREAM_IN_DEPTH),
			tmp_gemm_queue(STREAM_IN_DEPTH),
			tmp_store_queue(STREAM_IN_DEPTH),
			load_queue(STREAM_IN_DEPTH),
			gemm_queue(STREAM_IN_DEPTH),
			store_queue(STREAM_IN_DEPTH),
			l2g_dep_queue(STREAM_IN_DEPTH),
			s2g_dep_queue(STREAM_IN_DEPTH),
			g2l_dep_queue(STREAM_IN_DEPTH),
			g2s_dep_queue(STREAM_IN_DEPTH)
  		, insns ("insns") , uops ("uops") , data ("data")
  {

    SC_CTHREAD(vta_main, clock.pos());
    reset_signal_is(reset, true);

#pragma HLS RESET variable=reset

//#pragma HLS RESOURCE variable=insns_in core=AXI4Stream metadata="-bus_bundle S_AXIS_insns_in" port_map={{insns_in_0 TDATA} {insns_in_1 TLAST}}
//#pragma HLS RESOURCE variable=uops_in core=AXI4Stream metadata="-bus_bundle S_AXIS_uops_in" port_map={{uops_in_0 TDATA} {uops_in_1 TLAST}}
//#pragma HLS RESOURCE variable=data_in core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA3" port_map={{din3_0 TDATA} {din3_1 TLAST}}
//#pragma HLS RESOURCE variable=insns_out core=AXI4Stream metadata="-bus_bundle M_AXIS_insns_out" port_map={{insns_out_0 TDATA} {insns_out_1 TLAST}}
//#pragma HLS RESOURCE variable=uops_out core=AXI4Stream metadata="-bus_bundle M_AXIS_uops_out" port_map={{uops_out_0 TDATA} {uops_out_1 TLAST}}
//#pragma HLS RESOURCE variable=data_out core=AXI4Stream metadata="-bus_bundle M_AXIS_data_out" port_map={{data_out_0 TDATA} {data_out_1 TLAST}}


  }
};


#endif // VTA_H
















