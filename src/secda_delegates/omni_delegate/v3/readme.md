# OMNI Delegate v1_tensor_manager: Detailed Walkthrough

This document explains what the v1_tensor_manager variant does, how it differs from the regular v1 variant, and how its tensor-management flow is intended to support layer fusion.

## 1) What this variant is trying to achieve

The regular v1 delegate mostly reads and writes tensors through the normal TfLite tensor buffers.

The v1_tensor_manager delegate adds a software-side DMA tensor manager path so delegated ops can:

1. preload external input tensors into a DMA-backed memory region,
2. run delegated ops using pointers to DMA-backed buffers,
3. keep intermediate tensors inside DMA-backed memory,
4. only copy boundary outputs back to TfLite memory.

This is the key behavior needed for layer fusion: avoid round-tripping intermediate tensors through normal CPU/TfLite memory between delegated layers.

## 2) File map and what each file does

Top-level files in this folder:

- BUILD, omni_delegate.h, omni_delegate_provider.cc, omni_delegate_adaptor.cc:
	standard delegate plumbing for creating, registering, and exposing the delegate.
- omni_delegate.cc:
	core behavior, including all tensor-manager integration.
- util.h, util_prep.h:
	op data structures and per-op Prepare helpers.
- accelerator/*:
	accelerator config and SystemC/driver-side support files.

Important: the tensor-manager behavior is almost entirely in omni_delegate.cc. The accelerator-side changes are mostly dependency/API migration and buffer macro updates.

## 3) High-level flow (lifecycle)

The delegate lifecycle in this implementation is:

1. Init:
	 initialize SystemC/DMA objects and set global state.
2. Prepare:
	 gather delegated nodes and per-op prep data; register tensors with DMATensorManager.
3. Eval:
	 per delegated node:
	 - preload external inputs into DMA memory if needed,
	 - run op using DMA-backed pointers,
	 - copy external outputs back to TfLite memory if needed.
4. Delete:
	 save profiling, unmap/free resources, print tensor-manager debug info.

## 4) Exact differences from regular v1

### 4.1 Core functional differences in omni_delegate.cc

Compared to v1, v1_tensor_manager adds:

- Global tensor manager object:
	DMATensorManager dtm.
- DMA mmapped host buffer object:
	mm_buf_int dbuf1(dma_in0, MM_BL / 4).
- DMA buffer mode switch:
	bool use_dma_buffer = true.

During Init (SYSC path):

- binds DMA input and output to the same mmapped buffer (dbuf1),
- gives the buffer pointer and size to dtm via dtm.set_buffer(...).

During Prepare:

- for every op input/output tensor in the delegated partition, creates or updates a DMATensor record,
- records tensor metadata (id, rank/dims, bitwidth, static/external flags),
- tracks producer/consumer node ids via input_nodes/output_nodes.

During Eval:

- preloads each external input tensor to DMA memory via dtm.copy_to_mmap,
- for selected ops, swaps TfLite data pointers for DMA-backed pointers from dtm.get_tensor(...)->data,
- after each op, copies external outputs back from DMA memory via dtm.copy_from_mmap.

During Delete:

- adds dtm.print_tensors() debug dump before delegate destruction.

### 4.2 Delegated op set changed

Regular v1 only enables delegation for CONV2D in IsNodeSupportedByDelegate.

v1_tensor_manager enables delegation for:

- ADD
- CONV2D
- FC
- SOFTMAX
- DWCONV2D
- SHAPE

This is a big behavior difference beyond tensor memory management.

### 4.3 Accelerator-side and build differences

v1_tensor_manager also changes:

- accelerator/BUILD: secdav5 -> secdav6 dependency.
- accelerator/acc_config.sc.h:
	- includes axi_support v6 API,
	- MM_BL changed from 0x100000 to 0x20000000,
	- adds dma_buf/mm_buf_int aliases used by omni_delegate.cc.
- accelerator/driver/acc_container.h and systemc_binding.h:
	include v6 API headers.

One subtle but important point:

- v1_tensor_manager BUILD files still reference v1 accelerator driver targets
	in dependency paths (not v1_tensor_manager accelerator target paths).
	So some accelerator file changes in this folder may not be active unless
	build deps are intentionally redirected.

## 5) Tensor-manager model in this code

The DMATensorManager and DMATensor type definitions are not in this folder, but usage in omni_delegate.cc implies this model:

- DMATensorManager stores a map from TfLite tensor id (TID) to DMATensor metadata.
- Each DMATensor tracks:
	- tensor id and shape/bitwidth,
	- whether it is static (weights/const style),
	- whether it is external input (boundary input from TfLite side),
	- whether it is external output (boundary output back to TfLite side),
	- whether currently mmapped (has DMA-side buffer state),
	- producer/consumer node lists.
- Manager methods used:
	- set_buffer(ptr, size)
	- check_exists(TID)
	- add_allocate_tensor(meta, original_ptr)
	- get_tensor(TID)
	- copy_to_mmap(TID, src_ptr)
	- copy_from_mmap(TID, dst_ptr)
	- print_tensors()

## 6) Prepare phase, in detail

After normal per-op Prepare_* calls, v1_tensor_manager executes tensor registration logic:

1. Iterate all inputs of each delegated node.
2. Build tensor metadata from TfLite tensor:
	 - dims/rank,
	 - bitwidth: int8 -> 8 else 32,
	 - isStatic for inputs: allocation_type == kTfLiteMmapRo,
	 - isExternalInput = !isStatic.
3. Add or update DMATensor entry and append current node id to input_nodes.

Then output tensors:

1. Iterate outputs of each delegated node.
2. Build metadata similarly.
3. isStatic for outputs checks kTfLiteMmapRo or kTfLitePersistentRo.
4. isExternalOutput is true when this output tensor appears in delegate-node outputs (partition boundary output).
5. Add or update DMATensor entry and append current node id to output_nodes.

Effectively, Prepare builds a graph-aware tensor table that can later decide when copies are needed.

## 7) Eval phase, in detail

### 7.1 Preload stage (before op compute)

For each input tensor of current delegated node:

- if tensor is external input and not yet mmapped,
- call dtm.copy_to_mmap(TID, tensor->data.raw).

This makes the current node read from DMA-backed storage even when source data originated in TfLite-managed memory.

### 7.2 Compute stage: which ops use DMA-backed pointers

When use_dma_buffer is true (it is hardcoded true), these ops use DMATensor data pointers:

- ADD
- CONV2D
- FULLY_CONNECTED
- DEPTHWISE_CONV2D
- SOFTMAX

For these ops, input/output pointers are replaced with dtm tensor pointers before the math loops.

Ops that are delegated but do not really consume tensor payload (like SHAPE) are unaffected.

### 7.3 Store-back stage (after op compute)

For each output tensor of current delegated node:

- if tensor is external output and not yet mmapped,
- copy from DMA buffer back into TfLite tensor memory using dtm.copy_from_mmap.

This is the key rule enabling fusion behavior:

- internal tensors stay in DMA memory,
- only partition boundary outputs are written back.

## 8) Why this helps layer fusion

Without this manager, each delegated op naturally reads/writes TfLite buffers directly.
That can force repeated CPU-visible memory traffic between adjacent delegated ops.

With this manager:

- op N writes output to DMA-backed tensor memory,
- op N+1 reads same tensor from DMA-backed memory,
- no TfLite copy is needed for intermediate tensors.

This is exactly the software-side prerequisite for fusing or chaining multiple delegated layers with reduced memory movement.

## 9) Known limitations and gotchas in the current code

1. DMATensorManager implementation is external.
	 Semantics here are inferred from call sites.

2. use_dma_buffer is global and always true.
	 There is no runtime flag/plumbed option to disable for debug except source edits.

3. Non-SYSC path does not call dtm.set_buffer in Init.
	 If this path is used, verify DMATensorManager has a valid backing strategy.

4. BUILD dependency wiring points to v1 accelerator paths.
	 v1_tensor_manager accelerator edits may be inactive unless dependencies are updated.

5. MM_BL comment mismatch.
	 MM_BL is 0x20000000 but comments still say 1MB.

6. Graph lifetime information exists but is not yet used for reuse/free.
	 output_dependencies and node_output_needed are maintained, but tensor release/reuse is not implemented in this file.

## 10) Practical interpretation for your next step (fusion work)

What already exists and is useful for fusion:

- tensor metadata table with producer/consumer relationships,
- boundary tensor identification,
- staged preload + compute + store-back flow,
- DMA-resident intermediate tensor path.

What is still missing for robust fused scheduling:

- explicit lifetime-based tensor reclaim/reuse,
- deterministic policy for in-place updates or buffer aliasing,
- unified handling for all delegated ops that touch data,
- clear build wiring so the intended accelerator/runtime path is guaranteed.

## 11) Quick checklist to validate behavior

When running this delegate, validate:

1. prepare-time tensor table creation occurs for every delegated tensor id.
2. first use of external input prints preload path and copies once.
3. internal tensors do not get copied back each layer.
4. partition boundary outputs are copied back before TfLite consumes them.
5. dtm.print_tensors output at delegate destroy matches expected graph/tensor roles.

If those conditions hold, the software-side tensor-manager scaffold for fusion is functioning as intended.
