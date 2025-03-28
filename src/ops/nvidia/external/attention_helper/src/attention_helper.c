#include "attention_helper.h"

int flash3_attention_fwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra) {

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int arch = device_info -> arch_num;
	int sm_count = device_info -> sm_count;

	CUstream * streams = (CUstream *) (dataflow_handle -> streams);

	CUstream stream = streams[stream_id];

	void ** op_args = op -> op_args;

	// Follow same paradigm as submitting to command queue
	// pass in pointers to each of the arguments...

	// so now we can deference them...

	int flash_dtype_as_int = *((int *)(op_args[0]));

	int num_seqs = *((int *)(op_args[1]));
	int total_q = *((int *)(op_args[2]));
	int total_k = *((int *)(op_args[3]));
	int * cum_q_seqlens = *((int **) op_args[4]);
	int max_seqlen_q = *((int *)(op_args[5]));
	int * k_seqlens = *((int **) op_args[6]);
	int max_seqlen_k = *((int *)(op_args[7]));
	
	int num_q_heads = *((int *)(op_args[8]));
	int num_kv_heads = *((int *)(op_args[9]));
	int head_dim = *((int *)(op_args[10]));

	void * x_q = *((void **) op_args[11]);
	void * x_k = *((void **) op_args[12]);
	void * x_v = *((void **) op_args[13]);

	void * x_attn_out = *((void **) op_args[14]);
	void * softmax_lse = *((void **) op_args[15]);

	void * attn_workspace = *((void **) op_args[16]);

	int ret = flash3_fwd_wrapper(stream, arch, sm_count,
									num_seqs, total_q, total_k,
									cum_q_seqlens, max_seqlen_q,
									k_seqlens, max_seqlen_k,
									flash_dtype_as_int,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									attn_workspace);


	return ret;
}


int flash3_attention_bwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra) {

	return 0;
}