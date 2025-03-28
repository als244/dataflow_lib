#include "transformer_ops.h"


// FOR FLASH3 ATTENTION:

// Only support for FP16, BF16, and FP8
// if TYPE FP8, output must be BF16
// Softmax LSE is of type FP32 and has length total_q * num_q_heads

// To compute required size of attn_workspace:

// attn_workspace_size = 0

// Occum and LSE accum:
// If num_splits > 1:
//      attn_workspace_size += num_splits * sizeof(float) * num_q_heads * total_q * (1 + head_dim)

// Tile count sem: 
// If arch >= 90 || num_splits > 1:
//      attn_workspace_size += sizeof(int)

// Dynamic split ptr for each seq:
// If num_seqs <= 992:
//      attn_workspace_size += num_seqs * sizeof(int)


// ASSUME CAUSAL

// - cum_q_seqlens should be of length num_seqs + 1, starting with 0
//		- cumsum of # of queries in each sequence
// - k_seqlens should be of length num_seqs
//		- total number of keys in sequence (should be >= # of queries) 
//			- (assumes that if sequence has Q queries and K keys, the starting position of Q_0
//				occurs at position K - Q)

int submit_attention(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype fwd_dt,
						int num_seqs, int total_q, int total_k, 
						int * cum_q_seqlens, int max_seqlen_q,
						int * k_seqlens, int max_seqlen_k,
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, 
						void * x_attn_out, void * softmax_lse, 
						void * attn_workspace) {


	int ret;

	Op attention_op;

	set_external_flash3_attention_fwd_skeleton(&attention_op.op_skeleton);

	void ** op_args = attention_op.op_args;

	op_args[0] = &fwd_dt;
	op_args[1] = &num_seqs;
	op_args[2] = &total_q;
	op_args[3] = &total_k;
	op_args[4] = &cum_q_seqlens;
	op_args[5] = &max_seqlen_q;
	op_args[6] = &k_seqlens;
	op_args[7] = &max_seqlen_k;
	op_args[8] = &num_q_heads;
	op_args[9] = &num_kv_heads;
	op_args[10] = &head_dim;
	op_args[11] = &x_q;
	op_args[12] = &x_k;
	op_args[13] = &x_v;
	op_args[14] = &x_attn_out;
	op_args[15] = &softmax_lse;
	op_args[17] = &attn_workspace;

	ret = (handle -> submit_op)(handle, &attention_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention_op...\n");
		return -1;
	}

	return 0;



}