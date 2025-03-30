#ifndef TRANSFORMER_OPS_H
#define TRANSFORMER_OPS_H

#include "dataflow.h"
#include "ops.h"
#include "dataflow_handle.h"
#include "set_native_op_skeletons.h"
#include "set_external_op_skeletons.h"


// CORE COMPUTE FUNCTIONS!
//	- these end up calling external library implementations...

// From matmul.c

// If A, C, D are all stored in Row-Major
// And B is stored in Col-Major. If so, it compute:
// D = alpha * AB + beta * C

// If B is stored in Row-Major that implies it computes:
// D = alpha * AB^T + beta * C
int submit_matmul(Dataflow_Handle * handle, int stream_id, 
					DataflowDatatype a_dt, DataflowDatatype b_dt, DataflowDatatype c_dt, DataflowDatatype d_dt,
					DataflowDatatype compute_dt,
					int M, int K, int N,
					float alpha, float beta,
					uint64_t workspaceBytes, void * workspace,
					void * A, void * B, void * C, void * D,
					int num_procs);



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
						void * attn_workspace);



// TODO: 
/*
int submit_attention_bwd(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype fwd_dt,
						int num_seqs, int total_q, int total_k, 
						int * cum_q_seqlens, int max_seqlen_q,
						int * k_seqlens, int max_seqlen_k, 
						int flash_dtype_as_int, 
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, void * x_attn_out, void * softmax_lse, 
						void * attn_workspace);
*/


// NATIVE OPS

// From norm_ops.c

int submit_rms_norm(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * rms_weight, void * X, void * out, float * weighted_sums, float * rms_vals);


int submit_rms_norm_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_weighted_sums, float * fwd_rms_vals,
								 void * rms_weight, void * X_inp, void * upstream_dX, void * dX);


int submit_rms_norm_bwd_w(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals, void * X_inp, void * upstream_dX, void * dW);


// From attn_misc_ops.c

int submit_rope(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * X_q, void * X_k);

int submit_rope_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype bwd_dt, 
						uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * dX_q, void * dX_k);


int submit_copy_to_seq_context(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int total_tokens, int kv_dim, 
						void * X_k, void * X_v, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);


// From mlp_misc_ops.c

int submit_swiglu(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int num_rows, int num_cols, 
						void * x_w1, void * x_w3, void * out);


int submit_swiglu_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int num_rows, int num_cols, 
						void * x_w1, void * x_w3, 
						void * upstream_dX, void * dX_w1, void * dX_w3);

// From loss_misc_ops.c

int submit_softmax(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int n_rows, int n_cols,
						void * X, void * out);

int submit_cross_entropy_loss(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype bwd_dt,
								int n_rows, int n_cols,
								void * pred_logits, uint32_t * labels);




#endif
