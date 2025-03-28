#ifndef FLASH3_WRAPPER_H
#define FLASH3_WRAPPER_H

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

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
//      - cumsum of # of queries in each sequence
// - k_seqlens should be of length num_seqs
//      - total number of keys in sequence (should be >= # of queries) 
//          - (assumes that if sequence has Q queries and K keys, the starting position of Q_0
//              occurs at position K - Q)

int flash3_fwd_wrapper(CUstream stream, 
                            int flash_dtype_as_int, 
                            int arch, int num_sm,
                            int num_seqs, int total_q, int total_k, 
                            int * cum_q_seqlens, int max_seqlen_q,
                            int * k_seqlens, int max_seqlen_k,
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, void * softmax_lse,  
                            void * attn_workspace);

// TODO: BWD

#endif