#ifndef FLASH3_WRAPPER_H
#define FLASH3_WRAPPER_H

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

int flash3_fwd_wrapper(CUstream stream, int total_tokens, int num_seqs,  int * cu_seqlens, int max_seqlen, int flash_dtype_as_int, 
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, void * x_attn_out, void * softmax_lse, 
                            int arch, int num_sm, 
                            void * attn_workspace);

#endif