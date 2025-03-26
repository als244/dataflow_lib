typedef enum flash_dtype {
	FLASH_FP16,
	FLASH_FP32,
	FLASH_BF16,
	FLASH_FP8 // e4m3
} Flash_DType;


#ifdef __cplusplus
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "flash.h"
extern "C" {
#endif

int flash_fwd_wrapper(CUstream stream, int total_tokens, int num_seqs,  int * cu_seqlens, int max_seqlen, Flash_DType flash_dtype, void * x_q, void * x_k, void * x_v, void * x_attn_out, int num_q_heads, int num_kv_heads, int head_dim, void * softmax_lse, int arch, int num_sm, void * attn_workspace);

#ifdef __cplusplus
}
#endif
