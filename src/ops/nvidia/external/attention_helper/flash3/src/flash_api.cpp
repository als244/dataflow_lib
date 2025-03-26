/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include "cutlass/numeric_types.h"

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"


bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
}

int get_num_splits(Flash_fwd_params const& params) {
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl) {
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        if (!params.is_e4m3) {
                            if (params.is_bf16) {
                                if (params.d <= 64) {
                                    if (params.dv > 256 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else if (params.dv > 64 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 96) { run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 128) { run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 256) { run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            } else {
                                if (params.d <= 64) {
                                    if (params.dv > 256 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else if (params.dv > 64 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 96) { run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 128) { run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 256) { run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            }
                        } else {
                            if (params.d <= 64) { run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            if (params.d <= 96) { run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            if (params.d <= 128) { run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            if (params.d <= 192) {
                                if (params.dv <= 128 && Arch == 90) {
                                    run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                } else {
                                    run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                }
                            }
                            if (params.d <= 256) { run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                        }
                    });
                });
            });
        });
    });

    if (params.num_splits > 1){
        run_mha_fwd_combine(params, stream, true);
    }

    
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            if (!params.is_bf16) {
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap>(params, stream); }
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap>(params, stream); }
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap>(params, stream); }
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap>(params, stream); }
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap>(params, stream); }
            } else {
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
            }
        });
    });
}

typedef enum {
    DATAFLOW_NONE,
    DATAFLOW_VOID,
    DATAFLOW_FP64,
    DATAFLOW_FP32,
    DATAFLOW_FP16,
    DATAFLOW_BF16,
    DATAFLOW_FP8E4M3,
    DATAFLOW_FP8E5M2,
    DATAFLOW_UINT64,
    DATAFLOW_UINT32,
    DATAFLOW_UINT16,
    DATAFLOW_UINT8,
    DATAFLOW_LONG,
    DATAFLOW_INT,
    DATAFLOW_BOOL,
    DATAFLOW_FP64_SCALAR,
    DATAFLOW_FP32_SCALAR,
    DATAFLOW_FP16_SCALAR,
    DATAFLOW_BF16_SCALAR,
    DATAFLOW_FP8E4M3_SCALAR,
    DATAFLOW_FP8E5M2_SCALAR,
    DATAFLOW_UINT64_SCALAR,
    DATAFLOW_UINT32_SCALAR,
    DATAFLOW_UINT16_SCALAR,
    DATAFLOW_UINT8_SCALAR,
    DATAFLOW_LONG_SCALAR,
    DATAFLOW_INT_SCALAR,
    DATAFLOW_BOOL_SCALAR
} DataflowDatatype;

extern "C" {
    
    int flash_fwd_wrapper(CUstream stream, int total_tokens, int num_seqs,  int * cu_seqlens, int max_seqlen, int flash_dtype_as_int, 
                                int num_q_heads, int num_kv_heads, int head_dim, 
                                void * x_q, void * x_k, void * x_v, void * x_attn_out, void * softmax_lse, 
                                int arch, int num_sm) {

        int model_dim = num_q_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        Flash_fwd_params params;
        memset(&params, 0, sizeof(Flash_fwd_params));

        DataflowDatatype flash_dtype = (DataflowDatatype) flash_dtype_as_int;

        if (flash_dtype == DATAFLOW_FP32){
            params.is_fp32 = true;
        }
        if (flash_dtype == DATAFLOW_BF16){
            params.is_bf16 = true;
        }
        if (flash_dtype == DATAFLOW_FP8E4M3){
            params.is_e4m3 = true;
        }

        params.total_q = total_tokens;
        params.total_k = total_tokens;
        //params.sink_token_length = 0.0;


        params.q_ptr = x_q;
        params.k_ptr = x_k;
        params.v_ptr = x_v;
        params.o_ptr = x_attn_out;

        params.q_row_stride = model_dim;
        params.k_row_stride = kv_dim;
        params.v_row_stride = kv_dim;
        params.o_row_stride = model_dim;

        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;

        params.v_dim_stride = 1;

        params.cu_seqlens_q = cu_seqlens;
        params.cu_seqlens_k = cu_seqlens;
        
        params.cu_seqlens_knew = NULL;
        params.leftpad_k = NULL;
        params.seqused_q = NULL;
        params.seqused_k = NULL;

        params.knew_ptr = NULL;
        params.vnew_ptr = NULL;

        params.qv_ptr = NULL;

        params.softmax_lse_ptr = softmax_lse;

        params.b = num_seqs;
        params.b_k = num_seqs;
        params.h = num_q_heads;
        params.h_k = num_kv_heads;
        params.d = head_dim;
        params.d_rounded = head_dim;
        params.dv = head_dim;
        params.dv_rounded = head_dim;


        params.seqlen_q = max_seqlen;
        params.seqlen_k = max_seqlen;
        
        params.seqlen_q_rounded = max_seqlen;
        params.seqlen_k_rounded = max_seqlen;

        params.scale_softmax = 1.0 / sqrtf((float) model_dim);
        params.softcap = 0.0;

        params.p_dropout = 1.0;

        params.p_dropout_in_uint8_t = 255;

        params.rp_dropout = 1.0;

        params.is_causal = true;
        params.is_local = false;
        params.window_size_left = -1;
        params.window_size_right = -1;

        params.is_rotary_interleaved = false;
        params.rotary_dim = 0;

        params.num_splits = 1;
        params.pack_gqa = false;

        params.arch = arch;
        params.num_sm = num_sm;

        run_mha_fwd(params, stream);

        return 0;
    }   
}