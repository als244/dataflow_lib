/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include "cutlass/numeric_types.h"

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"


#define ROUND_UP_TO_128(x) (((x) + 127) & ~127)


inline int round_up_headdim(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}


inline bool get_pack_gqa(Flash_fwd_params const& params) {
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

inline int get_num_splits(Flash_fwd_params const& params) {
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
        bool enable_pdl = false;
        if (params.arch >= 90){
            enable_pdl = true;
        }
        run_mha_fwd_combine(params, stream, enable_pdl);
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

extern "C" {
    
    // if TYPE FP8, output must be BF16
    
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

    int flash3_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * cum_q_seqlens, int max_seqlen_q,
                        int * k_seqlens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, void * softmax_lse,
                        void * attn_workspace) {

        int model_dim = num_q_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        Flash_fwd_params params;
        

        params.is_fp32 = false;
        params.is_bf16 = false;
        params.is_e4m3 = false;

        DataflowDatatype flash_dt = (DataflowDatatype) flash_dtype_as_int;

        if (flash_dt == DATAFLOW_FP32){
            params.is_fp32 = true;
        }
        else if (flash_dt == DATAFLOW_BF16){
            params.is_bf16 = true;
        }
        else if (flash_dt == DATAFLOW_FP8E4M3){
            params.is_e4m3 = true;
        }
        else{
            if (flash_dt != DATAFLOW_FP16){
                fprintf(stderr, "Error: dtype of DataflowDatatype enum val of %d not supported in flash3...\n", flash_dtype_as_int);
                return -1;
            }
        }

        params.total_q = total_q;
        params.total_k = total_k;
        params.total_knew = 0;

        params.seqlen_q = max_seqlen_q;
        params.seqlen_q_rounded = ROUND_UP_TO_128(max_seqlen_q); 

        // Think it is ok to set this 0 and not take in max_seqlen_k...
        params.seqlen_k = max_seqlen_k;
        params.seqlen_k_rounded = ROUND_UP_TO_128(max_seqlen_k);

        params.seqlen_knew = 0;

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

        params.cu_seqlens_q = cum_q_seqlens;
        params.cu_seqlens_k = NULL;
        params.cu_seqlens_knew = NULL;
        params.leftpad_k = NULL;


        params.seqused_q = NULL;
        params.seqused_k = k_seqlens;

        params.knew_ptr = NULL;
        params.vnew_ptr = NULL;
        
        params.knew_batch_stride = 0;
        params.knew_row_stride = 0;
        params.knew_head_stride = 0;
        params.vnew_row_stride = 0;
        params.vnew_head_stride = 0;
        params.vnew_batch_stride = 0;

        params.q_descale_ptr = NULL;
        params.k_descale_ptr = NULL;
        params.v_descale_ptr = NULL;
    
        params.q_descale_batch_stride = 0;
        params.q_descale_head_stride = 0;
        params.k_descale_batch_stride = 0;
        params.k_descale_head_stride = 0;
        params.v_descale_batch_stride = 0;
        params.v_descale_head_stride = 0;

        params.q_batch_stride = 0;
        params.k_batch_stride = 0;
        params.v_batch_stride = 0;
        params.o_batch_stride = 0;

            

        params.qv_ptr = NULL;
        params.qv_batch_stride = 0;
        params.qv_row_stride = 0;
        params.qv_head_stride = 0;

        params.kv_batch_idx = NULL;
        params.page_table = NULL;
        params.page_table_batch_stride = 0;
        params.page_size = 0;
        params.num_pages = 0;
        params.pagedkv_tma = false;

        // Need to determine what to do here
        // (if dropout is non-zero...)
        params.rng_state = NULL;    

        // Will over-write if split
        params.oaccum_batch_stride = 0;
        params.oaccum_split_stride = 0;
        params.oaccum_row_stride = 0;
        params.oaccum_head_stride = 0;

        params.lseaccum_batch_stride = 0;
        params.lseaccum_split_stride = 0;
        params.lseaccum_head_stride = 0;


        params.softmax_lse_ptr = softmax_lse;

        int head_dim_rounded = round_up_headdim(head_dim);
        
        params.b = num_seqs;
        params.b_k = num_seqs;
        params.h = num_q_heads;
        params.h_k = num_kv_heads;
        params.d = head_dim;
        params.d_rounded = head_dim_rounded;
        params.dv = head_dim;
        params.dv_rounded = head_dim_rounded;

        params.scale_softmax = 1.0 / sqrtf((float) head_dim);
        params.softcap = 0.0f;

        params.p_dropout = 1.0f;

        params.p_dropout_in_uint8_t = (uint8_t) 255;

        params.rp_dropout = 1.0f;
    
        params.is_causal = true;
        params.is_local = false;
        params.window_size_left = -1;
        params.window_size_right = -1;
        
        params.rotary_dim = 0;
        params.rotary_cos_ptr = NULL;
        params.rotary_sin_ptr = NULL;
        params.seqlens_rotary = NULL;
        params.is_rotary_interleaved = false;
        

        params.arch = arch;
        params.num_sm = num_sm;


        void * cur_attn_workspace = attn_workspace;

        // FOLLOWING WHAT WAS DONE IN ORGINAL SOURCE...

        params.num_splits_dynamic_ptr = (int *) 1;

        int num_splits = get_num_splits(params);
        params.num_splits = num_splits;
        
        if (params.num_splits > 1){
            
            // (num_splits, num_heads, total_q, headdim)
            params.oaccum_ptr = (float *) cur_attn_workspace;
            cur_attn_workspace += (num_splits * num_q_heads * total_q * head_dim * sizeof(float));
            
            params.oaccum_split_stride = params.num_splits;
            params.oaccum_row_stride = model_dim;
            params.oaccum_head_stride = head_dim;

            // (num_splits, num_heads, total_q)
            params.softmax_lseaccum_ptr = (float *) cur_attn_workspace;
            cur_attn_workspace += (num_splits * num_q_heads * total_q * sizeof(float));
            params.lseaccum_split_stride = params.num_splits;
            params.lseaccum_head_stride = head_dim;
        }

        
        params.pack_gqa = get_pack_gqa(params);

        int to_use_dynamic_split = 0;

        // Harcoded number from original source
        if (params.b <= 992){
            to_use_dynamic_split = 1;
        } 

        int needs_sem = 0;
        if ((params.arch >= 90) || (params.num_splits > 1)){
            needs_sem = 1;
        }


        params.tile_count_semaphore = NULL;
        
        // reset back to null now
        params.num_splits_dynamic_ptr = NULL;
        
        if ((needs_sem) || (to_use_dynamic_split)) {
            if (needs_sem){
                if (!to_use_dynamic_split){
                    // ensure tile count semaphore set to zero
                    // should happen before call to this function
                }
                // only 1 int
                params.tile_count_semaphore = (int *) cur_attn_workspace;
                cur_attn_workspace += sizeof(int);
            }

            if (to_use_dynamic_split){
                // need params.b integers here or params.b - 1..?
                // is this +1 a bug if the sched doesn't need sem...?
                // they initialzed buffer as needs_sem + use_dynamic * params.b
                // assuming bug...
                // params.num_splits_dynamic_ptr = ((int *) cur_attn_workspace + 1);
                params.num_splits_dynamic_ptr = (int *) cur_attn_workspace;

            }
        }

        

        // copying from Original source...
        if (params.num_splits_dynamic_ptr){

            auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
            auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, params.num_splits > 1, params.softcap > 0.f, params.knew_ptr);
            int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
            int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
            prepare_varlen_num_blocks(params, stream, params.pack_gqa, kBlockM, kBlockN, false /*enable_pdl*/);
            CHECK_CUDA_KERNEL_LAUNCH();
        }


        // ^ did sched metadata above
        params.skip_scheduler_metadata_computation = true;

        // Also calls combine at end of function if 
        // num_splits > 1
        run_mha_fwd(params, stream);

        return 0;
    }

    int flash3_bwd_wrapper(CUstream stream, 
                            int num_seqs, int total_q, int total_k, 
                            int * cum_q_seqlens, int max_seqlen_q,
                            int * k_seqlens, int max_seqlen_k,
                            int flash_dtype_as_int, 
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, void * softmax_lse, 
                            int arch, int num_sm, 
                            void * attn_workspace) {
        
        fprintf(stderr, "Unimplemented Error: flash3_bwd_wrapper\n");
        return -1;

    }
}