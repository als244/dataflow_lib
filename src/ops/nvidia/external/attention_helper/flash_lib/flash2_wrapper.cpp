#include "flash_wrapper.h"

extern "C" {
	
	/*	
	int flash_fwd_wrapper_v2(CUstream stream, int total_tokens, int num_seqs,  int * cu_seqlens, int max_seqlen, Flash_DType flash_dtype, void * x_q, void * x_k, void * x_v, void * x_attn_out, int num_q_heads, int num_kv_heads, int head_dim, void * softmax_lse, int arch, int num_sm, void * attn_workspace){
	
		float softmax_scale = 1.0 / sqrtf((float) model_dim);
		bool is_causal = true;
		int window_left = -1;
		int window_right = -1;
		int sink_token_length = 0;
		float softcap = 0;
		bool is_rotary_interleaved = false;
		
		{
			at::Tensor t_q = torch::from_blob(x_q, {total_tokens, num_q_heads, head_dim}, torch::kFloat16);
			at::Tensor t_k = torch::from_blob(x_k, {total_tokens, num_kv_heads, head_dim}, torch::kFloat16);
			at::Tensor t_v = torch::from_blob(x_v, {total_tokens, num_kv_heads, head_dim}, torch::kFloat16);
			at::Tensor t_o = torch::from_blob(x_o, {total_tokens, num_q_heads, head_dim}, torch::kFloat16);
			at::Tensor t_cu_seqlens = torch::from_blob(cu_seqlens, {num_seqs + 1}, torch::kInt32);

			std::vector<at::Tensor> result = mha_fwd(t_q, t_k, t_v, NULL, NULL, NULL, t_o, t_cu_seqlens, t_cu_seqlens, NULL, NULL, NULL, max_seqlen, max_seqlen, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, softmax_scale, is_causal, window_left, window_right, softcap, is_rotary_interleaved, 0, NULL, 0);

			at::Tensor out = result[0];
			at::Tensor softmax_lse = result[1];
			at::Tensor out_accum = result[2];
			at::Tensor softmax_se_accum	
		}

		return 0;
	}
	*/



	int flash_fwd_wrapper(CUstream stream, int total_tokens, int num_seqs,  int * cu_seqlens, int max_seqlen, Flash_DType flash_dtype, void * x_q, void * x_k, void * x_v, void * x_attn_out, int num_q_heads, int num_kv_heads, int head_dim, void * softmax_lse, int arch, int num_sm, void * attn_workspace) {

		int model_dim = num_q_heads * head_dim;
		int kv_dim = num_kv_heads * head_dim;

		Flash_fwd_params params;
		
		params.q_descale_ptr = NULL;
		params.k_descale_ptr = NULL;
		params.v_descale_ptr = NULL;
	
		params.q_descale_batch_stride = 0;
		params.q_descale_head_stride = 0;
		params.k_descale_batch_stride = 0;
                params.k_descale_head_stride = 0;
		params.v_descale_batch_stride = 0;
                params.v_descale_head_stride = 0;
		
		params.total_knew = 0;

		params.q_batch_stride = 0;
		params.k_batch_stride = 0;
		params.v_batch_stride = 0;
		params.o_batch_stride = 0;
		

		params.is_fp32 = false;
		params.is_bf16 = false;
		params.is_e4m3 = false;

		if (flash_dtype == FLASH_FP32){
			params.is_fp32 = true;
		}
		if (flash_dtype == FLASH_BF16){
			params.is_bf16 = true;
		}
		if (flash_dtype == FLASH_FP8){
			params.is_e4m3 = true;
		}

		params.total_q = total_tokens;
		params.total_k = total_tokens;
		params.sink_token_length = 0.0;


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
		
		params.knew_batch_stride = 0;
		params.knew_row_stride = 0;
		params.knew_head_stride = 0;
		params.vnew_row_stride = 0;
		params.vnew_head_stride = 0;

			

		params.qv_ptr = NULL;
		params.qv_batch_stride = 0;
		params.qv_row_stride = 0;
		params.qv_head_stride = 0;

		params.kv_batch_idx = 0;
		params.page_table = NULL;
		params.page_table_batch_stride = 0;
		params.page_size = 0;
		params.num_pages = 0;

		params.rng_state = NULL;

		params.oaccum_batch_stride = 0;
		params.oaccum_split_stride = 0;
		params.oaccum_row_stride = 0;
		params.oaccum_head_stride = 0;

		params.lseaccum_batch_stride = 0;
                params.lseaccum_split_stride = 0;
                params.lseaccum_head_stride = 0;


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
		params.softcap = 0.0f;

		params.p_dropout = 1.0f;

		params.p_dropout_in_uint8_t = (uint8_t) 255;

		params.rp_dropout = 1.0f;
	
		params.is_causal = true;
		params.is_local = false;
		params.window_size_left = -1;
		params.window_size_right = -1;
		
		params.rotary_cos_ptr = NULL;
		params.rotary_sin_ptr = NULL;
		params.is_rotary_interleaved = false;
		params.rotary_dim = 0;

		params.arch = arch;
		params.num_sm = num_sm;
		
		int tile_size_m;
		int tile_size_n;

		if ((flash_dtype == FLASH_FP16) || (flash_dtype == FLASH_BF16)){
		
			if (head_dim <= 64){
				tile_size_m = 192;
				tile_size_n = 192;
			}
			else if (head_dim <= 96){
				tile_size_m = 192;
				tile_size_n = 144;
			}
			else if (head_dim <= 128){
				tile_size_m = 128;
				tile_size_n = 128;
			}
			else if (head_dim <= 192){
				tile_size_m = 128;
                                tile_size_n = 112;
			}
			else{
				tile_size_m = 128;
                                tile_size_n = 80;
			}
		}
		else{
			if (head_dim <= 64){
                                tile_size_m = 192;
                                tile_size_n = 160;
                        }
                        else if (head_dim <= 96){
                                tile_size_m = 192;
                                tile_size_n = 128;
                        }
                        else if (head_dim <= 128){
                                tile_size_m = 128;
                                tile_size_n = 224;
                        }
                        else if (head_dim <= 192){
                                tile_size_m = 128;
                                tile_size_n = 160;
                        }
                        else{
                                tile_size_m = 128;
                                tile_size_n = 128;
                        }
		}

		int ratio = params.h / params.h_k;
		int seqlen_q_packgqa = max_seqlen * ratio;

		int num_n_blocks = (max_seqlen + tile_size_n - 1) / tile_size_n;
		int num_m_blocks = (seqlen_q_packgqa + tile_size_m - 1) / tile_size_m;

		int batch_nheads_mblocks = num_seqs * params.h_k * num_m_blocks;
		
		int num_splits = 1;

		float best_eff = 0;
		int best_split = 1;
		float eff;
		float n_waves;
		if (batch_nheads_mblocks < 0.8f * num_sm){
			
			for (int i = 1; i < 128; i++){
				if (i == 1 || (ceilf((float) num_n_blocks / (float) i) != ceilf((float) num_n_blocks / (float) (i - 1)))) {
					n_waves = (float) (batch_nheads_mblocks * i) / (float) num_sm;
					eff = n_waves / ceilf(n_waves);
					if (eff > best_eff){
						best_eff = eff;
						best_split = i;
					}
				}			
			}
			for (int i = 1; i <= best_split; i++){
                                if (i == 1 || (ceilf((float) num_n_blocks / (float) i) != ceilf((float) num_n_blocks / (float) (i - 1)))) {
                                        n_waves = (float) (batch_nheads_mblocks * i) / (float) num_sm;
                                        eff = n_waves / ceilf(n_waves);
                                        if (eff >= .85f * best_eff){
						num_splits = i;
						break;
                                        }
                                }
                        }
		}

		params.num_splits = num_splits;
		//params.pack_gqa = get_pack_gqa(params);
		
		// ASSUMING GQA for now
		params.pack_gqa = true;

		float * cur_attn_workspace = (float *) attn_workspace;
	
		if (params.num_splits > 1){
			
			// (num_splits, num_heads, total_q, headdim)
			params.oaccum_ptr = (float *) cur_attn_workspace;
			cur_attn_workspace += (num_splits * num_q_heads * total_tokens * head_dim);
			
			params.oaccum_split_stride = params.num_splits;
			params.oaccum_row_stride = model_dim;
			params.oaccum_head_stride = head_dim;

			// (num_splits, num_heads, total_q)
			params.softmax_lseaccum_ptr = (float *) cur_attn_workspace;
			cur_attn_workspace += (num_splits * num_q_heads * total_tokens);
			params.lseaccum_split_stride = params.num_splits;
			params.lseaccum_head_stride = head_dim;
		}
	
		
		params.tile_count_semaphore = NULL;

		if (arch >= 90){
			params.tile_count_semaphore = (int *) cur_attn_workspace;
		}

		run_mha_fwd(params, stream);

		if (num_splits > 1){
			params.b = 1;
			params.seqlen_q = total_tokens;
			run_mha_fwd_combine(params, stream);
		}
		
		return 0;
	}	

}
