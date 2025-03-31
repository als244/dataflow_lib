#include "cuda_dataflow_handle.h"
#include "create_host_matrix.h"
#include "dataflow_ops.h"

int main(int argc, char * argv[]){

	// Seed the random number generator with a constant value
    srand(42);
	
	int ret;

	Dataflow_Handle cuda_dataflow_handle;
	
	ComputeType compute_type = COMPUTE_CUDA;
	int device_id = 0;

	// In case we want to create multiple contexts per device, 
	// higher level can create multiple instances of dataflow handles...
	int ctx_id = 0;
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;

	int num_streams = 8;
	int opt_stream_prios[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	char * opt_stream_names[8] = {"Inbound (a)", "Compute (a)", "Outbound (a)", "Peer (a)", "Inbound (b)", "Compute (b)", "Outbound (b)", "Peer (b)"};
	
	char * all_function_meta_filename = "../../../ops/nvidia/lib/cuda_all_functions_meta.dat";
	char * native_function_config_filename = "../../../ops/nvidia/lib/cuda_kernels_config.so";
	char * native_function_lib_filename = "../../../ops/nvidia/lib/cuda_kernels.cubin";

	ret = init_cuda_dataflow_handle(&cuda_dataflow_handle, compute_type, device_id, 
			ctx_id, ctx_flags, 
			num_streams, opt_stream_prios, opt_stream_names, 
			all_function_meta_filename, native_function_config_filename, native_function_lib_filename); 
	
	if (ret){
		fprintf(stderr, "Error: failed to init cuda dataflow handle...\n");
		return -1;
	}

	int alignment = 4096;

	void * host_mem;

	// 8 GB...
	size_t host_size_bytes = 1UL << 33;

	printf("Allocating host memory of size: %lu...\n", host_size_bytes);

	ret = posix_memalign(&host_mem, alignment, host_size_bytes);
	if (ret){
		fprintf(stderr, "Error: posix memalign failed...\n");
		return -1;
	}
	memset(host_mem, 0, host_size_bytes);


	printf("Registering host memory...\n");

	ret = cuda_dataflow_handle.enable_access_to_host_mem(&cuda_dataflow_handle, host_mem, host_size_bytes, 0);
	if (ret){
		fprintf(stderr, "Registration of host memory failed...\n");
		return -1;
	}



	int num_seqs = 1;

	
	size_t offsets_size = (num_seqs + 1) * sizeof(int);
	size_t lens_size = num_seqs * sizeof(int);

	void * q_seq_offsets = host_mem;
	void * q_seq_lens = q_seq_offsets + offsets_size;

	void * k_seq_offsets = q_seq_lens + lens_size;
	void * k_seq_lens = k_seq_offsets + offsets_size;

	// Harcoding for now
	int q_seqlens[] = {512};
	int total_q = 0;
	int max_seqlen_q = 0;

	
	int kv_seqlens[] = {512};
	int total_kv = 0;
	int max_seqlen_kv = 0;

	int * q_seq_offsets_casted = (int *) q_seq_offsets;
	int * q_seq_lens_casted = (int *) q_seq_lens;
	int * k_seq_offsets_casted = (int *) k_seq_offsets;
	int * k_seq_lens_casted = (int *) k_seq_lens;

	// hardcoding to 
	q_seq_offsets_casted[0] = 0;

	int cur_len = 0;


	// Ensuring to set values properly within pinned buffer
	// to avoid implicit sync during data transfer
	for (int i = 0; i < num_seqs; i++){
		q_seq_offsets_casted[i + 1] = cur_len + q_seqlens[i];
		q_seq_lens_casted[i] = q_seqlens[i];
		if (q_seqlens[i] > max_seqlen_q){
			max_seqlen_q = q_seqlens[i];
		}


		total_q += q_seqlens[i];
		cur_len += q_seqlens[i];
	}

	cur_len = 0;

	k_seq_offsets_casted[0] = 0;

	for (int i = 0; i < num_seqs; i++){
		k_seq_offsets_casted[i + 1] = cur_len + kv_seqlens[i];
		k_seq_lens_casted[i] = kv_seqlens[i];
		if (kv_seqlens[i] > max_seqlen_kv){
			max_seqlen_kv = kv_seqlens[i];
		}
		total_kv += kv_seqlens[i];
	}




	int num_q_heads = 64;
	int num_kv_heads = 8;
	int head_dim = 128;

	int model_dim = num_q_heads * head_dim;
	int kv_dim = num_kv_heads * head_dim;


	DataflowDatatype fwd_dt = DATAFLOW_FP16;

	size_t el_size = dataflow_sizeof_element(fwd_dt);

	DataflowDatatype out_dt = fwd_dt;


	size_t q_size =  (uint64_t) total_q * (uint64_t)  model_dim * (uint64_t)  el_size;
	size_t k_ctx_size = (uint64_t) total_kv * (uint64_t)  kv_dim * (uint64_t)  el_size;
	size_t v_ctx_size = (uint64_t) total_kv * (uint64_t)  kv_dim * (uint64_t)  el_size;

	// if FP8, output will be BF16
	if (out_dt == DATAFLOW_FP8E4M3){
		out_dt = DATAFLOW_BF16;
	}

	size_t out_el_size = dataflow_sizeof_element(out_dt);

	size_t out_size =  (uint64_t) total_q * (uint64_t)  model_dim * (uint64_t)  out_el_size;

	// lse output will be in FP32
	size_t softmax_lse_size = (uint64_t) total_q * (uint64_t) num_q_heads * sizeof(float);


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

	// just get enough...

	size_t max_num_splits = 256;


	size_t attn_workspace_size = 0;

	// cover oaccum and lse accum
	attn_workspace_size += max_num_splits * sizeof(float) * (uint64_t) num_q_heads * (uint64_t) total_q * (uint64_t) (1 + head_dim);
	
	// cover potential tile count sem
	attn_workspace_size += sizeof(int);

	// covert potential dynamic split
	attn_workspace_size += num_seqs * sizeof(int);

	void * x_q = k_seq_lens + lens_size;
	void * x_k = x_q + q_size;
	void * x_v = x_k + k_ctx_size;
	void * x_out = x_v + v_ctx_size;
	void * softmax_lse = x_out + out_size;
	void * attn_workspace = softmax_lse + softmax_lse_size;

	printf("Creating random Q & K & V host matrices (Num Seqs: %d, Total Q: %d, Total KV: %d, Datatype: %s)...\n", num_seqs, total_q, total_kv, dataflow_datatype_as_string(fwd_dt));

	float mean = 0.0;
	float std = 0.006;

	void * res = create_rand_host_matrix(total_q, model_dim, mean, std, fwd_dt, x_q);
	if (!res){
		fprintf(stderr, "Error: creating random X_q host memory matrix failed...\n");
		return -1;
	}

	res = create_rand_host_matrix(total_kv, kv_dim, mean, std, fwd_dt, x_k);
	if (!res){
		fprintf(stderr, "Error: creating random X_k host memory matrix failed...\n");
		return -1;
	}

	res = create_rand_host_matrix(total_kv, kv_dim, mean, std, fwd_dt, x_v);
	if (!res){
		fprintf(stderr, "Error: creating random X_v host memory matrix failed...\n");
		return -1;
	}


	printf("Saving orig seqlen metadata and X_q, X_k, X_v matrices...\n");

	
	char * q_seq_offsets_filename = "test_attention/q_seq_offsets.dat";
	char * q_seq_lens_filename = "test_attention/q_seq_lens.dat";

	char * k_seq_offsets_filename = "test_attention/k_seq_offsets.dat";
	char * k_seq_lens_filename = "test_attention/k_seq_lens.dat";

	char * q_matrix_filename = "test_attention/x_q.dat";
	char * k_matrix_filename = "test_attention/x_k.dat";
	char * v_matrix_filename = "test_attention/x_v.dat";


	ret = save_host_matrix(q_seq_offsets_filename, q_seq_offsets, 1, num_seqs + 1, DATAFLOW_INT);
	if (ret){
		fprintf(stderr, "Error: failed to save q_seq_offsets metadata...\n");
		return -1;
	}

	ret = save_host_matrix(q_seq_lens_filename, q_seq_lens, 1, num_seqs, DATAFLOW_INT);
	if (ret){
		fprintf(stderr, "Error: failed to save q_seq_lens metadata...\n");
		return -1;
	}

	ret = save_host_matrix(k_seq_offsets_filename, k_seq_offsets, 1, num_seqs + 1, DATAFLOW_INT);
	if (ret){
		fprintf(stderr, "Error: failed to save k_seq_offsets metadata...\n");
		return -1;
	}

	ret = save_host_matrix(k_seq_lens_filename, k_seq_lens, 1, num_seqs, DATAFLOW_INT);
	if (ret){
		fprintf(stderr, "Error: failed to save k_seq_lens metadata...\n");
		return -1;
	}

	ret = save_host_matrix(q_matrix_filename, x_q, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save X_q matrix...\n");
		return -1;
	}

	ret = save_host_matrix(k_matrix_filename, x_k, total_kv, kv_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save X_k matrix...\n");
		return -1;
	}

	ret = save_host_matrix(v_matrix_filename, x_v, total_kv, kv_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save X_v matrix...\n");
		return -1;
	}

	// 8 GB...	
	size_t dev_size_bytes = 1UL << 33;

	printf("Allocating device memory of size: %lu...\n", dev_size_bytes);


	void * dev_mem = cuda_dataflow_handle.alloc_mem(&cuda_dataflow_handle, dev_size_bytes);
	if (!dev_mem){
		fprintf(stderr, "Error: device memory allocation failed...\n");
		return -1;
	}




	void * d_q_seq_offsets = dev_mem;
	void * d_q_seq_lens = d_q_seq_offsets + offsets_size;

	void * d_k_seq_offsets = d_q_seq_lens + lens_size;
	void * d_k_seq_lens = d_k_seq_offsets + offsets_size;

	// Required by Tensor Cores
	int dev_alignment = 256;

	int align_spacer = dev_alignment - ((2 * (offsets_size + lens_size)) % dev_alignment);

	void * d_x_q = d_k_seq_lens + lens_size + align_spacer;
	void * d_x_k = d_x_q + q_size;
	void * d_x_v = d_x_k + k_ctx_size;
	void * d_x_out = d_x_v + v_ctx_size;
	void * d_softmax_lse = d_x_out + out_size;
	void * d_attn_workspace = d_softmax_lse + softmax_lse_size;

	printf("Transferring q & k offsets and sizes & Q, K, V, workspace=zero matrices on host to device of sizes -- Num Seqs: %d, Q: %lu, K ctx size: %lu, and V ctx size: %lu, Workspace Size: %lu ...\n", 
								num_seqs, q_size, k_ctx_size, v_ctx_size, attn_workspace_size);

	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_q_seq_offsets, q_seq_offsets, offsets_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for q_seq_offsets...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_q_seq_lens, q_seq_lens, lens_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for q_seq_lens...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_k_seq_offsets, k_seq_offsets, offsets_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for k_seq_offsets...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_k_seq_lens, k_seq_lens, lens_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for k_seq_lens...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_x_q, x_q, q_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for x_q...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_x_k, x_k, k_ctx_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for x_k...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_x_v, x_v, v_ctx_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for x_v...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_attn_workspace, attn_workspace, attn_workspace_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for attn_workspace...\n");
		return -1;
	}

	printf("Syncing with device after transfer...\n");

	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, inbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer...\n");
		return -1;
	}


	printf("Submitting attention op...!\n");

	ret = submit_attention(&cuda_dataflow_handle, compute_stream_id_a,
						 fwd_dt, 
						 num_seqs, total_q, total_kv,
						 d_q_seq_offsets, d_q_seq_lens, max_seqlen_q,
						 d_k_seq_offsets, d_k_seq_lens, max_seqlen_kv,
						 num_q_heads, num_kv_heads, head_dim,
						 d_x_q, d_x_k, d_x_v,
						 d_x_out, d_softmax_lse, 
						 d_attn_workspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention...\n");
		return -1;
	}

	printf("Submitting dependency for outbound transfer...\n");

	void * compute_stream_state = cuda_dataflow_handle.get_stream_state(&cuda_dataflow_handle, compute_stream_id_a);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_dependency(&cuda_dataflow_handle, outbound_stream_id_a, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	printf("Submitting outbound transfers for x_out and softmax_lse of sizes %lu and %lu...\n", out_size, softmax_lse_size);

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, x_out, d_x_out, out_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for x_out...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, softmax_lse, d_softmax_lse, softmax_lse_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for softmax_lse...\n");
		return -1;
	}

	printf("Syncing with outbound transfer...\n");


	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}


	printf("Saving output matrix and softmax lse...\n");

	char * out_matrix_filename = "test_attention/x_out.dat";
	char * softmax_lse_filename = "test_attention/softmax_lse.dat";

	ret = save_host_matrix(out_matrix_filename, x_out, total_q, model_dim, out_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_out matrix...\n");
		return -1;
	}


	ret = save_host_matrix(softmax_lse_filename, softmax_lse, total_q, num_q_heads, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save softmax_lse matrix...\n");
		return -1;
	}

	printf("\n\n\nSuccessfully Performed Op...!!!\n");

	return 0;
}
