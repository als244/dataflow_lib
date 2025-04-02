#include "cuda_dataflow_handle.h"
#include "create_host_matrix.h"
#include "dataflow_ops.h"

int main(int argc, char * argv[]){
	
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


	printf("Registering host memory...\n");

	ret = cuda_dataflow_handle.enable_access_to_host_mem(&cuda_dataflow_handle, host_mem, host_size_bytes, 0);
	if (ret){
		fprintf(stderr, "Registration of host memory failed...\n");
		return -1;
	}

	uint64_t fwd_M = 1024;
	uint64_t fwd_K = 256;
	uint64_t fwd_N = 512;

	DataflowDatatype dy_dt = DATAFLOW_FP16;
	DataflowDatatype w_dt = DATAFLOW_FP16;
	DataflowDatatype dx_dt = DATAFLOW_FP16;

	DataflowDatatype compute_dt = DATAFLOW_FP16;

	float alpha = 1.0f;
	float beta = 0.0f;

	uint64_t workspaceBytes = 1UL << 22;

	size_t dy_el_size = dataflow_sizeof_element(dy_dt);
	size_t w_el_size = dataflow_sizeof_element(w_dt);
	size_t dx_el_size = dataflow_sizeof_element(dx_dt);

	uint64_t dy_mat_size = fwd_M * fwd_N * dy_el_size;
	uint64_t w_mat_size = fwd_K * fwd_N * w_el_size;
	uint64_t dx_mat_size = fwd_M * fwd_K * dx_el_size;

	void * dy_matrix = host_mem;
	void * w_matrix = dy_matrix + dy_mat_size;
	void * dx_matrix = w_matrix + w_mat_size;

	printf("Loading in dy and W matrices... (fwd M: %lu, fwd K: %lu, fwd N: %lu, dt: %s)...\n", fwd_M, fwd_K, fwd_N, dataflow_datatype_as_string(dx_dt));

	void * res = load_host_matrix_from_file("test_matmul_bwd/dy_rowmajor.dat", fwd_M, fwd_N, dy_dt, dy_dt, dy_matrix);
	if (!res){
		fprintf(stderr, "Error: could not load in dY matrix...\n");
		return -1;
	}


	res = load_host_matrix_from_file("test_matmul_bwd/w_colmajor.dat", fwd_K, fwd_N, w_dt, w_dt, w_matrix);
	if (!res){
		fprintf(stderr, "Error: could not load in W matrix...\n");
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

	void * d_dy_matrix = dev_mem;
	void * d_w_matrix = d_dy_matrix + dy_mat_size;
	void * d_dx_matrix = d_w_matrix + w_mat_size;

	// need to start at multiple of 256...
	uint64_t init_workspace_start = (uint64_t) (d_dx_matrix + dx_mat_size);
	uint64_t remain = init_workspace_start % 256;
	void * d_workspace = d_dx_matrix + dx_mat_size + (256 - remain);

	printf("Transferring dY, W matrix on host to device of size: %lu and %lu...\n", dy_mat_size, w_mat_size);

	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_dy_matrix, dy_matrix, dy_mat_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_w_matrix, w_matrix, w_mat_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	printf("Syncing with device after transfer...\n");

	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, inbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer...\n");
		return -1;
	}


	printf("Submitting matmul op...!\n");

	// Assume weights are in col-major format.

	// But we want to process activations in row-major

	// Note that matmul interface assumes col-major storage format

	// Also note that FP8 tensor cores only available in TN format

	// During FWD pass we normally want:


	// Thus to compute Y = X @ W, 
	// we can do Y^T = W^T @ X^T
	// where from matmul perspective ^T means we interpret as row-major
	// However we store W as col-major so we need to transpose it.

	// Also for M, K, N (assuming X: (m, k), W (k, n))
	// we set M = n, K = k, N = m

	// The BWD pass is different because if we want dX's to be in row major we need:

	// dX = dY @ W^T
	// => dX^T = W @ dY^T

	// so if we store W in col-major format we shouldn't transpose it...

	// Now for bwd we set
	// M = k, K = n, N = m
	// where m, k, n are from fwd values of X, W, and Y

	int to_trans_a = 0;
	int to_trans_b = 0;

	ret = submit_matmul(&cuda_dataflow_handle, compute_stream_id_a,
						 w_dt, dy_dt, DATAFLOW_NONE, dx_dt, 
						 compute_dt,
						 to_trans_a, to_trans_b,
						 fwd_K, fwd_N, fwd_M,
						 alpha, beta,
						 d_w_matrix, d_dy_matrix, NULL, d_dx_matrix,
						 workspaceBytes, d_workspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit matmul...\n");
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


	printf("Submitting outbound transfer...\n");

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, dx_matrix, d_dx_matrix, dx_mat_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	printf("Syncing with outbound transfer...\n");


	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}


	printf("Saving transformed matrix...\n");

	char * d_matrix_filename = "test_matmul_bwd/dx_matrix.dat";

	ret = save_host_matrix(d_matrix_filename, dx_matrix, fwd_M, fwd_K, dx_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save output matrix...\n");
		return -1;
	}

	printf("\n\n\nSuccessfully Performed Op...!!!\n");

	return 0;
}
