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
	
	// char * all_function_meta_filename = "../../../ops/nvidia/lib/cuda_all_functions_meta.dat";
	// char * native_function_config_filename = "../../../ops/nvidia/lib/cuda_kernels_config.so";
	// char * native_function_lib_filename = "../../../ops/nvidia/lib/cuda_kernels.cubin";

	// ret = init_cuda_dataflow_handle(&cuda_dataflow_handle, compute_type, device_id, 
	// 		ctx_id, ctx_flags, 
	// 		num_streams, opt_stream_prios, opt_stream_names, 
	// 		all_function_meta_filename, native_function_config_filename, native_function_lib_filename); 

	ret = init_cuda_dataflow_handle(&cuda_dataflow_handle, compute_type, device_id, 
			ctx_id, ctx_flags, 
			num_streams, opt_stream_prios, opt_stream_names); 
	
	if (ret){
		fprintf(stderr, "Error: failed to init cuda dataflow handle...\n");
		return -1;
	}

	// Register matmul op

	int added_funcs;

	Op_Skeleton matmul_skeletons[1];
	set_external_matmul_skeleton(&matmul_skeletons[0]);


	char * matmul_lib = "../../../ops/nvidia/external/matmul_helper/lib/libmatmulwrapper.so";
	
	char * matmul_symbols[1] = {"cublas_matmul"};

	char * matmul_init_symbols[1] = {"cublas_matmul_init"};

	added_funcs = cuda_dataflow_handle.register_external_code(&cuda_dataflow_handle, matmul_lib, 1, matmul_skeletons, matmul_symbols, matmul_init_symbols);
	if (added_funcs != 1){
		fprintf(stderr, "Error: failed to register matmul op, expected 1 functions, got %d...\n", added_funcs);
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

	

	// Seed the random number generator with a constant value
    srand(42);

	uint64_t M = 2;
	uint64_t K = 4;
	uint64_t N = 3;


	int iM = (int) M;
	int iK = (int) K;
	int iN = (int) N;

	float mean = 0.0;
	float std = 0.006;


	float eps = 1e-5;

	DataflowDatatype a_dt = DATAFLOW_FP16;
	DataflowDatatype b_dt = DATAFLOW_FP16;
	DataflowDatatype c_dt = DATAFLOW_NONE;
	DataflowDatatype d_dt = DATAFLOW_FP16;

	DataflowDatatype compute_dt = DATAFLOW_FP16;

	float alpha = 1.0f;
	float beta = 0.0f;

	uint64_t workspaceBytes = 1UL << 22;

	size_t a_el_size = dataflow_sizeof_element(a_dt);
	size_t b_el_size = dataflow_sizeof_element(b_dt);
	size_t d_el_size = dataflow_sizeof_element(d_dt);

	uint64_t a_mat_size = K * M * a_el_size;
	uint64_t b_mat_size = K * N * b_el_size;
	uint64_t d_mat_size = N * N * d_el_size;

	void * a_matrix = host_mem;
	void * b_matrix = a_matrix + a_mat_size;
	void * d_matrix = b_matrix + b_mat_size;

	printf("Creating random A & B host matrices (M: %lu, K: %lu, N: %lu, dt: %s)...\n", M, K, N, dataflow_datatype_as_string(a_dt));

	/*
	void * res = create_rand_host_matrix(K, M, mean, std, a_dt, a_matrix);
	if (!res){
		fprintf(stderr, "Error: creating random A host memory matrix failed...\n");
		return -1;
	}

	res = create_rand_host_matrix(K, N, mean, std, b_dt, b_matrix);
	if (!res){
		fprintf(stderr, "Error: creating random B host memory matrix failed...\n");
		return -1;
	}
	*/

	void * res = create_index_identity_host_matrix(M, K, a_dt, a_matrix);
	if (!res){
		fprintf(stderr, "Error: creating index identity A host memory matrix failed...\n");
		return -1;
	}

	res = create_index_identity_host_matrix(K, N, b_dt, b_matrix);
	if (!res){
		fprintf(stderr, "Error: creating index identity B host memory matrix failed...\n");
		return -1;
	}


	printf("Saving orig A and B matrix...\n");

	char * a_matrix_filename = "test_matmul/A_matrix.dat";
	char * b_matrix_filename = "test_matmul/B_matrix.dat";

	ret = save_host_matrix(a_matrix_filename, a_matrix, M, K, a_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save A matrix...\n");
		return -1;
	}

	ret = save_host_matrix(b_matrix_filename, b_matrix, K, N, b_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save B matrix...\n");
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

	void * d_a_matrix = dev_mem;
	void * d_b_matrix = d_a_matrix + a_mat_size;
	void * d_d_matrix = d_b_matrix + b_mat_size;
	// need to start at multiple of 256...
	uint64_t init_workspace_start = (uint64_t) (d_d_matrix + d_mat_size);
	uint64_t remain = init_workspace_start % 256;
	void * d_workspace = d_d_matrix + d_mat_size + (256 - remain);

	printf("Transferring A, B matrix on host to device of size: %lu and %lu...\n", a_mat_size, b_mat_size);

	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_a_matrix, a_matrix, a_mat_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_b_matrix, b_matrix, b_mat_size);
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
	

	Op matmul_op;

	set_external_matmul_skeleton(&matmul_op.op_skeleton);

	void ** matmul_op_args = matmul_op.op_args;


	void * d_c_matrix = NULL;
	int num_procs = 0;

	int to_trans_a = 1;
	int to_trans_b = 0;


	matmul_op_args[0] = &num_procs;
	matmul_op_args[1] = &a_dt;
	matmul_op_args[2] = &b_dt;
	matmul_op_args[3] = &c_dt;
	matmul_op_args[4] = &d_dt;
	matmul_op_args[5] = &compute_dt;
	matmul_op_args[6] = &to_trans_a;
	matmul_op_args[7] = &to_trans_b;
	matmul_op_args[8] = &iM;
	matmul_op_args[9] = &iK;
	matmul_op_args[10] = &iN;
	matmul_op_args[11] = &alpha;
	matmul_op_args[12] = &beta;
	matmul_op_args[13] = &d_a_matrix;
	matmul_op_args[14] = &d_b_matrix;
	matmul_op_args[15] = &d_c_matrix;
	matmul_op_args[16] = &d_d_matrix;
	matmul_op_args[17] = &workspaceBytes;
	matmul_op_args[18] = &d_workspace;
	


	ret = cuda_dataflow_handle.submit_op(&cuda_dataflow_handle, &matmul_op, compute_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to submit op...\n");
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

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, d_matrix, d_d_matrix, d_mat_size);
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

	char * d_matrix_filename = "test_matmul/D_matrix.dat";

	ret = save_host_matrix(d_matrix_filename, d_matrix, M, N, d_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save output matrix...\n");
		return -1;
	}

	printf("\n\n\nSuccessfully Performed Op...!!!\n");

	return 0;
}
