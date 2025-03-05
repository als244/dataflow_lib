#include "cuda_dataflow_handle.h"
#include "create_host_matrix.h"
#include "fingerprint.h"


void create_rms_norm_op_skeleton(char * op_nickname, Op * op, DataflowDatatype datatype){

	int num_args = 7;

	Op_Skeleton * skeleton = &(op -> op_skeleton);

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);
	
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 


	skeleton_header -> num_args = 7;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = datatype;
	arg_dtypes[4] = datatype;
	arg_dtypes[5] = datatype;
	arg_dtypes[6] = DATAFLOW_FP32;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}


	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

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
	
	char * all_function_meta_filename = "/home/shein/Documents/grad_school/research/ml_dataflow/dataflow_lib/src/ops/nvidia/build/cuda_all_functions_meta.dat";
	char * native_function_config_filename = "/home/shein/Documents/grad_school/research/ml_dataflow/dataflow_lib/src/ops/nvidia/build/cuda_kernels_config.so";
	char * native_function_lib_filename = "/home/shein/Documents/grad_school/research/ml_dataflow/dataflow_lib/src/ops/nvidia/build/cuda_kernels.cubin";

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

	

	// Seed the random number generator with a constant value
    srand(42);

	uint64_t M = 16384;
	uint64_t N = 8192;

	float mean = 0.0;
	float std = 0.006;

	DataflowDatatype dt = DATAFLOW_FP16;

	size_t el_size = dataflow_sizeof_element(dt);
	uint64_t mat_size = M * N * el_size;

	void * orig_matrix = host_mem;
	void * out_matrix = orig_matrix + mat_size;
	void * rms_weight = out_matrix + mat_size;
	void * sq_sums = rms_weight + (el_size * N);

	printf("Creating random host matrix (M: %lu, N; %lu, dt: %s)...\n", M, N, dataflow_datatype_as_string(dt));

	void * res = create_rand_host_matrix(M, N, mean, std, dt, orig_matrix);
	if (!res){
		fprintf(stderr, "Error: creating random host memory matrix failed...\n");
		return -1;
	}

	res = create_rand_host_matrix(N, 1, mean, std, dt, rms_weight);
	if (!res){
		fprintf(stderr, "Error: creating random host memory matrix failed...\n");
		return -1;
	}

	printf("Saving orig matrix...\n");

	char * orig_matrix_filename = "test_data/orig_matrix.dat";
	char * rms_weight_filename = "test_data/weights.dat";

	ret = save_host_matrix(orig_matrix_filename, orig_matrix, M, N, dt);
	if (ret){
		fprintf(stderr, "Error: failed to save original matrix...\n");
		return -1;
	}

	ret = save_host_matrix(rms_weight_filename, rms_weight, N, 1, dt);
	if (ret){
		fprintf(stderr, "Error: failed to save original matrix...\n");
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

	void * d_orig_matrix = dev_mem;
	void * d_out_matrix = d_orig_matrix + mat_size;
	void * d_rms_weight = d_out_matrix + mat_size;
	void * d_sq_sums = d_rms_weight + (el_size * N);


	printf("Transferring matrix on host to device of size: %lu...\n", mat_size);

	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_orig_matrix, orig_matrix, mat_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_rms_weight, rms_weight, el_size * N);
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



	printf("Submitting RMS norm op...!\n");

	Op rms_norm_op;

	char * rms_nickname = "rms_norm_fp16";

	create_rms_norm_op_skeleton(rms_nickname, &rms_norm_op, dt);

	void ** op_args = rms_norm_op.op_args;

	float eps = 1e-5;

	int iM = (int) M;
	int iN = (int) N;

	op_args[0] = &iM;
	op_args[1] = &iN;
	op_args[2] = &eps;
	op_args[3] = &d_rms_weight;
	op_args[4] = &d_orig_matrix;
	op_args[5] = &d_out_matrix;
	op_args[6] = &d_sq_sums;


	ret = cuda_dataflow_handle.submit_op(&cuda_dataflow_handle, &rms_norm_op, compute_stream_id_a);
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

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, out_matrix, d_out_matrix, mat_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, sq_sums, d_sq_sums, M * el_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	printf("Syncing wiht outbound transfer...\n");


	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}


	printf("Saving transformed matrix...\n");

	char * out_matrix_filename = "test_data/out_matrix.dat";
	char * sq_sums_filename = "test_data/sq_sums.dat";

	ret = save_host_matrix(out_matrix_filename, out_matrix, M, N, dt);
	if (ret){
		fprintf(stderr, "Error: failed to save output matrix...\n");
		return -1;
	}

	printf("Saving extra data returned from op...\n");

	ret = save_host_matrix(sq_sums_filename, sq_sums, M, 1, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save sq sums matrix...\n");
		return -1;
	}


	printf("\n\n\nSuccessfully Performed Op...!!!\n");




	return 0;
}
