#include "cuda_dataflow_handle.h"
#include "create_host_matrix.h"

int main(int argc, char * argv[]){
	
	int ret;

	Dataflow_Handle cuda_dataflow_handle;
	
	ComputeType compute_type = COMPUTE_CUDA;
	int device_id = 0;

	// In case we want to create multiple contexts per device, 
	// higher level can create multiple instances of compute handles...
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
		fprintf(stderr, "Error: failed to init cuda compute handle...\n");
		return -1;
	}

	printf("Success! Created cuda compute handle...\n");

}
