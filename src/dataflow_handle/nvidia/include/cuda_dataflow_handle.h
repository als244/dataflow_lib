#ifndef CUDA_DATAFLOW_HANDLE_H
#define CUDA_DATAFLOW_HANDLE_H

#include "dataflow_common.h"
#include "dataflow_handle.h"
#include "cuda_drv.h"

#define CUDA_DEFAULT_STREAM_PRIO 0

// when querying max threads per multiprocessor, 
// some devices return value of 1536, but then fail to 
// launch kernels because 1024 is true upper bound...!
#define CUDA_DEV_UPPER_BOUND_MAX_THREADS_ALL_FUNC 1024

typedef struct cuda_device_info {
	size_t total_mem;
	int arch_num;
	int sm_count;
	int max_threads_per_block;
	int max_smem_per_block;
	int optin_max_smem_per_block;
	int host_numa_id;
} Cuda_Device_Info;

typedef struct cuda_function_config {
	int func_max_smem;
	int func_max_threads_per_block;
} Cuda_Function_Config;

typedef struct cuda_launch_config {
	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ;
	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ;
	unsigned int sharedMemBytes;
} Cuda_Launch_Config;


typedef struct cuda_function Cuda_Function;

struct cuda_function {
	// Native Attributes
	Cuda_Function_Config function_config;
	// if native the will call native launch
	// otherwise will call the function pointer loaded from external
	bool is_native;
	CUfunction function_handle;
	int (*set_launch_config)(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
	
	// External Attributes
	// external function is defined as a wrapper around third party function...
	int (*cuda_external_func)(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);
	void * op_extra;

	// Op Skeleton should be implementation and vendor-agnostic and follow standard API
	Op_Skeleton op_skeleton;
};


typedef int (*Cuda_Set_Func_Attribute)(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function);


int init_cuda_dataflow_handle(Dataflow_Handle * dataflow_handle, ComputeType compute_type, int device_id, 
								int ctx_id, unsigned int ctx_flags, 
								int num_streams, int * opt_stream_prios, char ** opt_stream_names,
								char * all_function_meta_filename, char * native_function_config_lib_filename, char * native_function_lib_filename);

#endif