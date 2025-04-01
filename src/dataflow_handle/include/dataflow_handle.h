#ifndef DATAFLOW_HANDLE_H
#define DATAFLOW_HANDLE_H

#include "dataflow_common.h"
#include "table.h"
#include "ops.h"

#define MAX_OPS 65536

#define MAX_STREAMS 256

#define FUNC_NAME_MAX_LEN 1024

typedef enum compute_type {
	COMPUTE_NONE,
	COMPUTE_CPU,
	COMPUTE_CUDA,
	COMPUTE_HSA,
	COMPUTE_LEVEL_ZERO
} ComputeType;


typedef struct func_meta {
	// symbol to reference for setting the launch configuration
	char native_func_lib_name[FUNC_NAME_MAX_LEN];
	char native_func_config_lib_set_attribute_symbol_name[FUNC_NAME_MAX_LEN];
	char native_func_config_lib_set_launch_symbol_name[FUNC_NAME_MAX_LEN];
	// If not native then external should be set
	char external_lib_path[PATH_MAX];
	char external_lib_func_init_symbol[FUNC_NAME_MAX_LEN];
	char external_lib_func_symbol[FUNC_NAME_MAX_LEN];
	// the function arguments of external lib should match
	Op_Skeleton op_skeleton;
} Func_Meta;


typedef struct dataflow_handle Dataflow_Handle;

struct dataflow_handle {

	// Breakdown of core components...
	// Could group these into more compact data structures, 
	// but clearer if each distinct component has its own field
	
	ComputeType compute_type;
	// should be -1 for CPU
	int device_id;
	void * device_handle;
	// backend specific device info maybe needed by attribute setting and launch config...
	void * device_info;
	// user defined id in case multiple handles are created on same device
	int ctx_id;
	// backend specific context handle
	void * ctx;
	// user defined number of streams created at init time
	int num_streams;
	// array of backend specific streams
	void * streams;
	// optional user defined stream priorities and names
	int stream_prios[MAX_STREAMS];
	char * stream_names[MAX_STREAMS];
	// array of backend specific events, with size of num streams created at init time
	// CUevents for cuda
	void * stream_states;

	// space to actually load native module and functions
	void * function_lib;

	// Table containing mapping of op_skeleton.fingerprint => backend specific function info/launch func pointers
	// initially populated using combination of native function lib (pre-compiled into native assembly)
	// and external function pointers from other shared libs
	Table op_table;
	

	// Backend Required Functions...

	
	// 1.) COMPUTE Functionality
	
	int (*submit_op)(Dataflow_Handle * dataflow_handle, Op * op, int stream_id);
	int (*submit_host_op)(Dataflow_Handle * dataflow_handle, void * host_func, void * host_func_arg, int stream_id);


	
	// 2.) DEPENDENCIES Functionality 
	
	// records event and returns a reference to event that can be passed to same/different dataflow handle
	void * (*get_stream_state)(Dataflow_Handle * dataflow_handle, int stream_id);
	int (*submit_dependency)(Dataflow_Handle * dataflow_handle, int stream_id, void * other_stream_state);
	int (*sync_stream)(Dataflow_Handle * dataflow_handle, int stream_id);
	// Synchronizes all streams
	int (*sync_handle)(Dataflow_Handle * dataflow_handle);

	
	// 3.) MEMORY Functionality
	
	// Note: Memory allocs and frees are slow and globally synchronizing, so these should be embedded within
	// a higher layer of memory management. Bulk call to alloc that then can be oragnized elsewhere...
	void * (*alloc_mem)(Dataflow_Handle * dataflow_handle, uint64_t size_bytes);
	void (*free_mem)(Dataflow_Handle * dataflow_handle, void * dev_ptr);
	int (*set_mem)(Dataflow_Handle * dataflow_handle, int stream_id, void * host_ptr, uint8_t val, uint64_t size_bytes);
	// The host pointer should be on same numa node as device
	// ensures good transfer performance by page-locking host memory in way that streaming device driver can understand
	int (*enable_access_to_host_mem)(Dataflow_Handle * dataflow_handle, void * host_ptr, uint64_t size_bytes, unsigned int flags);
	// undoes the enable access step above
	int (*disable_access_to_host_mem)(Dataflow_Handle * dataflow_handle, void * host_ptr);
	// lets this context be able to access memory allocated on peer device
	int (*enable_access_to_peer_mem)(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle);
	int (*disable_access_to_peer_mem)(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle);
	

	// 4.) TRANSFER Functionality
	
	// From/to host memory
	int (*submit_inbound_transfer)(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * host_src, uint64_t size_bytes);
	int (*submit_outbound_transfer)(Dataflow_Handle * dataflow_handle, int stream_id, void * host_dest, void * dev_src, uint64_t size_bytes);
	int (*submit_peer_transfer)(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * dev_src, uint64_t size_bytes);
	
	// TODO: Network
	
};


// this is the function signature of optional init function
// that is called during handle initialization 
// The op_table_value argument can be casted to specific backend function structure (e.g. Cuda_Function)
// which is stored within the op table
// It should be able to hold an "extra" field that can be populated by init function and later retrieved
// during actual call to the external function
typedef int (*External_Lib_Func_Init)(Dataflow_Handle * dataflow_handle, void * op_table_value);

// this is the function signature of all external functions
// the extra argument allows for passing of library handles and other attributes
// the extra argument can be populated from the initialization function and saved within
typedef int (*External_Lib_Func)(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);


int init_dataflow_handle(Dataflow_Handle * dataflow_handle, ComputeType compute_type, int device_id, int ctx_id, unsigned int ctx_flags, int num_streams, int * opt_stream_prios, char ** opt_stream_names, 
							char * all_function_meta_filename, char * native_function_config_lib_filename, char * native_function_lib_filename);




#endif
