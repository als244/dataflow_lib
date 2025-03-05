#include "cuda_dataflow_handle.h"

void CUDA_CB cuda_post_sem_callback(CUstream stream, CUresult status, void * data) {

	sem_t * sem_to_post = (sem_t *) data;

	int ret = sem_post(sem_to_post);
	if (ret){
		fprintf(stderr, "Error: failed to post to semaphore within cuda sem callback...\n");
	}
}


// Can use this instead of callback functionality by launching it as a host function in appropriate stream...
void CUDA_CB cuda_post_sem(void * data) {

	// this serves as the CUhostFn type 
	// within cu_host_func_launch

	sem_t * sem_to_post = (sem_t *) data;

	int ret = sem_post(sem_to_post);
	if (ret){
		fprintf(stderr, "Error: failed to post to semaphore within cuda sem callback...\n");
	}
}



// re-definition from within fingerprint.c to avoid dependency...
static uint64_t cuda_fingerprint_to_least_sig64(uint8_t * fingerprint, int fingerprint_num_bytes){
	uint8_t * least_sig_start = fingerprint + fingerprint_num_bytes - sizeof(uint64_t);
	uint64_t result = 0;
    for(int i = 0; i < 8; i++){
        result <<= 8;
        result |= (uint64_t)least_sig_start[i];
    }
    return result;
}

uint64_t cuda_op_table_hash_func(void * op_fingerprint, uint64_t table_size) {
	uint64_t least_sig_64bits = cuda_fingerprint_to_least_sig64((void *) op_fingerprint, OP_IDENTIFIER_FINGERPRINT_NUM_BYTES);
	return least_sig_64bits % table_size;
}




// FUFILLING ALL FUNCTION POINTERS WIHTIN COMPUTE_HANDLE INTERFACE...


/* 1. COMPUTE FUNCTIONALITY */

int cuda_submit_op(Dataflow_Handle * dataflow_handle, Op * op, int stream_id){

	int ret;

	
	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream;

	if (stream_id == -1){
		stream = NULL;
	}
	else if (stream_id < dataflow_handle -> num_streams){
		stream = cu_streams[stream_id];
	}
	else{
		fprintf(stderr, "Error: cannot submit op with nickname %s to stream id %d, when device only has %d streams. Null stream specified by -1...\n", 
								(op -> op_skeleton).header.op_nickname, stream_id, dataflow_handle -> num_streams);
		return -1;
	}

	// lookup Cuda_Function in table based on op skeleton hash

	Table * op_table = &(dataflow_handle -> op_table);

	Op_Identifier * op_identifier = &((op -> op_skeleton).identifier);

	Cuda_Function * cuda_op_function_ref = NULL;

	long table_ind = find_table(op_table, op_identifier -> fingerprint, false, (void **) &cuda_op_function_ref);

	if ((table_ind == -1) || (!cuda_op_function_ref)){
		fprintf(stderr, "Error: failed to find op with nickname %s and matching dtypes...\n", (op -> op_skeleton).header.op_nickname);
		return -1;
	}


	// now determine if native, or should just call external...

	if (cuda_op_function_ref -> is_native){

		

		// need launch config
		Cuda_Launch_Config cuda_launch_config;

		// if native then launch config function pointer must be set

		// get launch config using function pointer in Cuda Function
		ret = (cuda_op_function_ref -> set_launch_config)(&cuda_launch_config, dataflow_handle, cuda_op_function_ref, op);
		if (ret){
			fprintf(stderr, "Error: failed to correctly set launch config for op with nickname %s...\n", (op -> op_skeleton).header.op_nickname);
			return -1;
		}

		void ** func_params = op -> op_args;

		// for now ignoring extra..
		void ** op_extra = NULL;

		// can error check args here to set kernel params

		// call cuLaunchKernel
		ret = cu_func_launch(stream, cuda_op_function_ref -> function_handle, func_params, 
					cuda_launch_config.sharedMemBytes, 
					cuda_launch_config.gridDimX, cuda_launch_config.gridDimY, cuda_launch_config.gridDimZ, 
					cuda_launch_config.blockDimX, cuda_launch_config.blockDimY, cuda_launch_config.blockDimZ,
					op_extra);

		if (ret){
			fprintf(stderr, "Error: failed to launch kernel for op with nickname %s...\n", (op -> op_skeleton).header.op_nickname);
			return -1;
		}

		return 0;
	}
	
	// otherwise handle external lib function calls...

	if (!(cuda_op_function_ref -> cuda_external_func)){
		fprintf(stderr, "Error: if op is not native it must have a reference to external function (error with op having nickname %s)\n", (op -> op_skeleton).header.op_nickname);
		return -1;
	}
	
	ret = (cuda_op_function_ref -> cuda_external_func)(dataflow_handle, stream_id, op);
	if (ret){
		fprintf(stderr, "Error: failed to do external cuda function for op with nickname %s...\n", (op -> op_skeleton).header.op_nickname);
		return -1;
	}

	return 0;

}


/* 2. DEPENDENCY FUNCTIONALITY */

void * cuda_get_stream_state(Dataflow_Handle * dataflow_handle, int stream_id){
	
	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	CUevent * cu_events = (CUevent *) dataflow_handle -> stream_states;

	CUevent event_to_wait_on = cu_events[stream_id];

	ret = cu_record_event(event_to_wait_on, stream);
	if (ret){
		fprintf(stderr, "Error: failed to record event when trying to get stream state, cannot return ref...\n");
		return NULL;
	}

	return &(cu_events[stream_id]);
}



int cuda_submit_dependency(Dataflow_Handle * dataflow_handle, int stream_id, void * other_stream_state){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	CUevent event_to_wait_on = *((CUevent *) other_stream_state);

	ret = cu_stream_wait_event(stream, event_to_wait_on);
	if (ret){
		fprintf(stderr, "Error: failed to have current stream wait on event during submitting dependency...\n");
		return -1;
	}

	return 0;
}

int cuda_submit_stream_post_sem_callback(Dataflow_Handle * dataflow_handle, int stream_id, sem_t * sem_to_post){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	ret = cu_stream_add_callback(stream, cuda_post_sem_callback, (void *) sem_to_post);
	if (ret){
		fprintf(stderr, "Error: failed to add post callback...\n");
		return -1;
	}

	return 0;
}

int cuda_sync_stream(Dataflow_Handle * dataflow_handle, int stream_id){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	ret = cu_stream_synchronize(stream);
	if (ret){
		fprintf(stderr, "Error: unable ot do cuda stream synchronize...\n");
		return -1;
	}

	return 0;
}


int cuda_sync_ctx(Dataflow_Handle * dataflow_handle){

	int ret;	

	int device_id = dataflow_handle -> device_id;
	int ctx_id = dataflow_handle -> ctx_id;

	ret = cu_ctx_synchronize();
	if (ret){
		fprintf(stderr, "Error: was unable to sync ctx id: %d on device id: %d...\n", ctx_id, device_id);
		return -1;
	}

	return 0;

}


/* 3. MEMORY FUNCTIONALITY */

void * cuda_alloc_mem(Dataflow_Handle * dataflow_handle, uint64_t size_bytes){

	int ret;

	int device_id = dataflow_handle -> device_id;

	void * dev_ptr = NULL;

	ret = cu_alloc_mem(&dev_ptr, size_bytes);
	if (ret || !dev_ptr){
		fprintf(stderr, "Error: failed to alloc memory on device id %d of size %lu bytes...\n", device_id, size_bytes);
		return NULL;
	}

	return dev_ptr;
}

void cuda_free_mem(Dataflow_Handle * dataflow_handle, void * dev_ptr){

	int ret;

	int device_id = dataflow_handle -> device_id;

	ret = cu_free_mem(dev_ptr);
	if (ret){
		fprintf(stderr, "Error: was unable to free memory on device id %d with ptr %p...\n", device_id, dev_ptr);
	}

	return;
}


// PAGE-LOCKS THE HOST MEMORY...use sparingly...
// Makes this context know that host memory is pinned...
int cuda_enable_access_to_host_mem(Dataflow_Handle * dataflow_handle, void * host_ptr, uint64_t size_bytes, unsigned int flags){

	int ret;

	int device_id = dataflow_handle -> device_id;

	ret = cu_register_host_mem(host_ptr, size_bytes, flags);
	if (ret){
		fprintf(stderr, "Error: failed to enable device id %d to access host memory with ptr %p of size %lu and flags %u...\n", device_id, host_ptr, size_bytes, flags);
		return -1;
	}

	return 0;
}

// Releases the page-locked memory that 
int cuda_disable_access_to_host_mem(Dataflow_Handle * dataflow_handle, void * host_ptr){

	int ret;

	int device_id = dataflow_handle -> device_id;

	ret = cu_unregister_host_mem(host_ptr);
	if (ret){
		fprintf(stderr, "Error: failed to disable host memory access with ptr %p from device id %d...\n", host_ptr, device_id);
		return -1;
	}

	return 0;
}


int cuda_enable_access_to_peer_mem(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle){

	int ret;	

	if (peer_dataflow_handle -> compute_type != dataflow_handle -> compute_type){
		fprintf(stderr, "Error: cannot enable peer access, must be same compute type...\n");
		return -1;
	}

	int this_device_id = dataflow_handle -> device_id;
	int other_device_id = dataflow_handle -> device_id;

	CUcontext peer_context = *((CUcontext *) peer_dataflow_handle -> ctx);

	ret = cu_enable_peer_ctx(peer_context);
	if (ret){
		fprintf(stderr, "Error: was unable to allow this device (id %d) to access memory on peer device (id %d)\n", this_device_id, other_device_id);
		return -1;
	}

	return 0;
}


int cuda_disable_access_to_peer_mem(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle){

	int ret;	

	if (peer_dataflow_handle -> compute_type != dataflow_handle -> compute_type){
		fprintf(stderr, "Error: cannot enable peer access, must be same compute type...\n");
		return -1;
	}

	int this_device_id = dataflow_handle -> device_id;
	int other_device_id = dataflow_handle -> device_id;

	CUcontext peer_context = *((CUcontext *) peer_dataflow_handle -> ctx);

	ret = cu_disable_peer_ctx(peer_context);
	if (ret){
		fprintf(stderr, "Error: was unable to disable this device (id %d) from accessing memory on peer device (id %d)\n", this_device_id, other_device_id);
		return -1;
	}

	return 0;
}

/* 4. TRANSFER FUNCTIONALITY */


int cuda_submit_inbound_transfer(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * host_src, uint64_t size_bytes){

	if (stream_id == -1){
		return cu_transfer_host_to_dev_blocking(dev_dest, host_src, size_bytes);
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	return cu_transfer_host_to_dev_async(stream, dev_dest, host_src, size_bytes);
}

int cuda_submit_outbound_transfer(Dataflow_Handle * dataflow_handle, int stream_id, void * host_dest, void * dev_src, uint64_t size_bytes){

	if (stream_id == -1){
		return cu_transfer_dev_to_host_blocking(host_dest, dev_src, size_bytes);
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	return cu_transfer_dev_to_host_async(stream, host_dest, dev_src, size_bytes);
	
}

int cuda_submit_peer_transfer(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * dev_src, uint64_t size_bytes){

	if (stream_id == -1){
		return cu_transfer_dev_to_dev_blocking(dev_dest, dev_src, size_bytes);
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	return cu_transfer_dev_to_dev_async(stream, dev_dest, dev_src, size_bytes);
	
}









int cuda_set_device_info(Cuda_Device_Info * device_info, CUdevice dev){

	int ret;

	ret = cu_get_dev_total_mem(&(device_info -> total_mem), dev);
	if (ret){
		fprintf(stderr, "Error: failed to get total mem when setting device info...\n");
		return -1;
	}

	int major_arch_num;
	int minor_arch_num;

	ret = cu_get_dev_attribute(&major_arch_num, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
	if (ret){
		fprintf(stderr, "Error: failed to get major arch num for cuda device...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&minor_arch_num, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
	if (ret){
		fprintf(stderr, "Error: failed to get major arch num for cuda device...\n");
		return -1;
	}

	device_info -> arch_num = 10 * major_arch_num + minor_arch_num;

	
	ret = cu_get_dev_attribute(&(device_info -> sm_count), dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
	if (ret){
		fprintf(stderr, "Error: failed to get total sm count when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> max_threads_per_block), dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
	if (ret){
		fprintf(stderr, "Error: failed to get max threads per sm when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> max_smem_per_block), dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
	if (ret){
		fprintf(stderr, "Error: failed to get max smem per block when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> optin_max_smem_per_block), dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
	if (ret){
		fprintf(stderr, "Error: failed to get optin max smem per block when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> host_numa_id), dev, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID);
	if (ret){
		fprintf(stderr, "Error: failed to get host numa id when setting device info...\n");
		return -1;
	}

	return 0;
}


int init_cuda_dataflow_handle(Dataflow_Handle * dataflow_handle, ComputeType compute_type, int device_id, 
								int ctx_id, unsigned int ctx_flags, 
								int num_streams, int * opt_stream_prios, char ** opt_stream_names, 
								char * all_function_meta_filename, char * native_function_config_lib_filename, char * native_function_lib_filename) {

	int ret;

	dataflow_handle -> compute_type = compute_type;
	
	// 1.) Ensure cuda has been initialized
	ret = cu_initialize_drv();
	if (ret){
		fprintf(stderr, "Error: failed to initialize cuda driver...\n");
		return -1;
	}

	// 2.) Get device handle
	dataflow_handle -> device_handle = malloc(sizeof(CUdevice));
	if (!dataflow_handle -> device_handle){
		fprintf(stderr, "Error: malloc failed to alloc space for device handle...\n");
		return -1;
	}

	dataflow_handle -> device_id = device_id;

	ret = cu_get_device(dataflow_handle -> device_handle, device_id);
	if (ret){
		fprintf(stderr, "Error: failed to get device handle for dev id: %d...\n", device_id);
		return -1;
	}


	dataflow_handle -> device_info = malloc(sizeof(Cuda_Device_Info));
	if (!dataflow_handle -> device_info){
		fprintf(stderr, "Error: malloc failed to alloc space for compute handle device info...\n");
		return -1;
	}

	ret = cuda_set_device_info(dataflow_handle -> device_info, *((CUdevice *) dataflow_handle -> device_handle));
	if (ret){
		fprintf(stderr, "Error: was unable to set cuda device info for dev id: %d...\n", device_id);
		return -1;
	}



	// 3.) Init context

	dataflow_handle -> ctx_id = ctx_id;

	dataflow_handle -> ctx = malloc(sizeof(CUcontext));
	if (!dataflow_handle -> ctx){
		fprintf(stderr, "Error: malloc failed to allocate space for cucontex container...\n");
		return -1;
	}

	ret = cu_initialize_ctx(dataflow_handle -> ctx, *((CUdevice *) dataflow_handle -> device_handle), ctx_flags);
	if (ret){
		fprintf(stderr, "Error: unable to initialize cuda context for device %d...\n", device_id);
		return -1;
	}



	// 4.) Init streams
	
	dataflow_handle -> num_streams = num_streams;

	// Handle optional arguments...

	// a.) stream priorities
	if (opt_stream_prios){
		memcpy(dataflow_handle -> stream_prios, opt_stream_prios, num_streams * sizeof(int));
	}
	else{
		for (int i = 0; i < num_streams; i++){
			(dataflow_handle -> stream_prios)[i] = CUDA_DEFAULT_STREAM_PRIO;
		}
	}

	for (int i = num_streams; i < MAX_STREAMS; i++){
		(dataflow_handle -> stream_prios)[i] = -1;
	}


	// b.) stream names (for niceness in profiling...)
	size_t stream_name_len;
	if (opt_stream_names){
		for (int i = 0; i < num_streams; i++){
			stream_name_len = strlen(opt_stream_names[i]);
			(dataflow_handle -> stream_names)[i] = malloc(stream_name_len + 1);
			if (!(dataflow_handle -> stream_names)[i]){
				fprintf(stderr, "Error: failed to alloc space to hold stream name...\n");
				return -1;
			}
			strcpy((dataflow_handle -> stream_names)[i], opt_stream_names[i]);
		}
	}
	else{
		for (int i = 0; i < num_streams; i++){
			(dataflow_handle -> stream_names)[i] = NULL;
		}
	}

	for (int i = num_streams; i < MAX_STREAMS; i++){
		(dataflow_handle -> stream_names)[i] = NULL;
	}

	// Actually create the streams + events

	dataflow_handle -> streams = malloc(num_streams * sizeof(CUstream));
	if (!dataflow_handle -> streams){
		fprintf(stderr, "Error: malloc failed to allocate space for custream container...\n");
		return -1;
	}

	dataflow_handle -> stream_states = malloc(num_streams * sizeof(CUevent));
	if (!dataflow_handle -> stream_states){
		fprintf(stderr, "Error: malloc failed to allocate space for cuevent container...\n");
		return -1;
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;
	CUevent * cu_events = (CUevent *) dataflow_handle -> stream_states;

	for (int i = 0; i < num_streams; i++){
		ret = cu_initialize_stream(&(cu_streams[i]), (dataflow_handle -> stream_prios)[i]);
		if (ret){
			fprintf(stderr, "Error: unable to initialize stream #%d on device id %d...\n", i, device_id);
			return -1;
		}

		ret = cu_initialize_event(&(cu_events[i]));
		if (ret){
			fprintf(stderr, "Error: unable to initialize event for stream #%d on device id %d...\n", i, device_id);
			return -1;
		}
	}


	// 5.) Init function lib

	dataflow_handle -> function_lib = malloc(sizeof(CUmodule));
	if (!dataflow_handle -> function_lib){
		fprintf(stderr, "Error: malloc failed to allocate space for cumodule container...\n");
		return -1;
	}

	ret = cu_load_module(dataflow_handle -> function_lib, native_function_lib_filename);
	if (ret){
		fprintf(stderr, "Error: unable to load function lib from path: %s...\n", native_function_lib_filename);
		return -1;
	}


	// 6.) Load the function configuration shared library

	void * config_lib_handle = dlopen(native_function_config_lib_filename, RTLD_LAZY);
	if (!config_lib_handle){
		fprintf(stderr, "Error: could not load function config shared lib with dlopen() from path: %s\n", native_function_config_lib_filename);
		return -1;
	}


	// 7.) Read all of the function metadata info from file...

	FILE * meta_fp = fopen(all_function_meta_filename, "rb");
	if (!meta_fp){
		fprintf(stderr, "Error: could not open function metadata file from path: %s\n", all_function_meta_filename);
		return -1;
	}


	fseek(meta_fp, 0, SEEK_END);
	long meta_file_size = ftell(meta_fp);
	rewind(meta_fp);


	long num_funcs = meta_file_size / sizeof(Func_Meta);


	Func_Meta * function_metadata = malloc(num_funcs * sizeof(Func_Meta));
	if (!function_metadata){
		fprintf(stderr, "Error: malloc failed to allocate space to read in function metadata info...\n");
		return -1;
	}

	size_t nread = fread(function_metadata, sizeof(Func_Meta), num_funcs, meta_fp);
	if (nread != num_funcs){
		fprintf(stderr, "Error: could not read correct amount of functions from func metadata. Expected: %lu, Read: %lu...\n", num_funcs, nread);
		return -1;
	}

	fclose(meta_fp);


	// 7.) Init op table

	Hash_Func hash_func = &cuda_op_table_hash_func;
	uint64_t key_size_bytes = OP_IDENTIFIER_FINGERPRINT_NUM_BYTES;
	uint64_t value_size_bytes = sizeof(Cuda_Function);

	uint64_t min_table_size = 4 * num_funcs;
	uint64_t max_table_size = 4 * num_funcs;

	float load_factor = 1.0f;
	float shrink_factor = 1.0f;

	ret = init_table(&(dataflow_handle -> op_table), hash_func, key_size_bytes, value_size_bytes, min_table_size, max_table_size, load_factor, shrink_factor);
	if (ret){
		fprintf(stderr, "Error: failed to init op table...\n");
		return -1;
	}


	// Handle all of the native functions here...
	
	// Iterate over all of the function metadata, set function attributes, set launch function pointer, and insert into op table...
	
	Cuda_Function cur_cuda_function;
	CUmodule module = *((CUmodule *) dataflow_handle -> function_lib);

	char * native_func_lib_name;

	for (int i = 0; i < num_funcs; i++){

		// copy over the op skeleton
		memcpy(&(cur_cuda_function.op_skeleton), &(function_metadata[i].op_skeleton), sizeof(Op_Skeleton));

		native_func_lib_name = function_metadata[i].native_func_lib_name;

		if (native_func_lib_name[0] != '\0'){

			cur_cuda_function.is_native = true;

			ret = cu_module_get_function(&(cur_cuda_function.function_handle), module, function_metadata[i].native_func_lib_name);
			if (ret){
				fprintf(stderr, "Error: failed to load function #%d with name %s from function lib...\n", i, function_metadata[i].native_func_lib_name);
				return -1;
			}

			// call the set attribute function from shared library with symbol specified by func_config_lib_set_attribute_symbol_name
			if (function_metadata[i].native_func_config_lib_set_attribute_symbol_name[0] != '\0'){
				Cuda_Set_Func_Attribute set_func_attribute = dlsym(config_lib_handle, function_metadata[i].native_func_config_lib_set_attribute_symbol_name);
				if (!set_func_attribute){
					fprintf(stderr, "Error: failed to load symbol to initialize function #%d with name %s and attribute setting function as %s...\n", i, function_metadata[i].native_func_lib_name, function_metadata[i].native_func_config_lib_set_attribute_symbol_name);
				}

				// now call set func attribute
				ret = set_func_attribute(dataflow_handle, &(cur_cuda_function));
				if (ret){
					fprintf(stderr, "Error: failed to set function attributes for function #%d with name %s and attribute setting function as %s\n", i, function_metadata[i].native_func_lib_name, function_metadata[i].native_func_config_lib_set_attribute_symbol_name);
				}
			}


			ret = cu_func_load(cur_cuda_function.function_handle);
			if (ret){
				fprintf(stderr, "Error: could not load cuda function with name %s from function lib...\n", function_metadata[i].native_func_lib_name);
				return -1;
			}


			// now set the pointer for launch config as part of 
			cur_cuda_function.set_launch_config = dlsym(config_lib_handle, function_metadata[i].native_func_config_lib_set_launch_symbol_name);
			if (!cur_cuda_function.set_launch_config){
				fprintf(stderr, "Error: failed to get function pointer to launch config for for function #%d with name %s and set_launch_config function name as %s\n", i, function_metadata[i].native_func_lib_name, function_metadata[i].native_func_config_lib_set_launch_symbol_name);
			}

		
			// now insert the cuda function into the table...

			// the table copies the value
			ret = insert_table(&(dataflow_handle -> op_table), &(cur_cuda_function.op_skeleton.identifier.fingerprint), &cur_cuda_function);
			if (ret){
				fprintf(stderr, "Error: failed to insert op for function #%d with name %s to op table...\n", i, function_metadata[i].native_func_lib_name);
			}

		}
		else{


			// now need to deal with loading external lib and obtaining function pointer ref....


		}


		
	}

	free(function_metadata);

	// SET FUNCTION POINTERS SO COMPUTE HANDLE CAN BE USEFUL...!

	// Compute Functionality
	dataflow_handle -> submit_op = &cuda_submit_op;

	// Dependency Functionality
	dataflow_handle -> get_stream_state = &cuda_get_stream_state;
	dataflow_handle -> submit_dependency = &cuda_submit_dependency;
	dataflow_handle -> submit_stream_post_sem_callback = &cuda_submit_stream_post_sem_callback;
	dataflow_handle -> sync_stream = &cuda_sync_stream;
	dataflow_handle -> sync_ctx = &cuda_sync_ctx;

	// Memory Functionality
	dataflow_handle -> alloc_mem = &cuda_alloc_mem;
	dataflow_handle -> free_mem = &cuda_free_mem;
	dataflow_handle -> enable_access_to_host_mem = &cuda_enable_access_to_host_mem;
	dataflow_handle -> disable_access_to_host_mem = &cuda_disable_access_to_host_mem;
	dataflow_handle -> enable_access_to_peer_mem = &cuda_enable_access_to_peer_mem;
	dataflow_handle -> disable_access_to_peer_mem = &cuda_disable_access_to_peer_mem;

	// Transfer Functionality
	dataflow_handle -> submit_inbound_transfer = &cuda_submit_inbound_transfer;
	dataflow_handle -> submit_outbound_transfer = &cuda_submit_outbound_transfer;
	dataflow_handle -> submit_peer_transfer = &cuda_submit_outbound_transfer;
	
	

	return 0;

}
