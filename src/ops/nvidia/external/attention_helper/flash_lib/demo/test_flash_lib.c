#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

#include <cuda.h>

#include "flash_wrapper.h"

int initialize_driv(){

	CUresult result;
	const char * err;

	unsigned long flags = 0;
	result = cuInit(flags);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not init driver: %s\n", err);
    	return -1;
	}
	return 0;
}

int initialize_ctx(CUcontext * ctx, int device_id){

	CUresult result;
	const char * err;



	CUdevice dev;
	result = cuDeviceGet(&dev, device_id);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not get device: %s\n", err);
    	return -1;
	}


	// Set the host thread to spin waiting for completetion from GPU
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;
	
	result = cuCtxCreate(ctx, ctx_flags, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not create context: %s\n", err);
    	return -1;
	}

	result = cuCtxPushCurrent(*ctx);
        if (result != CUDA_SUCCESS){
                fprintf(stderr, "Error: could not set context...\n");
                return -1;
        }

	// SUCCESS!
	return 0;
}


int initialize_stream(CUstream * stream, int prio){

	CUresult result;
	const char * err;

	
	result = cuStreamCreateWithPriority(stream, CU_STREAM_NON_BLOCKING, prio);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to create cuda stream\n");
		return -1;
	}

	return 0;
}


int reserve_dev_memory(int total_tokens, int num_seqs, size_t x_dt_bytes, int num_q_heads, int num_kv_heads, int head_dim, void ** cu_seqlens, void ** x_q, void  ** x_k, void ** x_v, void ** x_attn_out, void ** softmax_lse, void ** attn_workspace){

	CUresult result;
	const char * err;

	int seqlen_size = (num_seqs + 1) * sizeof(int);

	result = cuMemAlloc((CUdeviceptr *) cu_seqlens, seqlen_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate cu_seqlens on device: %s\n", err);
    	return -1;
	}

	int q_size = num_q_heads * head_dim * total_tokens * x_dt_bytes;

	result = cuMemAlloc((CUdeviceptr *) x_q, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_q on device: %s\n", err);
    	return -1;
	}


	int kv_size = num_kv_heads * head_dim * total_tokens * x_dt_bytes;

	result = cuMemAlloc((CUdeviceptr *) x_k, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_k on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) x_v, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_v on device: %s\n", err);
    	return -1;
	}


	result = cuMemAlloc((CUdeviceptr *) x_attn_out, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_attn_out on device: %s\n", err);
    	return -1;
	}



	int softmax_lse_size = num_q_heads * total_tokens * sizeof(float);

	
	result = cuMemAlloc((CUdeviceptr *) softmax_lse, softmax_lse_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate softmax_lse on device: %s\n", err);
    	return -1;
	}


	int attn_workspace_size = 1 * sizeof(int);

	result = cuMemAlloc((CUdeviceptr *) attn_workspace, attn_workspace_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not allocate attn workspace on device: %s\n", err);
        	return -1;
	}

	int zero_sem = 0;

	result = cuMemcpyHtoD((CUdeviceptr) *attn_workspace, &zero_sem, sizeof(int));
        if (result != CUDA_SUCCESS){
                cuGetErrorString(result, &err);
                fprintf(stderr, "Error: Could not set workspace to zero before attn: %s\n", err);
                return -1;
        }

	return 0;
}


int load_from_file_to_dev(char * filepath, size_t size_bytes, void * dev_ptr){

	CUresult result;
	const char * err;

	FILE * fp = fopen(filepath, "rb");
	if (!fp){
		fprintf(stderr, "Error: could not open file: %s", filepath);
		return -1;
	}

	void * sys_temp = malloc(size_bytes);

	if (!sys_temp){
		fprintf(stderr, "Error: could not alloc temp memory...\n");
		return -1;
	}

	size_t nread = fread(sys_temp, 1, size_bytes, fp);
	if (nread != size_bytes){
		fprintf(stderr, "Error: could not ready expected # of bytes from file %s. Expected %lu, read %lu...\n", filepath, size_bytes, nread);
		return -1;
	}

	fclose(fp);


	result = cuMemcpyHtoD((CUdeviceptr) dev_ptr, sys_temp, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not memcpy contents from file %s (loaded in sys mem) to device: %s\n", err);
    	return -1;
	}

	free(sys_temp);
}


int load_and_copy_sample_inputs(char * data_dir, int num_seqs, int total_tokens, int model_dim, int kv_dim, int dtype_size, void * cu_seqlens, void * x_q, void  * x_k, void * x_v){

	
	char * exts[4];
       	exts[0]	= "cu_seqlens.dat\0";
       	exts[1] = "x_q.dat\0";
	exts[2] = "x_k.dat\0";
	exts[3] = "x_v.dat\0";
	size_t sizes[4];

	sizes[0] = (num_seqs + 1) * sizeof(int);
	sizes[1] = (total_tokens * model_dim * dtype_size);
	sizes[2] = (total_tokens * kv_dim * dtype_size);
	sizes[3] = (total_tokens * kv_dim * dtype_size);

	void * dev_ptrs[4];
	dev_ptrs[0] = cu_seqlens;
	dev_ptrs[1] = x_q;
	dev_ptrs[2] = x_k;
	dev_ptrs[3] = x_v;

	int ret;

	char filepath[PATH_MAX];

	for (int i = 0; i < 4; i++){
		
		sprintf(filepath, "%s/%s", data_dir, exts[i]);

		ret = load_from_file_to_dev(filepath, sizes[i], dev_ptrs[i]);
		if (ret){
			fprintf(stderr, "Error: could not load file: %s to device...\n", filepath);
			return -1;
		}
	}

	return 0;




}


int save_file_from_dev(char * filepath, size_t size_bytes, void * dev_ptr){

	CUresult result;
        const char * err;

	FILE * fp = fopen(filepath, "wb");
	if (!fp){
		fprintf(stderr, "Error: could not open filepath: %s to write...\n", filepath);
	}

	void * sys_temp = malloc(size_bytes);

	result = cuMemcpyDtoH(sys_temp, (CUdeviceptr) dev_ptr, size_bytes);
        if (result != CUDA_SUCCESS){
                cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not memcpy contents to file %s (loaded in sys mem) from device: %s\n", filepath, err);
        	return -1;
        }

	size_t nwritten = fwrite(sys_temp, 1, size_bytes, fp);
	if (nwritten != size_bytes){
		fprintf(stderr, "Error: could not write  expected # of bytes from file %s. Expected %lu, read %lu...\n", filepath, size_bytes, nwritten);
                return -1;
	}

	fclose(fp);

	return 0;

}


int save_flash_lib_out(char * data_dir, int total_tokens, int model_dim, int num_q_heads, size_t x_dt_bytes, void * x_attn_out, void * softmax_lse){
	
	char * exts[2];
	exts[0] = "flash_x_out.dat\0";
	exts[1] = "flash_softmax_lse.dat\0";

	size_t sizes[2];
	sizes[0] = total_tokens * model_dim * x_dt_bytes;
	sizes[1] = total_tokens * num_q_heads * sizeof(float);

	void * dev_ptrs[2];
	dev_ptrs[0] = x_attn_out;
	dev_ptrs[1] = softmax_lse;

	int ret;

        char filepath[PATH_MAX];

        for (int i = 0; i < 2; i++){

                sprintf(filepath, "%s/%s", data_dir, exts[i]);

                ret = save_file_from_dev(filepath, sizes[i], dev_ptrs[i]);
                if (ret){
                        fprintf(stderr, "Error: could not load file: %s to device...\n", filepath);
                        return -1;
                }
        }

	return 0;

}

int main (int argc, char * argv[]){
	
	int ret;

	if (argc != 3){
		fprintf(stderr, "Error Usage: ./test_flash_lib <num_seqs> <seq_len>\n");
		return -1;
	}	

	int num_seqs = atoi(argv[1]);
	int seq_len = atoi(argv[2]);

	char data_dir[PATH_MAX];

	sprintf(data_dir, "/home/as1669/ml_dataflow/flash_lib/demo/data/%dx%d", num_seqs, seq_len);

	ret = initialize_driv();
	if (ret){
		fprintf(stderr, "Error: could not initialize cuda driver...\n");
		return -1;
	}

	CUcontext ctx;

	int device_id = 0;

	ret = initialize_ctx(&ctx, device_id);
	if (ret){
		fprintf(stderr, "Error: could not initialize context...\n");
		return -1;
	}


	CUstream stream;

	ret = initialize_stream(&stream, 0);
	if (ret){
		fprintf(stderr, "Error: could not initialize stream...\n");
		return -1;
	}



	Flash_DType flash_dtype = FLASH_FP16;

	int total_tokens = num_seqs * seq_len;
	int max_seqlen = seq_len;
	size_t x_dt_bytes = 2;
	
	
	int num_q_heads = 64;
	int num_kv_heads = 8;
	int head_dim = 128;	

	void * cu_seqlens;
	void * x_q;
	void * x_k;
	void * x_v;
	void * x_attn_out;
	void * softmax_lse;
	void * attn_workspace;

	ret = reserve_dev_memory(total_tokens, num_seqs, x_dt_bytes, num_q_heads, num_kv_heads, head_dim, &cu_seqlens, &x_q, &x_k, &x_v, &x_attn_out, &softmax_lse, &attn_workspace);
	if (ret){
		fprintf(stderr, "Error: could not reserve device memory...\n");
		return -1;
	}


	int model_dim = head_dim * num_q_heads;
	int kv_dim = head_dim * num_kv_heads;


	ret = load_and_copy_sample_inputs(data_dir, num_seqs, total_tokens, model_dim, kv_dim, x_dt_bytes, cu_seqlens, x_q, x_k, x_v);
	if (ret){
		fprintf(stderr, "Error: could not load and copy sample inputs...\n");
		return -1;
	}


	printf("CALLING FLASH ATTENTION...!\n\n");


	int arch = 90;
	int num_sm = 132;

	for (int i = 0; i < 1; i++) {
		printf("Iter: %d\n", i);
		flash_fwd_wrapper(stream, total_tokens, num_seqs, cu_seqlens, max_seqlen, flash_dtype, x_q, x_k, x_v, x_attn_out, num_q_heads, num_kv_heads, head_dim, softmax_lse, arch, num_sm, attn_workspace);
		printf("Waiting for stream sync...!\n\n");
		cuStreamSynchronize(stream);

		save_flash_lib_out(data_dir, total_tokens, model_dim, num_q_heads, x_dt_bytes, x_attn_out, softmax_lse);
	}



}
