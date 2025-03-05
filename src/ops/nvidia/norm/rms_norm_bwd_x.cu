#include "nvidia_ops.h"


extern "C" __global__ void rms_norm_bwd_x_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * rms_weight, float * X_inp, float * sq_sums, float * upstream_dX, float * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * rms_weight[i];
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = X_inp[row_ind_start + i];
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * deriv;

		}
	}
}


// Here dX is (N, model_dim) and contains the backprop loss flow that we will update in-place
// This needs to be called after the bwd_weight because the weight we use the updstream dL/dX and this function will
// modify the same pointer...
extern "C" __global__ void rms_norm_bwd_x_fp16_fp16_kernel(int n_rows, int n_cols, float eps, __half * rms_weight, __half * X_inp, float * sq_sums, __half * upstream_dX, __half * dX){
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * __half2float(rms_weight[i]);
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __half2float(X_inp[row_ind_start + i]);
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * __float2half(deriv);

		}
	}
}


extern "C" __global__ void rms_norm_bwd_x_bf16_bf16_kernel(int n_rows, int n_cols, float eps, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X_inp, float * sq_sums, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * __bfloat162float(rms_weight[i]);
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __bfloat162float(X_inp[row_ind_start + i]);
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * __float2bfloat16(deriv);

		}
	}
}


extern "C" __global__ void rms_norm_bwd_x_fp8e4m3_fp16_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X_inp, float * sq_sums, __half * upstream_dX, __half * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * float(rms_weight[i]);
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * __float2half(deriv);

		}
	}
}

extern "C" __global__ void rms_norm_bwd_x_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X_inp, float * sq_sums, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * float(rms_weight[i]);
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * __float2bfloat16(deriv);

		}
	}
}


extern "C" __global__ void rms_norm_bwd_x_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X_inp, float * sq_sums, __half * upstream_dX, __half * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * float(rms_weight[i]);
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * __float2half(deriv);

		}
	}
}

extern "C" __global__ void rms_norm_bwd_x_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X_inp, float * sq_sums, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * weights_scaled = (float *) sdata;

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_sq_sums = (float *) (weights_scaled + n_cols); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	
	int thread_id = threadIdx.x;

	float dim_scale = rsqrt((float) n_cols); 

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights_scaled[i] = dim_scale * float(rms_weight[i]);
	}

	// retrieve back the recip squared avgs
	for (uint64_t i = thread_id; i < n_rows; i+=blockDim.x){
		shared_sq_sums[i] = sq_sums[i];
	}

	__syncthreads();

	float deriv;
	float cur_sq_sum;
	float cur_sq_sum_rsqrt;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_sq_sum = shared_sq_sums[row_id];
		cur_sq_sum_rsqrt = rsqrtf(cur_sq_sum);
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (weights_scaled[i] * (cur_sq_sum - (inp_val * inp_val)) * cur_sq_sum_rsqrt) / cur_sq_sum;

			// now update dX
			dX[row_id * n_cols + i] = upstream_dX[row_id * n_cols + i] * __float2bfloat16(deriv);

		}
	}
}