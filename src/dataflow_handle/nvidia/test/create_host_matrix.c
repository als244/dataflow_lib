#include "create_matrix.h"


float rand_normal(float mean, float std) {

	if ((mean == 0) && (std == 0)){
		return 0;
	}

    static float spare;
    static int has_spare = 0;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    } else {
        float u, v, s;
        do {
            u = (rand() / (float)RAND_MAX) * 2.0 - 1.0;
            v = (rand() / (float)RAND_MAX) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        s = sqrtf(-2.0 * logf(s) / s);
        spare = v * s;
        has_spare = 1;
        return mean + std * u * s;
    }
}


void * create_zero_host_matrix(uint64_t M, uint64_t N, DataflowDatatype dt, void * opt_dest) {

	int ret;
	
	uint64_t num_els = M * N;
	uint64_t dtype_size = dataflow_sizeof_element(dt);

	void * zero_matrix;
	if (!opt_dest){
		zero_matrix = malloc(num_els * dtype_size);
		if (!zero_matrix){
			fprintf(stderr, "Error: could not allocate zero matrix on host of size %lu...\n", num_els * dtype_size);
			return NULL;
		}
	}
	else{
		zero_matrix = opt_dest;
	}

	memset(zero_matrix, 0, num_els * dtype_size);

	return zero_matrix;
}

void * create_rand_host_matrix(uint64_t M, uint64_t N, float mean, float std, DataflowDatatype dt, void * opt_dest) {

	

	uint64_t num_els = M * N;

	float * rand_float_matrix = malloc(num_els * sizeof(float));

	if (!rand_float_matrix){
		fprintf(stderr, "Error: could not allocate temp random float matrix\n\tM: %lu\n\tN: %lu\n\n", M, N);
		return NULL;
	}

	for (uint64_t i = 0; i < num_els; i++){
		rand_float_matrix[i] = rand_normal(mean, std);
	}

	if (dt == DATAFLOW_FP32){
		if (opt_dest){
			memcpy(opt_dest, rand_float_matrix, num_els * sizeof(float));
			free(rand_float_matrix);
			return opt_dest;
		}
		return rand_float_matrix;
	}


	DataflowConversionType conversion_type;

	switch (dt){
		case DATAFLOW_FP16:
			conversion_type = DATAFLOW_FP32_TO_FP16;
			break;
		case DATAFLOW_BF16:
			conversion_type = DATAFLOW_FP32_TO_BF16;
			break;
		default:
			fprintf(stderr, "Error: cannot create rand host matrix of dtype %d, because no available conversion type from FP32 to datatype with enum %d...\n", dt);
			free(rand_float_matrix);
			return NULL;
	}

	size_t dtype_size = dataflow_sizeof_element(dt);

	void * dest_matrix;

	if (opt_dest){
		dest_matrix = opt_dest;
	}
	else{
		dest_matrix = malloc(num_els * dtype_size);
		if (!dest_matrix){
			fprintf(stderr, "Error: could not allocate space for random matrix of dtype: %lu with \n\tM: %lu\n\tN: %lu\n\n", dtype_size, M, N);
			free(rand_float_matrix);
			return NULL;
		}
	}

	// hardcoding num threads here to be reasonable...

	int num_threads = 8;

	int ret = dataflow_convert_datatype(conversion_type, rand_float_matrix, dest_matrix, (long) (num_els * dtype_size), num_threads);
	
	free(rand_float_matrix);
	
	if (ret){
		fprintf(stderr, "Error: failure in conversion from random float matrix to dest matrix of dtype with enum %d...\n", dt);
		if (opt_dest){
			free(dest_matrix);
		}
		return NULL;
	}

	return dest_matrix;
}

void * load_host_matrix_from_file(char * filepath, uint64_t M, uint64_t N, DataflowDatatype orig_dt, DataflowDatatype new_dt, void * opt_dest) {

	FILE * fp = fopen(filepath, "rb");
	if (!fp){
		fprintf(stderr, "Error: could not load matrix from file: %s\n", filepath);
		return NULL;
	}

	uint64_t num_els = M * N;
	uint64_t orig_dtype_size = dataflow_sizeof_element(orig_dt);

	void * orig_matrix;
	void * dest_matrix;

	if ((orig_dt == dt) && (opt_dest)) {
		orig_matrix = opt_dest;
	}
	else{
		orig_matrix = malloc(num_els * orig_dtype_size);
		if (!orig_matrix){
			fprintf(stderr, "Error: malloc failed to allocate memory for orig matrix (M = %lu, N = %lu)\n", M, N);
			return NULL;
		}
	}

	size_t nread = fread(orig_matrix, orig_dtype_size, num_els, fp);
	if (nread != num_els){
		fprintf(stderr, "Error: couldn't read expected number of elements from matrix file: %s. (Expected %lu, read: %lu)\n", filepath, num_els, nread);
		free(orig_matrix);
		return NULL;
	}

	fclose(fp);

	if (orig_dt == dt){
		return orig_matrix;
	}

	

	
	uint64_t new_dtype_size = sizeof_dtype(dt);

	void * new_matrix = malloc(num_els * new_dtype_size);
	if (!new_matrix){
		fprintf(stderr, "Error: malloc failed to allocate memory for new matrix (M = %lu, N = %lu)\n", M, N);
		free(orig_matrix);
		return NULL;
	}

	void * cur_orig_matrix_el = orig_matrix;
	void * cur_new_matrix_el = new_matrix;

	float orig_val_fp32;
	uint16_t orig_val_fp16;
	uint8_t orig_val_fp8;

	float fp32_upcast;
	
	for (uint64_t i = 0; i < num_els; i++){

		if (orig_dt == FP32){

			orig_val_fp32 = *((float *) cur_orig_matrix_el);

			if (dt == FP16){
				*((uint16_t *) cur_new_matrix_el) = fp32_to_fp16(orig_val_fp32);
			}
			else if (dt == FP8){
				*((uint8_t *) cur_new_matrix_el) = fp32_to_fp8(orig_val_fp32, 4, 3);
			}
			else{
				fprintf(stderr, "Error: dt conversion not supported (from %d to %d)...\n", orig_dt, dt);
				free(orig_matrix);
				free(new_matrix);
				return NULL;
			}
		}
		else if (orig_dt == FP16){

			orig_val_fp16 = *((uint16_t *) cur_orig_matrix_el);

			fp32_upcast = fp16_to_fp32(orig_val_fp16);

			if (dt == FP32){
				*((float *) cur_new_matrix_el) = fp32_upcast;
			}
			else if (dt == FP8){
				*((uint8_t *) cur_new_matrix_el) = fp32_to_fp8(fp32_upcast, 4, 3);
			}
			else{
				fprintf(stderr, "Error: dt conversion not supported (from %d to %d)...\n", orig_dt, dt);
				free(orig_matrix);
				free(new_matrix);
				return NULL;
			}

		}
		else{
			fprintf(stderr, "Error: dt conversion not supported (from %d to %d)...\n", orig_dt, dt);
			free(orig_matrix);
			free(new_matrix);
			return NULL;
		}


		cur_orig_matrix_el += orig_dtype_size;
		cur_new_matrix_el += new_dtype_size;

	}

	free(orig_matrix);

	return new_matrix;
}

