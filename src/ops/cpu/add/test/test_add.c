#include "cpu_add.h"

#define NUM_TO_PRINT 5

int main(int argc, char * argv[]){

	long size = 100000;
	int num_threads = 1;

	float orig_fp32_val = 3.0;

	float * fp32_x_arr = malloc(size * sizeof(float));
	float * fp32_y_arr = malloc(size * sizeof(float));

	float y_start = 0.0f;

	for (long i = 0; i < size; i++){
		fp32_x_arr[i] = 1.0f;
		fp32_y_arr[i] = 2.0f;
	}

	uint16_t * fp16_x_arr = malloc(size * sizeof(uint16_t));
	uint16_t * fp16_y_arr = malloc(size * sizeof(uint16_t));

	for (long i = 0; i < size; i++){
                fp16_x_arr[i] = 0x3C00; // 1.0
                fp16_y_arr[i] = 0x4000; // 2.0
        }

	uint16_t * bf16_x_arr = malloc(size * sizeof(uint16_t));
	uint16_t * bf16_y_arr = malloc(size * sizeof(uint16_t));

	 for (long i = 0; i < size; i++){
                bf16_x_arr[i] = 0x3F80; // 1.0
                bf16_y_arr[i] = 0x4000; // 2.0
        }

	
	float x_scale = 1.0f;

	int ret;

	ret = cpu_add_avx2(DATAFLOW_FP32, fp32_x_arr, fp32_y_arr, size, x_scale, num_threads);
	if (ret){
		fprintf(stderr, "Error: unable to do add fp32...\n");
		return -1;
	}

	ret = cpu_add_avx2(DATAFLOW_FP16, fp16_x_arr, fp16_y_arr, size, x_scale, num_threads);
        if (ret){
                fprintf(stderr, "Error: unable to do add fp16...\n");
                return -1;
        }

	ret = cpu_add_avx2(DATAFLOW_BF16, bf16_x_arr, bf16_y_arr, size, x_scale, num_threads);
        if (ret){
                fprintf(stderr, "Error: unable to do add bf16...\n");
                return -1;
        }

	
	for (long i = 0; i < NUM_TO_PRINT; i++){
                printf("fp32_y_arr[%ld] = %.3f\n", (long) i, fp32_y_arr[i]);
        }

        printf("\n\n\n");

	for (long i = 0; i < NUM_TO_PRINT; i++){
		printf("fp16_y_arr[%ld] = 0x%04X\n", (long) i, fp16_y_arr[i]);
	}

	printf("\n\n\n");

	for (long i = 0; i < NUM_TO_PRINT; i++){
		printf("bf16_y_arr[%ld] = 0x%04X\n", (long) i, bf16_y_arr[i]);
	}

	printf("\n\n\n");

	return 0;
}
