#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "set_native_op_skeletons.h"
#include "set_external_op_skeletons.h"

void set_native_func_meta_strings(Func_Meta * func_meta, char * kernel_pre, char * datatype, char * opt_datatype_other, char * kernel_post, char * set_attrib_symb, char * set_launch_symb) {

	if (opt_datatype_other){
		sprintf(func_meta -> native_func_lib_name, "%s_%s_%s_%s", kernel_pre, datatype, opt_datatype_other, kernel_post);
	}
	else{
		sprintf(func_meta -> native_func_lib_name, "%s_%s_%s", kernel_pre, datatype, kernel_post);
	}

	if (set_attrib_symb){
		sprintf(func_meta -> native_func_config_lib_set_attribute_symbol_name, "%s", set_attrib_symb);
	}
	else{
		(func_meta -> native_func_config_lib_set_attribute_symbol_name)[0] = '\0';
	}

	sprintf(func_meta -> native_func_config_lib_set_launch_symbol_name, "%s", set_launch_symb);


	(func_meta -> external_lib_path)[0] = '\0';
	(func_meta -> external_lib_func_init_symbol)[0] = '\0';
	(func_meta -> external_lib_func_symbol)[0] = '\0';
}

void set_external_func_meta_strings(Func_Meta * func_meta, char * external_lib_path, char * external_lib_func_init_symbol, char * external_lib_func_symbol) {

	sprintf(func_meta -> external_lib_path, "%s", external_lib_path);

	// Init function is optional
	if (external_lib_func_init_symbol){
		sprintf(func_meta -> external_lib_func_init_symbol, "%s", external_lib_func_init_symbol);
	}
	else{
		(func_meta -> external_lib_func_init_symbol)[0] = '\0';
	}

	sprintf(func_meta -> external_lib_func_symbol, "%s", external_lib_func_symbol);

	(func_meta -> native_func_lib_name)[0] = '\0';
	(func_meta -> native_func_config_lib_set_attribute_symbol_name)[0] = '\0';
	(func_meta -> native_func_config_lib_set_launch_symbol_name)[0] = '\0';
}



int main(int argc, char * argv[]){

	// for now test, 5 rms_norm fwd, 7 rms_norm bwd x, 7 rms norm bwd w, 5 rope, 3 rope bwd, 5 copy seq to context, 5 swiglu fwd, 7 swiglu bwd, 7 softmax, 3 cross entropy
	int total_native_functions = 54;

	// matmul, attention fwd, attention bwd
	
	int total_external_functions = 1;
	//int total_external_functions = 3;

	int total_functions = total_native_functions + total_external_functions;

	char * kernel_pre_strings[10] = {"rms_norm", "rms_norm_bwd_x", "rms_norm_bwd_w", "rope", "rope_bwd_x", "copy_to_seq_context", "silu_hadamard", "silu_hadamard_bwd_x", "softmax", "cross_entropy_loss"};
	char * kernel_post_strings[10] = {"kernel", "kernel", "kernel", "kernel", "kernel", "kernel", "kernel", "kernel", "kernel", "kernel"};
	char * kernel_set_attribute_symbols[10] = {"rms_norm_set_attribute_config", "rms_norm_set_attribute_config", "rms_norm_set_attribute_config", NULL, NULL, NULL, NULL, NULL, NULL, NULL};
	char * kernel_set_launch_symbols[10] = {"rms_norm_set_launch_config", "rms_norm_bwd_x_set_launch_config", "rms_norm_bwd_w_set_launch_config", 
											"rope_set_launch_config", "rope_set_launch_config", "copy_to_seq_context_set_launch_config", 
											"swiglu_set_launch_config", "swiglu_set_launch_config", 
											"softmax_set_launch_config", "cross_entropy_loss_set_launch_config"};
	
	char * datatype_strings[5] = {"fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2"};
	DataflowDatatype fwd_datatypes[5] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2};

	Func_Meta * all_func_meta = malloc(total_functions * sizeof(Func_Meta));
	memset(all_func_meta, 0, total_functions * sizeof(Func_Meta));

	Func_Meta * cur_func_meta;
	Op_Skeleton * cur_op_skeleton;


	// Native Functions!

	int cur_func_cnt = 0;

	// RMS FORWARD
	int cur_func_type = 0;

	char * cur_op_nickname = malloc(MAX_OP_NICKNAME_SIZE);

	for (int i = 0; i < 5; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_rms_norm_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], NULL, kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	cur_func_type++;

	// RMS BWD X
	for (int i = 0; i < 3; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_rms_norm_bwd_x_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[i], kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	// for fp8 where fwd and back differ
	for (int i = 3; i < 5; i++){
		for (int j = 1; j < 3; j++){
			cur_func_meta = &(all_func_meta[cur_func_cnt]);
			cur_op_skeleton = &(cur_func_meta -> op_skeleton);

			set_native_rms_norm_bwd_x_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[j]);

			set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[j], kernel_post_strings[cur_func_type],
											kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

			cur_func_cnt++;
		}
	}

	cur_func_type++;


	// RMS BWD W
	for (int i = 0; i < 3; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_rms_norm_bwd_w_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[i], kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	// for fp8 where fwd and back differ
	for (int i = 3; i < 5; i++){
		for (int j = 1; j < 3; j++){
			cur_func_meta = &(all_func_meta[cur_func_cnt]);
			cur_op_skeleton = &(cur_func_meta -> op_skeleton);

			set_native_rms_norm_bwd_w_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[j]);

			set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[j], kernel_post_strings[cur_func_type],
											kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

			cur_func_cnt++;
		}
	}

	cur_func_type++;

	// ROPE
	for (int i = 0; i < 5; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_rope_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], NULL, kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	cur_func_type++;

	for (int i = 0; i < 3; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		// takes bwd dtype, but only have fp32, fp16 and bf16 for now
		set_native_rope_bwd_x_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], NULL, kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	cur_func_type++;


	// COPY TO SEQ CONTEXT
	for (int i = 0; i < 5; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_copy_to_seq_context_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], NULL, kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	cur_func_type++;




	// SWIGLU FORWARD
	for (int i = 0; i < 5; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_swiglu_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], NULL, kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	cur_func_type++;

	// SWIGLU BACKWARD

	// where fwd and back are same dtype
	for (int i = 0; i < 3; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_swiglu_bwd_x_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[i], kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	// for fp8 where fwd and back differ
	for (int i = 3; i < 5; i++){
		for (int j = 1; j < 3; j++){
			cur_func_meta = &(all_func_meta[cur_func_cnt]);
			cur_op_skeleton = &(cur_func_meta -> op_skeleton);

			set_native_swiglu_bwd_x_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[j]);

			set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[j], kernel_post_strings[cur_func_type],
											kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

			cur_func_cnt++;
		}
	}

	cur_func_type++;

	// SOFTMAX
	for (int i = 0; i < 3; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_softmax_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[i], kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	// for fp8 where fwd and back differ
	for (int i = 3; i < 5; i++){
		for (int j = 1; j < 3; j++){
			cur_func_meta = &(all_func_meta[cur_func_cnt]);
			cur_op_skeleton = &(cur_func_meta -> op_skeleton);

			set_native_softmax_skeleton(cur_op_skeleton, fwd_datatypes[i], fwd_datatypes[j]);

			set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[j], kernel_post_strings[cur_func_type],
											kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

			cur_func_cnt++;
		}
	}

	cur_func_type++;

	// CROSS ENTROPY LOSS
	for (int i = 0; i < 3; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_cross_entropy_loss_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], NULL, kernel_post_strings[cur_func_type],
										kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

		cur_func_cnt++;
	}

	cur_func_type++;


	// External Functions!

	// MATMUL

	cur_func_meta = &(all_func_meta[cur_func_cnt]);
	cur_op_skeleton = &(cur_func_meta -> op_skeleton);
	set_external_matmul_skeleton(cur_op_skeleton);

	char * cublas_matmul_lib_path = "/home/shein/Documents/grad_school/research/ml_dataflow/dataflow_lib/src/ops/nvidia/external/matmul_helper/build/libmatmulwrapper.so";
	char * cublas_matmul_func_init_symbol = "cublas_matmul_init";
	char * cublas_matmul_func_symbol = "cublas_matmul";

	set_external_func_meta_strings(cur_func_meta, cublas_matmul_lib_path, cublas_matmul_func_init_symbol, cublas_matmul_func_symbol);

	cur_func_cnt++;


	/*
	// ATTENTION FWD

	char * attention_lib_path = "/home/shein/Documents/grad_school/research/ml_dataflow/dataflow_lib/src/ops/nvidia/external/attention_helper/build/libattentionwrapper.so";

	cur_func_meta = &(all_func_meta[cur_func_cnt]);
	cur_op_skeleton = &(cur_func_meta -> op_skeleton);
	set_external_attention_fwd_skeleton(cur_op_skeleton);

	
	char * attention_fwd_func_init_symbol = NULL;
	char * attention_fwd_func_symbol = "flash_attention_fwd";

	set_native_func_meta_strings(cur_func_meta, attention_lib_path, attention_fwd_func_init_symbol, attention_fwd_func_symbol);

	cur_func_cnt++;


	// ATTENTION BWD

	cur_func_meta = &(all_func_meta[cur_func_cnt]);
	cur_op_skeleton = &(cur_func_meta -> op_skeleton);
	set_external_attention_bwd_skeleton(cur_op_skeleton);

	char * attention_bwd_func_init_symbol = NULL;
	char * attention_bwd_func_symbol = "flash_attention_bwd";

	set_native_func_meta_strings(cur_func_meta, attention_lib_path, attention_bwd_func_init_symbol, attention_bwd_func_symbol);

	cur_func_cnt++;
	*/



	// NOW SAVE ALL FUNC META ARRAY....!

	char * all_func_meta_filepath = "../ops/nvidia/build/cuda_all_functions_meta.dat";

	FILE * fp = fopen(all_func_meta_filepath, "wb");
	if (!fp){
		fprintf(stderr, "Error: unable to open file to save all function metadata: %s\n", all_func_meta_filepath);
		return -1;
	}

	size_t n_written = fwrite(all_func_meta, sizeof(Func_Meta), cur_func_cnt, fp);
	if (n_written != cur_func_cnt){
		fprintf(stderr, "Error: failed to write all func meta, expected: %d, wrote: %lu...\n", cur_func_cnt, n_written);
		fclose(fp);
		free(all_func_meta);
		return -1;
	}

	fclose(fp);
	free(all_func_meta);

	printf("Success! Saved %d functions with path: %s\n", cur_func_cnt, all_func_meta_filepath);

	return 0;
}
