#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dataflow.h"
#include "fingerprint.h"
#include "ops.h"
#include "dataflow_handle.h"



void set_native_rms_norm_skeleton(Op_Skeleton * skeleton, DataflowDatatype datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);
	
	int num_args = 7;

	skeleton_header -> num_args = num_args;

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
}

void set_native_swiglu_skeleton(Op_Skeleton * skeleton, DataflowDatatype datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	int num_args = 5;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = datatype;
	arg_dtypes[3] = datatype;
	arg_dtypes[4] = datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

}

void set_native_swiglu_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	int num_args = 7;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = fwd_datatype;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = bwd_datatype;
	arg_dtypes[5] = bwd_datatype;
	arg_dtypes[6] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

}

// Assumes skeleton -> header has been fully filled out...
void set_op_identifier(char * op_nickname, Op_Skeleton * skeleton){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);
	
	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}


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
	(func_meta -> external_lib_func_symbol)[0] = '\0';
}


int main(int argc, char * argv[]){

	// for now test, 5 rms_norm fwd, 5 swiglu fwd, 7 swiglu bwd
	int total_functions = 17;

	char * kernel_pre_strings[3] = {"rms_norm", "silu_hadamard", "silu_hadamard_bwd_x"};
	char * kernel_post_strings[3] = {"kernel", "kernel", "kernel"};
	char * kernel_set_attribute_symbols[3] = {"rms_norm_set_attribute_config", NULL, NULL};
	char * kernel_set_launch_symbols[3] = {"rms_norm_set_launch_config", "swiglu_set_launch_config", "swiglu_set_launch_config"};
	
	char * datatype_strings[5] = {"fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2"};
	DataflowDatatype fwd_datatypes[5] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2};

	Func_Meta * all_func_meta = malloc(total_functions * sizeof(Func_Meta));
	memset(all_func_meta, 0, total_functions * sizeof(Func_Meta));

	Func_Meta * cur_func_meta;
	Op_Skeleton * cur_op_skeleton;


	int cur_func_cnt = 0;

	// RMS FORWARD
	int cur_func_type = 0;

	char * cur_op_nickname = malloc(MAX_OP_NICKNAME_SIZE);

	for (int i = 0; i < 5; i++){
		cur_func_meta = &(all_func_meta[cur_func_cnt]);
		cur_op_skeleton = &(cur_func_meta -> op_skeleton);

		set_native_rms_norm_skeleton(cur_op_skeleton, fwd_datatypes[i]);

		sprintf(cur_op_nickname, "%s_%s", kernel_pre_strings[cur_func_type], datatype_strings[i]);

		set_op_identifier(cur_op_nickname, cur_op_skeleton);

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

		sprintf(cur_op_nickname, "%s_%s", kernel_pre_strings[cur_func_type], datatype_strings[i]);

		set_op_identifier(cur_op_nickname, cur_op_skeleton);

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

		sprintf(cur_op_nickname, "%s_%s_%s", kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[i]);

		set_op_identifier(cur_op_nickname, cur_op_skeleton);

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

			sprintf(cur_op_nickname, "%s_%s_%s", kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[j]);

			set_op_identifier(cur_op_nickname, cur_op_skeleton);

			set_native_func_meta_strings(cur_func_meta, kernel_pre_strings[cur_func_type], datatype_strings[i], datatype_strings[j], kernel_post_strings[cur_func_type],
											kernel_set_attribute_symbols[cur_func_type], kernel_set_launch_symbols[cur_func_type]);

			cur_func_cnt++;
		}
	}



	// NOW SAVE ALL FUNC META ARRAY....!

	char * all_func_meta_filepath = "build/cuda_all_functions_meta.dat";

	FILE * fp = fopen(all_func_meta_filepath, "wb");
	if (!fp){
		fprintf(stderr, "Error: unable to open file to save all function metadata: %s\n", all_func_meta_filepath);
		return -1;
	}

	printf("Func Meta size: %lu\n", sizeof(Func_Meta));

	size_t n_written = fwrite(all_func_meta, sizeof(Func_Meta), cur_func_cnt, fp);
	if (n_written != cur_func_cnt){
		fprintf(stderr, "Error: failed to write all func meta, expected: %d, wrote: %lu...\n", cur_func_cnt, n_written);
		return -1;
	}

	fclose(fp);

	printf("Success! Saved %d functions with path: %s...\n", cur_func_cnt, all_func_meta_filepath);

	return 0;
}