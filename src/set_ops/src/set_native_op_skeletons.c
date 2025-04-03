#include "set_native_op_skeletons.h"

// This function should just be called during the registration of the op
// Not performance senesitive so the strcmps are fine...
int register_native_ops(Dataflow_Handle * dataflow_handle, char * native_function_code_filename, char * native_function_config_filename) {

	int num_fwd_datatypes = 5;
	int num_bwd_datatypes = 3;

	DataflowDatatype fwd_datatypes[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2};
	DataflowDatatype bwd_datatypes[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16};

	int num_base_ops = 6;
	char * op_base_names[6] = {"embedding", "rms_norm", "rope", "swiglu", "softmax", "cross_entropy_loss"};

	char * op_init_symbols[6] = {NULL, "rms_norm_set_attribute_config", NULL, NULL, NULL, NULL};
	
	
	// cross entropy loss doesn't have function for fp8 yet...
	bool num_fwd_ops[6] = {5, 5, 5, 5, 7, 3};
	bool num_bwd_ops[6] = {0, 14, 3, 7, 0, 0};

	int num_funcs = 54;

	bool has_bwd_x[6] = {false, true, true, true, false, false};
	bool has_bwd_w[6] = {false, true, false, false, false, false};

	int bwd_combos = 7;

	char * fwd_strs[] = {"fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2"};
	char * bwd_strs[] = {"fp32", "fp16", "bf16"};
	char * bwd_combo_strs[] = {"fp32_fp32", "fp16_fp16", "bf16_bf16", \
								"fp8e4m3_fp16", "fp8e4m3_bf16", "fp8e5m2_fp16", "fp8e5m2_bf16"};


	DataflowDatatype bwd_combo_fwd_dts[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, 
											DATAFLOW_FP8E4M3, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2, DATAFLOW_FP8E5M2};
	DataflowDatatype bwd_combo_bwd_dts[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, 
											DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP16, DATAFLOW_BF16};

	char * suffix = "kernel";

	Op_Skeleton * native_op_skeletons = (Op_Skeleton *) malloc(num_funcs * sizeof(Op_Skeleton));

	char ** native_func_symbols = (char **) malloc(num_funcs * sizeof(char *));
	char ** native_func_launch_symbols = (char **) malloc(num_funcs * sizeof(char *));
	for (int i = 0; i < num_funcs; i++){
		// must 
		native_func_symbols[i] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
		native_func_launch_symbols[i] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
	}
	char ** native_func_init_symbols = (char **) malloc(num_funcs * sizeof(char *));
	for (int i = 0; i < num_funcs; i++){
		native_func_init_symbols[i] = NULL;
	}
	
	int cur_func = 0;

	char op_base_bwd_extented[FUNC_SYMBOL_MAX_LEN];

	for (int i = 0; i < num_base_ops; i++){
		// all ops have fwd

		

		if ((strcmp(op_base_names[i], "softmax") != 0) && (strcmp(op_base_names[i], "cross_entropy_loss") != 0)) {
			for (int s = 0; s < num_fwd_datatypes; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_names[i], fwd_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_names[i]);

				set_native_op_skeleton(&native_op_skeletons[cur_func], op_base_names[i], fwd_datatypes[s], DATAFLOW_NONE);
				cur_func++;
			}
		}
		// special cases for softmax and cross entropy loss

		// softmax transitions from fwd dt to bwd and takes both (all combos)
		if (strcmp(op_base_names[i], "softmax") == 0) {
			for (int s = 0; s < bwd_combos; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_names[i], bwd_combo_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_names[i]);
				set_native_op_skeleton(&native_op_skeletons[cur_func], op_base_names[i], bwd_combo_fwd_dts[s], bwd_combo_bwd_dts[s]);
				cur_func++;
			}
		}

		// only do cross entropy loss for fp32, fp16, bf16
		// only the bwd_dt is used for cross entropy loss
		if (strcmp(op_base_names[i], "cross_entropy_loss") == 0) {
			for (int s = 0; s < num_bwd_datatypes; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_names[i], bwd_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_names[i]);

				set_native_op_skeleton(&native_op_skeletons[cur_func], op_base_names[i], DATAFLOW_NONE, bwd_datatypes[s]);
				cur_func++;
			}
		}

		if (has_bwd_x[i]){
			sprintf(op_base_bwd_extented, "%s_bwd_x", op_base_names[i]);

			// rope_bwd_x only takes in the bwd_dt...
			if (strcmp(op_base_bwd_extented, "rope_bwd_x") != 0){

				for (int s = 0; s < bwd_combos; s++){
					if (op_init_symbols[i]){
						native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
						sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
					}
					sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_bwd_extented, bwd_combo_strs[s], suffix);
					sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_bwd_extented);
					set_native_op_skeleton(&native_op_skeletons[cur_func], op_base_bwd_extented, bwd_combo_fwd_dts[s], bwd_combo_bwd_dts[s]);
					cur_func++;
				}
			}
			else {
				for (int s = 0; s < num_bwd_datatypes; s++){
					if (op_init_symbols[i]){
						native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
						sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
					}
					sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_bwd_extented, bwd_strs[s], suffix);
					sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_bwd_extented);
					set_native_op_skeleton(&native_op_skeletons[cur_func], op_base_bwd_extented, DATAFLOW_NONE, bwd_datatypes[s]);
					cur_func++;
				}
			}
		}
		

		if (has_bwd_w[i]){
			sprintf(op_base_bwd_extented, "%s_bwd_w", op_base_names[i]);
			for (int s = 0; s < bwd_combos; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_bwd_extented, bwd_combo_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_bwd_extented);
				set_native_op_skeleton(&native_op_skeletons[cur_func], op_base_bwd_extented, bwd_combo_fwd_dts[s], bwd_combo_bwd_dts[s]);
				cur_func++;
			}
		}
	}

	// Now finally register all of the functions...

	int added_funcs = (dataflow_handle -> register_native_code)(dataflow_handle, native_function_code_filename, native_function_config_filename, 
																num_funcs, native_op_skeletons, 
																native_func_symbols, native_func_launch_symbols, native_func_init_symbols);
	if (added_funcs != num_funcs){
		fprintf(stderr, "WARNING: failed to register all, native ops, expected %d functions, got %d...\n", num_funcs, added_funcs);
	}

	// Now can free the metadata used to register...

	for (int i = 0; i < num_funcs; i++){
		free(native_func_symbols[i]);
		free(native_func_launch_symbols[i]);
		if (native_func_init_symbols[i]){
			free(native_func_init_symbols[i]);
		}
	}

	free(native_func_symbols);
	free(native_func_launch_symbols);
	free(native_func_init_symbols);
	free(native_op_skeletons);

	return added_funcs;
}

// This function should just be called during the registration of the op
// Not performance senesitive so the strcmps are fine...

// switches into respective function based on op_base_name
// return 0 on success, -1 on failure
// depending on function might either use fwd_dt or bwd_dt or both
int set_native_op_skeleton(Op_Skeleton * skeleton, char * op_name, DataflowDatatype fwd_dt, DataflowDatatype bwd_dt) {

	if (strcmp(op_name, "embedding") == 0) {
        set_native_embedding_skeleton(skeleton, fwd_dt);
    } 
	else if (strcmp(op_name, "rms_norm") == 0) {
        set_native_rms_norm_skeleton(skeleton, fwd_dt);
    }
	else if (strcmp(op_name, "rms_norm_bwd_x") == 0) {
		set_native_rms_norm_bwd_x_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "rms_norm_bwd_w") == 0) {
		set_native_rms_norm_bwd_w_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "rope") == 0) {
		set_native_rope_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "rope_bwd_x") == 0) {
		set_native_rope_bwd_x_skeleton(skeleton, bwd_dt);
	}
	else if (strcmp(op_name, "copy_to_seq_context") == 0) {
		set_native_copy_to_seq_context_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "swiglu") == 0) {
		set_native_swiglu_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "swiglu_bwd_x") == 0) {
		set_native_swiglu_bwd_x_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "softmax") == 0) {
		set_native_softmax_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "cross_entropy_loss") == 0) {
		set_native_cross_entropy_loss_skeleton(skeleton, bwd_dt);
	}
	else {
		printf("Cannot set skeleton, unknown op: %s, with fwd_dt: %s, bwd_dt: %s\n", op_name, dataflow_datatype_as_string(fwd_dt), dataflow_datatype_as_string(bwd_dt));
		return -1;
	}
	return 0;
}

void set_native_embedding_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "embedding", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 5;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// num tokens
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// token ids
	arg_dtypes[2] = DATAFLOW_UINT32;
	// embedding table	
	arg_dtypes[3] = fwd_datatype;
	// output
	arg_dtypes[4] = fwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
	
	
}

void set_native_rms_norm_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "rms_norm", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = fwd_datatype;
	arg_dtypes[6] = DATAFLOW_FP32;
	arg_dtypes[7] = DATAFLOW_FP32;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);

}

void set_native_rms_norm_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "rms_norm_bwd_x", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 9;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = DATAFLOW_FP32;
	arg_dtypes[4] = DATAFLOW_FP32;
	arg_dtypes[5] = fwd_datatype;
	arg_dtypes[6] = fwd_datatype;
	arg_dtypes[7] = bwd_datatype;
	arg_dtypes[8] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

void set_native_rms_norm_bwd_w_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "rms_norm_bwd_w", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 7;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = DATAFLOW_FP32;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = bwd_datatype;
	arg_dtypes[6] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

void set_native_rope_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "rope", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_UINT64_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	arg_dtypes[4] = DATAFLOW_INT_SCALAR;
	arg_dtypes[5] = DATAFLOW_INT;
	arg_dtypes[6] = fwd_datatype;
	arg_dtypes[7] = fwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

void set_native_rope_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "rope_bwd_x", dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_UINT64_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	arg_dtypes[4] = DATAFLOW_INT_SCALAR;
	arg_dtypes[5] = DATAFLOW_INT;
	arg_dtypes[6] = bwd_datatype;
	arg_dtypes[7] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

void set_native_copy_to_seq_context_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "copy_to_seq_context", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_UINT64_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = DATAFLOW_INT;
	arg_dtypes[6] = DATAFLOW_UINT64;
	arg_dtypes[7] = DATAFLOW_INT;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);

}

void set_native_swiglu_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "swiglu", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 

	int num_args = 5;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = fwd_datatype;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);

}

void set_native_swiglu_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "swiglu_bwd_x", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0';

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

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

void set_native_softmax_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "softmax", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0';
	
	int num_args = 4;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = fwd_datatype;
	arg_dtypes[3] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

void set_native_cross_entropy_loss_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];
	
	sprintf(op_nickname, "%s_%s", "cross_entropy_loss", dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0';
	
	int num_args = 4;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = bwd_datatype;
	arg_dtypes[3] = DATAFLOW_UINT32;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}
