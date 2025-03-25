#include "set_native_op_skeletons.h"

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
