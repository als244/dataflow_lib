#include "set_external_op_skeletons.h"


void set_external_matmul_skeleton(Op_Skeleton * skeleton) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s", "matmul");

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 16;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// A DataflowDatatype
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	// B DataflowDatatype
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// C DataflowDatatype
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	// D DataflowDatatype
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	// Compute Type as DataflowDatatype (FP32, FP16, or BF16)
	arg_dtypes[4] = DATAFLOW_INT_SCALAR;

	// M
	arg_dtypes[5] = DATAFLOW_INT_SCALAR;
	// K
	arg_dtypes[6] = DATAFLOW_INT_SCALAR;
	// N
	arg_dtypes[7] = DATAFLOW_INT_SCALAR;
	// alpha
	arg_dtypes[8] = DATAFLOW_FP32_SCALAR;
	// beta
	arg_dtypes[9] = DATAFLOW_FP32_SCALAR;
	// workspace bytes
	arg_dtypes[10] = DATAFLOW_UINT64;
	// workspace
	arg_dtypes[11] = DATAFLOW_VOID;

	// A
	arg_dtypes[12] = DATAFLOW_VOID;
	// B
	arg_dtypes[13] = DATAFLOW_VOID;
	// C
	arg_dtypes[14] = DATAFLOW_VOID;
	// D
	arg_dtypes[15] = DATAFLOW_VOID;

	

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);

}

void set_external_attention_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

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

void set_external_attention_bwd_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

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
