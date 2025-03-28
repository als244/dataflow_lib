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

void set_external_flash3_attention_fwd_skeleton(Op_Skeleton * skeleton) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s", "flash3_attention_fwd");

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	

	int num_args = 17;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// flash_dtype_as_int
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	
	// num_seqs
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// total_q
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	// total_k
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	// cum_q_seqlens
	arg_dtypes[4] = DATAFLOW_INT;
	// max_seqlen_q
	arg_dtypes[5] = DATAFLOW_INT_SCALAR;
	// k_seqlens
	arg_dtypes[6] = DATAFLOW_INT;
	// max_seqlen_k
	
	arg_dtypes[7] = DATAFLOW_INT_SCALAR;
	// num_q_heads
	arg_dtypes[8] = DATAFLOW_INT_SCALAR;
	// num_kv_heads
	arg_dtypes[9] = DATAFLOW_INT_SCALAR;
	// head_dim
	arg_dtypes[10] = DATAFLOW_INT_SCALAR;
	
	// x_q
	arg_dtypes[11] = DATAFLOW_VOID;
	// x_k
	arg_dtypes[12] = DATAFLOW_VOID;
	// x_v
	arg_dtypes[13] = DATAFLOW_VOID;

	
	// x_attn_out
	arg_dtypes[14] = DATAFLOW_VOID;
	// softmax_lse
	arg_dtypes[15] = DATAFLOW_VOID;

	// attn_workspace
	arg_dtypes[16] = DATAFLOW_VOID;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}


// TODO:

void set_external_flash3_attention_bwd_skeleton(Op_Skeleton * skeleton) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s", "flash3_attention_bwd");

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	

	int num_args = 17;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// flash_dtype_as_int
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	
	// num_seqs
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// total_q
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	// total_k
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	// cum_q_seqlens
	arg_dtypes[4] = DATAFLOW_INT;
	// max_seqlen_q
	arg_dtypes[5] = DATAFLOW_INT_SCALAR;
	// k_seqlens
	arg_dtypes[6] = DATAFLOW_INT;
	// max_seqlen_k
	
	arg_dtypes[7] = DATAFLOW_INT_SCALAR;
	// num_q_heads
	arg_dtypes[8] = DATAFLOW_INT_SCALAR;
	// num_kv_heads
	arg_dtypes[9] = DATAFLOW_INT_SCALAR;
	// head_dim
	arg_dtypes[10] = DATAFLOW_INT_SCALAR;
	
	// x_q
	arg_dtypes[11] = DATAFLOW_VOID;
	// x_k
	arg_dtypes[12] = DATAFLOW_VOID;
	// x_v
	arg_dtypes[13] = DATAFLOW_VOID;

	
	// x_attn_out
	arg_dtypes[14] = DATAFLOW_VOID;
	// softmax_lse
	arg_dtypes[15] = DATAFLOW_VOID;

	// attn_workspace
	arg_dtypes[16] = DATAFLOW_VOID;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint, OP_IDENTIFIER_FINGERPRINT_TYPE);
}

