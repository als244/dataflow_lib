#include "transformer_ops.h"


// If A, C, D are all stored in Row-Major
// And B is stored in Col-Major. If so, it compute:
// D = alpha * AB + beta * C

// If B is stored in Row-Major that implies it computes:
// D = alpha * AB^T + beta * C
int submit_matmul(Dataflow_Handle * handle, int stream_id, 
					DataflowDatatype a_dt, DataflowDatatype b_dt, DataflowDatatype c_dt, DataflowDatatype d_dt,
					DataflowDatatype compute_dt,
					int M, int K, int N,
					float alpha, float beta,
					uint64_t workspaceBytes, void * workspace,
					void * A, void * B, void * C, void * D,
					int num_procs) {


	int ret;

	Op matmul_op;

	set_external_matmul_skeleton(&matmul_op.op_skeleton);

	void ** op_args = matmul_op.op_args;

	op_args[0] = &a_dt;
	op_args[1] = &b_dt;
	op_args[2] = &c_dt;
	op_args[3] = &d_dt;
	op_args[4] = &compute_dt;
	op_args[5] = &M;
	op_args[6] = &K;
	op_args[7] = &N;
	op_args[8] = &alpha;
	op_args[9] = &beta;
	op_args[10] = &workspaceBytes;
	op_args[11] = &workspace;
	op_args[12] = &A;
	op_args[13] = &B;
	op_args[14] = &C;
	op_args[15] = &D;
	op_args[16] = &num_procs;

	ret = (handle -> submit_op)(handle, &matmul_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit matmul_op...\n");
		return -1;
	}

	return 0;


}