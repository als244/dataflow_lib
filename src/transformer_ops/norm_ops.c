#include "transformer_ops.h"


int submit_rms_norm(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * rms_weight, void * X, void * out, float * weighted_sums, float * rms_vals){
		
	Op rms_norm_op;

	set_native_rms_norm_skeleton(&rms_norm_op.op_skeleton, fwd_dt);

	void ** fwd_op_args = rms_norm_op.op_args;

	fwd_op_args[0] = &n_rows;
	fwd_op_args[1] = &n_cols;
	fwd_op_args[2] = &eps;
	fwd_op_args[3] = &rms_weight;
	fwd_op_args[4] = &X;
	fwd_op_args[5] = &out;
	fwd_op_args[6] = &weighted_sums;
	fwd_op_args[7] = &rms_val;


	ret = handle.submit_op(&handle, &rms_norm_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to rms norm...\n");
		return -1;
	}

	return 0;
}


int submit_rms_norm_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_weighted_sums, float * fwd_rms_vals,
								 void * rms_weight, void * X_inp, void * upstream_dX, void * dX){
		
	Op rms_norm_bwd_x_op;

	set_native_rms_norm_bwd_x_skeleton(&rms_norm_bwd_x_op.op_skeleton, fwd_dt, bwd_dt);

	void ** bwd_x_op_args = rms_norm_bwd_x_op.op_args;

	bwd_x_op_args[0] = &n_rows;
	bwd_x_op_args[1] = &n_cols
	bwd_x_op_args[2] = &eps;
	bwd_x_op_args[3] = &fwd_weighted_sums;
	bwd_x_op_args[4] = &fwd_rms_vals;
	bwd_x_op_args[5] = &rms_weight;
	bwd_x_op_args[6] = &X_inp;
	bwd_x_op_args[7] = &upstream_dX;
	bwd_x_op_args[8] = %dX;


	ret = handle.submit_op(&handle, &rms_norm_bwd_x_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd x op...\n");
		return -1;
	}

	return 0;
}


int submit_rms_norm_bwd_w(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals, void * X_inp, void * upstream_dX, void * dW) {
	Op rms_norm_bwd_w_op;

	set_native_rms_norm_bwd_w_skeleton(&rms_norm_bwd_w_op.op_skeleton, fwd_dt, bwd_dt);

	void ** bwd_w_op_args = rms_norm_bwd_w_op.op_args;

	bwd_w_op_args[0] = &iM;
	bwd_w_op_args[1] = &iN;
	bwd_w_op_args[2] = &eps;
	bwd_w_op_args[3] = &d_sq_sums;
	bwd_w_op_args[4] = &d_orig_matrix;
	bwd_w_op_args[5] = &d_upstream_dX;
	bwd_w_op_args[6] = &d_dW;


	ret = handle.submit_op(&handle, &rms_norm_bwd_w_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd w op...\n");
		return -1;
	}
}
