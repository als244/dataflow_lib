#ifndef SET_NATIVE_OPS_H
#define SET_NATIVE_OPS_H

#include "dataflow.h"
#include "fingerprint.h"
#include "ops.h"
#include "dataflow_handle.h"

// helper function to register native ops
// Should only be called once per file (for each handle)
int register_native_ops(Dataflow_Handle * dataflow_handle, char * native_function_code_filename, char * native_function_config_filename);

// switches into respective function based on op_base_name
// return 0 on success, -1 on failure
// depending on function might either use fwd_dt or bwd_dt or both
int set_native_op_skeleton(Op_Skeleton * skeleton, char * op_name, DataflowDatatype fwd_dt, DataflowDatatype bwd_dt);

void set_native_embedding_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);

void set_native_rms_norm_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void set_native_rms_norm_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void set_native_rms_norm_bwd_w_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);

void set_native_rope_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void set_native_rope_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);
void set_native_copy_to_seq_context_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);

void set_native_swiglu_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void set_native_swiglu_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);

void set_native_softmax_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void set_native_cross_entropy_loss_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);

#endif