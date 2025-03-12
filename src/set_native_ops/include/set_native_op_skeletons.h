#ifndef SET_NATIVE_OPS_H
#define SET_NATIVE_OPS_H

#include "dataflow.h"
#include "fingerprint.h"
#include "ops.h"
#include "dataflow_handle.h"

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