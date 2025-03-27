#ifndef SET_EXTERNAL_OPS_H
#define SET_EXTERNAL_OPS_H

#include "dataflow.h"
#include "fingerprint.h"
#include "ops.h"
#include "dataflow_handle.h"

// Cublas Matmul Helper
void set_external_matmul_skeleton(Op_Skeleton * skeleton);


// 
// Flash3 Attention Helpder
void set_external_flash3_attention_fwd_skeleton(Op_Skeleton * skeleton);
void set_external_flash3_attention_bwd_skeleton(Op_Skeleton * skeleton);

#endif