#ifndef ATTENTION_HELPER_H
#define ATTENTION_HELPER_H

#include "dataflow.h"
#include "dataflow_datatype.h"
#include "dataflow_handle.h"

// to get device info
#include "cuda_dataflow_handle.h"
#include "ops.h"


// calling flash3_fwd_wrapper from libflash3.so
#include "flash3_wrapper.h"

// functions to export

// No need
// int flash3_attention_init(Dataflow_Handle * dataflow_handle, void * op_table_value)

int flash3_attention_fwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);

int flash3_attention_bwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);


#endif