#ifndef MATMUL_HELPER_H
#define MATMUL_HELPER_H

#include "dataflow.h"
#include "dataflow_datatype.h"
#include "solo_conversion.h"
#include "dataflow_handle.h"
#include "cuda_dataflow_handle.h" // for Cuda_Function
#include "ops.h"

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif


#include <cublasLt.h>


typedef struct cublas_matmul_op_extra {
	cublasLtHandle_t cublas_handle;
} Cublas_Matmul_Op_Extra;

// responsible for setting cuda function extra
int cublas_matmul_init(Dataflow_Handle * dataflow_handle, void * op_table_value);

int cublas_matmul(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);

#endif
