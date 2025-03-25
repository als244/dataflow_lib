#ifndef DATAFLOW_DATATYPE_H
#define DATAFLOW_DATATYPE_H

#include "dataflow_common.h"
#include "solo_conversion.h"

typedef enum {
	DATAFLOW_NONE,
	DATAFLOW_VOID,
	DATAFLOW_FP64,
	DATAFLOW_FP32,
	DATAFLOW_FP16,
	DATAFLOW_BF16,
	DATAFLOW_FP8E4M3,
	DATAFLOW_FP8E5M2,
	DATAFLOW_UINT64,
	DATAFLOW_UINT32,
	DATAFLOW_UINT16,
	DATAFLOW_UINT8,
	DATAFLOW_LONG,
	DATAFLOW_INT,
	DATAFLOW_BOOL,
	DATAFLOW_FP64_SCALAR,
	DATAFLOW_FP32_SCALAR,
	DATAFLOW_FP16_SCALAR,
	DATAFLOW_BF16_SCALAR,
	DATAFLOW_FP8E4M3_SCALAR,
	DATAFLOW_FP8E5M2_SCALAR,
	DATAFLOW_UINT64_SCALAR,
	DATAFLOW_UINT32_SCALAR,
	DATAFLOW_UINT16_SCALAR,
	DATAFLOW_UINT8_SCALAR,
	DATAFLOW_LONG_SCALAR,
	DATAFLOW_INT_SCALAR,
	DATAFLOW_BOOL_SCALAR
} DataflowDatatype;

// E.g. returns the datatype size corresponding to the elements within the array
// not the 64-bits corresponding to pointer iteself

// may need to eventually reutrn double to express 4-bit values...
size_t dataflow_sizeof_element(DataflowDatatype arr_dtype);

char * dataflow_datatype_as_string(DataflowDatatype dtype);

int dataflow_convert_datatype(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads);

#endif
