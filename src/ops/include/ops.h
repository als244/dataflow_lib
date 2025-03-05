#ifndef OPS_H
#define OPS_H

#include "dataflow.h"

#define MAX_OP_ARGS 16

#define MAX_OP_NICKNAME_SIZE 255

// sha256 encoding of each op skeleton
#define OP_IDENTIFIER_FINGERPRINT_TYPE 0
#define OP_IDENTIFIER_FINGERPRINT_NUM_BYTES 32


typedef struct op_skeleton_header {
	// can specify different variants here
	char op_nickname[MAX_OP_NICKNAME_SIZE + 1];
	int num_args;
	DataflowDatatype arg_dtypes[MAX_OP_ARGS];
} Op_Skeleton_Header;

typedef struct op_identifier {
	uint8_t fingerprint[OP_IDENTIFIER_FINGERPRINT_NUM_BYTES];
} Op_Identifier;

typedef struct Op_Skeleton {
	Op_Skeleton_Header header;
	// identifier is tied to the hash of the header
	Op_Identifier identifier;
} Op_Skeleton;

typedef struct op {
	Op_Skeleton op_skeleton;
	void * op_args[MAX_OP_ARGS];
} Op;




// int cpu_adam_update_avx2(DataflowDatatype dtype, void * w, void * m, void * v, void * g, long n, float beta1, float beta2, float lr, float eps, int num_threads);
// int cpu_adam_update_avx512(DataflowDatatype dtype, void * w, void * m, void * v, void * g, long n, float beta1, float beta2, float lr, float eps, int num_threads);

#endif
