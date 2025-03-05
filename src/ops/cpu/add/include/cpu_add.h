#ifndef CPU_ADD_H
#define CPU_ADD_H

#include "dataflow.h"
#include "ops.h"

// Thread argument structure used for operations.
typedef struct {
    // Generic pointer types: the implementations know how to cast.
    const void *x;
    void *y;
    long start;
    long end;
    // Some operations need extra parameters.
    float x_scale;   // For add with scale factor.
    // (Other parameters like beta1, beta2, etc. for Adam can be added.)
} add_thread_args_t;

int cpu_add_avx2(DataflowDatatype dtype, const void * x, void * y, long n, float x_scale, int num_threads);
int cpu_add_avx512(DataflowDatatype dtype, const void * x, void * y, long n, float x_scale, int num_threads);


#endif
