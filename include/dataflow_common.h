#ifndef DATAFLOW_COMMON_H
#define DATAFLOW_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

#include <semaphore.h>
#include <string.h>
#include <immintrin.h>

// path math
#include <linux/limits.h>

// loading shared library and using symbols
#include <dlfcn.h>


#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

typedef int (*Item_Cmp)(void * item, void * other_item);
typedef uint64_t (*Hash_Func)(void * item, uint64_t table_size);

#define MY_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MY_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MY_CEIL(a, b) ((a + b - 1) / b)



#endif
