#include "cpu_add.h"

static void *thread_add_func_fp32_avx512(void *arg) {
    add_thread_args_t *args = (add_thread_args_t *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // 16 FP32 elements per 512-bit vector.
    long limit = start + (len / chunk) * chunk;
    
    float * x = (float *) args -> x;
    float * y = (float *) args -> y;

    // Broadcast the scalar factor into a 512-bit vector.
    __m512 factor_vec = _mm512_set1_ps(args->x_scale);
    
    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP32 values from x and y.
        __m512 x_vec = _mm512_loadu_ps(&x[i]);
        __m512 y_vec = _mm512_loadu_ps(&y[i]);
        
        // Compute: result = y + factor * x.
        __m512 result = _mm512_add_ps(y_vec, _mm512_mul_ps(x_vec, factor_vec));
        
        // Store the result back to y.
        _mm512_storeu_ps(&y[i], result);
    }
    
    // Scalar fallback for any remaining elements.
    for (long i = limit; i < end; i++) {
        y[i] = y[i] + args->x_scale * x[i];
    }
    
    return NULL;
}

static void *thread_add_func_fp16_avx512(void *arg) {
    add_thread_args_t *args = (add_thread_args_t *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // 16 FP16 elements per 256-bit load
    long limit = start + (len / chunk) * chunk;

    uint16_t * x = (uint16_t *) args -> x;
    uint16_t * y = (uint16_t *) args -> y;

    // Broadcast the scalar factor into a 512-bit vector.
    __m512 factor_vec = _mm512_set1_ps(args->x_scale);

    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP16 values from x and y as 256-bit integer vectors.
        __m256i x_half = _mm256_loadu_si256((__m256i*)(&x[i]));
        __m256i y_half = _mm256_loadu_si256((__m256i*)(&y[i]));

        // Convert the FP16 values to FP32.
        __m512 x_f = _mm512_cvtph_ps(x_half);
        __m512 y_f = _mm512_cvtph_ps(y_half);

        // Multiply x by the scalar factor and add to y.
        __m512 result_f = _mm512_add_ps(y_f, _mm512_mul_ps(x_f, factor_vec));

        // Convert the FP32 result back to FP16.
        __m256i res_half = _mm512_cvtps_ph(result_f, _MM_FROUND_TO_NEAREST_INT);

        // Store the result back to y.
        _mm256_storeu_si256((__m256i*)(&y[i]), res_half);
    }

    // Scalar fallback for any remaining elements.
    for (long i = limit; i < end; i++) {
        float xf = solo_fp16_to_fp32(x[i]);
        float yf = solo_fp16_to_fp32(y[i]);
        float result = yf + args->x_scale * xf;
        y[i] = solo_fp32_to_fp16(result);
    }

    return NULL;
}


#ifndef _mm256_loadu_bf16
static inline __m256bh my_mm256_loadu_bf16(const void *addr) {
    return (__m256bh)_mm256_loadu_si256((const __m256i *)addr);
}
#define _mm256_loadu_bf16(addr) my_mm256_loadu_bf16(addr)
#endif

#ifndef _mm256_storeu_bf16
static inline void my_mm256_storeu_bf16(void *addr, __m256bh a) {
    _mm256_storeu_si256((__m256i *)addr, (__m256i)a);
}
#define _mm256_storeu_bf16(addr, a) my_mm256_storeu_bf16(addr, a)
#endif


static void *thread_add_func_bf16_avx512(void *arg) {
    add_thread_args_t *args = (add_thread_args_t *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // Process 16 BF16 elements per iteration.
    long limit = start + (len / chunk) * chunk;

    uint16_t * x = (uint16_t *) args -> x;
    uint16_t * y = (uint16_t *) args -> y;

    // Broadcast the scalar factor into a 512-bit vector.
    __m512 factor_vec = _mm512_set1_ps(args->x_scale);

    for (long i = start; i < limit; i += chunk) {
        // Load 16 BF16 values from x and y using BF16 load intrinsic.
        __m256bh x_bf16 = _mm256_loadu_bf16(&x[i]);
        __m256bh y_bf16 = _mm256_loadu_bf16(&y[i]);

        // Convert BF16 values to FP32.
        __m512 x_f = _mm512_cvtpbh_ps(x_bf16);
        __m512 y_f = _mm512_cvtpbh_ps(y_bf16);

        // Multiply x by the scalar factor and add to y.
        __m512 result_f = _mm512_add_ps(y_f, _mm512_mul_ps(x_f, factor_vec));

        // Convert the FP32 result back to BF16.
        __m256bh res_bf16 = _mm512_cvtneps_pbh(result_f);

        // Store the result back to y.
        _mm256_storeu_bf16(&y[i], res_bf16);
    }

    // Scalar fallback: process any remaining elements.
    for (long i = limit; i < end; i++) {
        float xf = solo_bf16_to_fp32(x[i]);
        float yf = solo_bf16_to_fp32(y[i]);
        float result = yf + args->x_scale * xf;
        y[i] = solo_fp32_to_bf16(result);
    }

    return NULL;
}


int cpu_add_avx512(DataflowDatatype dtype, const void * x, void * y, long n, float x_scale, int num_threads){

    void * (*thread_add_func)(void *);

    switch (dtype){
        case DATAFLOW_FP32:
            thread_add_func = &thread_add_func_fp32_avx512;
            break;
        case DATAFLOW_FP16:
            thread_add_func = &thread_add_func_fp16_avx512;
            break;
        case DATAFLOW_BF16:
            thread_add_func = &thread_add_func_bf16_avx512;
            break;
        default:
            thread_add_func = NULL;
            break;
    }

    if (!thread_add_func){
        fprintf(stderr, "Error: add with dtype type of %d unavailable for avx512\n", dtype);
        return -1;
    }


    if (num_threads <= 1) {
        add_thread_args_t args = {x, y, 0, n, x_scale};
        thread_add_func(&args);
        return 0;
    }

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    add_thread_args_t *targs = malloc(num_threads * sizeof(add_thread_args_t));
    if (!threads || !targs) {
        fprintf(stderr, "Error: Could not alloc space for threads or args...\n");
        return -1;
    }

    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    
    for (int t = 0; t < num_threads; t++) {
        targs[t].x = x;
        targs[t].y = y;
        targs[t].x_scale = x_scale;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_add_func, &targs[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(targs);

    return 0;
}