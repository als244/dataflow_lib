#include "cpu_add.h"

static void *thread_add_func_fp32_avx2(void *arg) {
    add_thread_args_t *args = (add_thread_args_t *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 8;  // 8 FP32 elements per 256-bit vector
    long limit = start + (len / chunk) * chunk;

    float *x = (float *)args->x;
    float *y = (float *)args->y;

    // Broadcast the scalar factor into a 256-bit vector.
    __m256 factor_vec = _mm256_set1_ps(args->x_scale);

    for (long i = start; i < limit; i += chunk) {
        // Load 8 FP32 values from x and y.
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);

        // Compute: y[i] = y[i] + x_scale * x[i]
        __m256 result = _mm256_add_ps(y_vec, _mm256_mul_ps(x_vec, factor_vec));

        // Store the results back to y.
        _mm256_storeu_ps(&y[i], result);
    }

    // Fallback scalar loop for any remaining elements.
    for (long i = limit; i < end; i++) {
        y[i] = y[i] + args->x_scale * x[i];
    }

    return NULL;
}

static void *thread_add_func_fp16_avx2(void *arg) {
    add_thread_args_t *args = (add_thread_args_t *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 8;  // 8 FP16 elements per 128-bit load (8*16 = 128 bits)
    long limit = start + (len / chunk) * chunk;

    // Broadcast the scalar factor into a 256-bit vector (8 floats).
    __m256 factor_vec = _mm256_set1_ps(args->x_scale);

    // Cast the generic pointers to FP16 (uint16_t) arrays.
    uint16_t *x = (uint16_t *)args->x;
    uint16_t *y = (uint16_t *)args->y;

    for (long i = start; i < limit; i += chunk) {
        // Load 8 FP16 values from x and y.
        __m128i x_half = _mm_loadu_si128((__m128i*)(&x[i]));
        __m128i y_half = _mm_loadu_si128((__m128i*)(&y[i]));

        // Convert the 8 FP16 values to FP32.
        __m256 x_f = _mm256_cvtph_ps(x_half);
        __m256 y_f = _mm256_cvtph_ps(y_half);

        // Multiply x by the scalar factor and add to y.
        __m256 result_f = _mm256_add_ps(y_f, _mm256_mul_ps(x_f, factor_vec));

        // Convert the FP32 results back to FP16.
        __m128i res_half = _mm256_cvtps_ph(result_f, _MM_FROUND_TO_NEAREST_INT);

        // Store the resulting 8 FP16 values back to y.
        _mm_storeu_si128((__m128i*)(&y[i]), res_half);
    }

    // Process any remaining elements with a scalar fallback.
    for (long i = limit; i < end; i++) {
        float xf = solo_fp16_to_fp32(x[i]);
        float yf = solo_fp16_to_fp32(y[i]);
        float result = yf + args->x_scale * xf;
        y[i] = solo_fp32_to_fp16(result);
    }

    return NULL;
}

static void *thread_add_func_bf16_avx2(void *arg) {
    add_thread_args_t *args = (add_thread_args_t *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 8;  // 8 bf16 values per 128-bit load (8*16 = 128 bits)
    long limit = start + (len / chunk) * chunk;

    // Broadcast the scalar factor into a 256-bit vector (8 floats).
    __m256 factor_vec = _mm256_set1_ps(args->x_scale);

    // Cast the generic pointers to bf16 (uint16_t) arrays.
    uint16_t *x = (uint16_t *)args->x;
    uint16_t *y = (uint16_t *)args->y;

    for (long i = start; i < limit; i += chunk) {
        // Load 8 bf16 values from x and y.
        __m128i x_bf16 = _mm_loadu_si128((__m128i*)(&x[i]));
        __m128i y_bf16 = _mm_loadu_si128((__m128i*)(&y[i]));

        // Convert bf16 to float:
        // 1. Expand the 8 bf16 values to 32-bit integers.
        __m256i x_epi32 = _mm256_cvtepu16_epi32(x_bf16);
        __m256i y_epi32 = _mm256_cvtepu16_epi32(y_bf16);
        // 2. Shift left by 16 bits so the bf16 bits become the upper half of a float.
        x_epi32 = _mm256_slli_epi32(x_epi32, 16);
        y_epi32 = _mm256_slli_epi32(y_epi32, 16);
        // 3. Reinterpret the integers as floats.
        __m256 x_f = _mm256_castsi256_ps(x_epi32);
        __m256 y_f = _mm256_castsi256_ps(y_epi32);

        // Multiply x by the scalar factor and add to y.
        __m256 result_f = _mm256_add_ps(y_f, _mm256_mul_ps(x_f, factor_vec));

        // Convert the FP32 results back to bf16:
        // 1. Get the bit-level representation of the float.
        __m256i res_int = _mm256_castps_si256(result_f);
        // 2. Apply round-to-nearest by adding 0x00008000.
        __m256i rnd = _mm256_set1_epi32(0x00008000);
        res_int = _mm256_add_epi32(res_int, rnd);
        // 3. Shift right by 16 bits; the lower 16 bits of each 32-bit word become the bf16.
        __m256i res_bf16_epi32 = _mm256_srli_epi32(res_int, 16);
        // 4. Pack the eight 32-bit bf16 values into a 128-bit vector.
        __m128i low  = _mm256_extracti128_si256(res_bf16_epi32, 0);
        __m128i high = _mm256_extracti128_si256(res_bf16_epi32, 1);
        __m128i res_bf16 = _mm_packus_epi32(low, high);

        // Store the resulting 8 bf16 values back to y.
        _mm_storeu_si128((__m128i*)(&y[i]), res_bf16);
    }

    // Process any remaining elements with a scalar fallback.
    for (long i = limit; i < end; i++) {
        float xf = solo_bf16_to_fp32(x[i]);
        float yf = solo_bf16_to_fp32(y[i]);
        float result = yf + args->x_scale * xf;
        y[i] = solo_fp32_to_bf16(result);
    }

    return NULL;
}

int cpu_add_avx2(DataflowDatatype dtype, const void * x, void * y, long n, float x_scale, int num_threads){

    void * (*thread_add_func)(void *);

    switch (dtype){
        case DATAFLOW_FP32:
            thread_add_func = &thread_add_func_fp32_avx2;
            break;
        case DATAFLOW_FP16:
            thread_add_func = &thread_add_func_fp16_avx2;
            break;
        case DATAFLOW_BF16:
            thread_add_func = &thread_add_func_bf16_avx2;
            break;
        default:
            thread_add_func = NULL;
            break;
    }

    if (!thread_add_func){
        fprintf(stderr, "Error: add with dtype type of %d unavailable for avx2\n", dtype);
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