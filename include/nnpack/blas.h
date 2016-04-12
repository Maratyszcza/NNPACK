#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nnp_fast_sgemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*nnp_full_sgemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);

void nnp_sgemm_only_4x24__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_sgemm_upto_4x24__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_sgemm_only_4x8__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_sgemm_upto_4x8__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

typedef void (*nnp_tuple_gemm_function)(size_t, size_t, const float*, const float*, float*, size_t, size_t);

void nnp_c8gemm1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemm1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemm2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemm2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4c6gemm2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemm1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemm1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemm2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_c8gemmca1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemmca1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemmca2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemmca2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4c6gemmca1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemmca1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemmca2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemmca2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_c8gemmcb1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemmcb1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemmcb2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c8gemmcb2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4c6gemmcb1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemmcb1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemmcb2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c6gemmcb2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s8gemm1x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm1x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm1x3__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm1x4__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm2x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm2x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm2x3__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm2x4__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm3x1__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm3x2__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm3x3__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s8gemm3x4__fma3(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);


void nnp_c4gemm1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemm1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemm2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemm2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4c2gemm2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemm1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemm1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemm2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_c4gemmca1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemmca1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemmca2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemmca2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4c2gemmca2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemmca1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemmca1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemmca2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_c4gemmcb1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemmcb1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemmcb2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_c4gemmcb2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4c2gemmcb2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemmcb1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemmcb1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4c2gemmcb2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);

void nnp_s4gemm1x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm1x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm1x3__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm1x4__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm2x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm2x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm2x3__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm2x4__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm3x1__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm3x2__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm3x3__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);
void nnp_s4gemm3x4__psimd(size_t k, size_t k_tile, const float* a, const float* b, float* c, size_t row_stride, size_t column_stride);


typedef void (*nnp_sdotxf_function)(const float*, const float*, size_t, float*, size_t);
void nnp_sdotxf1__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf2__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf3__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf4__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf5__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf6__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf7__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf8__avx2(const float* x, const float* y, size_t stride_y, float* sum, size_t n);

void nnp_sdotxf1__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf2__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf3__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf4__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf5__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf6__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf7__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf8__psimd(const float* x, const float* y, size_t stride_y, float* sum, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif
