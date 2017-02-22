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

void nnp_sgemm_only_4x12__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_sgemm_upto_4x12__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_sgemm_only_4x3__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_sgemm_upto_4x3__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

typedef void (*nnp_fast_tuple_gemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*nnp_full_tuple_gemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);

void nnp_c8gemm_only_2x2__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c8gemm_upto_2x2__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c8gemm_conjb_only_2x2__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c8gemm_conjb_upto_2x2__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c8gemm_conjb_transc_only_2x2__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c8gemm_conjb_transc_upto_2x2__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c6gemm_only_2x2__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c6gemm_upto_2x2__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c6gemm_conjb_only_2x2__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c6gemm_conjb_upto_2x2__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c6gemm_conjb_transc_only_2x2__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c6gemm_conjb_transc_upto_2x2__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s8gemm_only_3x4__fma3(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s8gemm_upto_3x4__fma3(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4gemm_only_3x4__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4gemm_upto_3x4__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c4gemm_only_2x2__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c4gemm_upto_2x2__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c4gemm_conjb_only_2x2__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c4gemm_conjb_upto_2x2__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c4gemm_conjb_transc_only_2x2__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c4gemm_conjb_transc_upto_2x2__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c2gemm_only_2x2__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c2gemm_upto_2x2__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c2gemm_conjb_only_2x2__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c2gemm_conjb_upto_2x2__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c2gemm_conjb_transc_only_2x2__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c2gemm_conjb_transc_upto_2x2__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4gemm_only_3x4__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4gemm_upto_3x4__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c4gemm_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c4gemm_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c4gemm_conjb_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c4gemm_conjb_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_c4gemm_conjb_transc_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_c4gemm_conjb_transc_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c2gemm_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c2gemm_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c2gemm_conjb_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c2gemm_conjb_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s4c2gemm_conjb_transc_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s4c2gemm_conjb_transc_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s2gemm_only_2x2__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s2gemm_upto_2x2__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_s2gemm_transc_only_2x2__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_s2gemm_transc_upto_2x2__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_cgemm_only_2x2__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_cgemm_upto_2x2__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_cgemm_conjb_only_2x2__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_cgemm_conjb_upto_2x2__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void nnp_cgemm_conjb_transc_only_2x2__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void nnp_cgemm_conjb_transc_upto_2x2__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

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

void nnp_sdotxf1__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf2__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf3__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf4__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf5__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf6__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf7__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf8__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);

void nnp_sdotxf1__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf2__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf3__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf4__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf5__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf6__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf7__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void nnp_sdotxf8__scalar(const float* x, const float* y, size_t stride_y, float* sum, size_t n);

typedef void (*nnp_shdotxf_function)(const float*, const void*, size_t, float*, size_t);
void nnp_shdotxf1__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf2__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf3__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf4__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf5__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf6__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf7__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf8__avx2(const float* x, const void* y, size_t stride_y, float* sum, size_t n);

void nnp_shdotxf1__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf2__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf3__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf4__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf5__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf6__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf7__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf8__psimd(const float* x, const void* y, size_t stride_y, float* sum, size_t n);

void nnp_shdotxf1__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf2__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf3__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf4__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf5__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf6__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf7__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);
void nnp_shdotxf8__scalar(const float* x, const void* y, size_t stride_y, float* sum, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif
