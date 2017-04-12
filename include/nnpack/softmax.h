#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nnp_exp_function)(size_t, const float*, float*);

void nnp_vector_exp__psimd(size_t n, const float* x, float* y);

void nnp_softmax__avx2(size_t n, const float* x, float* y);
void nnp_inplace_softmax__avx2(size_t n, float* v);

void nnp_softmax__psimd(size_t n, const float* x, float* y);
void nnp_inplace_softmax__psimd(size_t n, float* v);

void nnp_softmax__scalar(size_t n, const float* x, float* y);
void nnp_inplace_softmax__scalar(size_t n, float* v);

#ifdef __cplusplus
} /* extern "C" */
#endif
