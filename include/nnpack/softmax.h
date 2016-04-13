#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nnp_exp_function)(size_t, const float*, float*);

void nnp_vector_exp__psimd(size_t n, const float* x, float* y);

typedef void (*nnp_inplace_softmax_function)(size_t, float*);
typedef void (*nnp_outplace_softmax_function)(size_t, const float*, float*);

void nnp_inplace_softmax__psimd(size_t n, float* v);
void nnp_outplace_softmax__psimd(size_t n, const float* x, float* y);

#ifdef __cplusplus
} /* extern "C" */
#endif
