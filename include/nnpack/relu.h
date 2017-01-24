#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nnp_inplace_relu_function)(float*, size_t, float);
typedef void (*nnp_outplace_relu_function)(const float*, float*, size_t, float);
typedef void (*nnp_gradient_relu_function)(const float*, const float*, float*, size_t, float);

void nnp_inplace_relu_forward__avx2(float* data, size_t length, float negative_slope);
void nnp_outplace_relu_forward__avx2(const float* input, float* output, size_t length, float negative_slope);
void nnp_relu_backward__avx2(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

void nnp_inplace_relu_forward__psimd(float* data, size_t length, float negative_slope);
void nnp_outplace_relu_forward__psimd(const float* input, float* output, size_t length, float negative_slope);
void nnp_relu_backward__psimd(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

void nnp_inplace_relu_forward__scalar(float* data, size_t length, float negative_slope);
void nnp_outplace_relu_forward__scalar(const float* input, float* output, size_t length, float negative_slope);
void nnp_relu_backward__scalar(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

#ifdef __cplusplus
} /* extern "C" */
#endif
