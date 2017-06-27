#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void nnp_relu__avx2(const float* input, float* output, size_t length, float negative_slope);
void nnp_inplace_relu__avx2(float* data, size_t length, float negative_slope);
void nnp_grad_relu__avx2(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

void nnp_relu__neon(const float* input, float* output, size_t length, float negative_slope);
void nnp_inplace_relu__neon(float* data, size_t length, float negative_slope);
void nnp_grad_relu__neon(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

void nnp_relu__psimd(const float* input, float* output, size_t length, float negative_slope);
void nnp_inplace_relu__psimd(float* data, size_t length, float negative_slope);
void nnp_grad_relu__psimd(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

void nnp_relu__scalar(const float* input, float* output, size_t length, float negative_slope);
void nnp_inplace_relu__scalar(float* data, size_t length, float negative_slope);
void nnp_grad_relu__scalar(const float* output_gradient, const float* input, float* input_gradient, size_t length, float negative_slope);

#ifdef __cplusplus
} /* extern "C" */
#endif
