#pragma once

#include <stddef.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

void nnp_convolution_output__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	struct nnp_size output_subsampling,
	const float input_pointer[],
	const float kernel_pointer[],
	const float bias[],
	float output_pointer[],
	pthreadpool_t threadpool);

void nnp_convolution_input_gradient__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	const float grad_output[],
	const float kernel[],
	float grad_input[],
	pthreadpool_t threadpool);

void nnp_convolution_kernel_gradient__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	const float input[],
	const float grad_output[],
	float grad_kernel[],
	pthreadpool_t threadpool);

void nnp_fully_connected_output_f32__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const float* kernel,
	float* output,
	pthreadpool_t threadpool);

void nnp_fully_connected_output_f16f32__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const void* kernel,
	float* output,
	pthreadpool_t threadpool);

void nnp_max_pooling_output__reference(
	size_t batch_size,
	size_t channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size pooling_size,
	struct nnp_size pooling_stride,
	const float input[],
	float output[],
	pthreadpool_t threadpool);

void nnp_relu_output__reference(
	size_t batch_size,
	size_t channels,
	const float input[],
	float output[],
	float negative_slope,
	pthreadpool_t threadpool);

void nnp_relu_input_gradient__reference(
	size_t batch_size,
	size_t channels,
	const float grad_output[],
	const float input[],
	float grad_input[],
	float negative_slope,
	pthreadpool_t threadpool);

void nnp_softmax_output__reference(
    size_t batch_size,
    size_t channels,
    const float input[],
    float output[],
    pthreadpool_t threadpool);

#ifdef __cplusplus
} /* extern "C" */
#endif
