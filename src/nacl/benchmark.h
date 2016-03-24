#pragma once

#include <stdint.h>
#include <nnpack.h>

enum benchmark_type {
	benchmark_type_convolution_forward = 1,
	benchmark_type_batch_transform = 2,
};

extern struct nnp_profile benchmark_convolution_output(
	enum nnp_convolution_algorithm algorithm,
	size_t batch_size, size_t input_dimensions, size_t output_dimensions,
	struct nnp_size input_size, struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	pthreadpool_t threadpool,
	size_t max_iterations);
