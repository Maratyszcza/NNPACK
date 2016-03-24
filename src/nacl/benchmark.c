#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <nnpack.h>

#include <nacl/benchmark.h>

static int compare_profile(const void *a_ptr, const void *b_ptr) {
	const double a_total = ((const struct nnp_profile*) a_ptr)->total;
	const double b_total = ((const struct nnp_profile*) b_ptr)->total;
	if (a_total < b_total) {
		return -1;
	} else if (a_total > b_total) {
		return 1;
	} else {
		return 0;
	}
}

static inline struct nnp_profile average_profile(struct nnp_profile a, struct nnp_profile b) {
	return (struct nnp_profile) {
		.total = 0.5 * (a.total + b.total),
		.input_transform = 0.5 * (a.input_transform + b.input_transform),
		.kernel_transform = 0.5 * (a.kernel_transform + b.kernel_transform),
		.output_transform = 0.5 * (a.output_transform + b.output_transform),
		.block_multiplication = 0.5 * (a.block_multiplication + b.block_multiplication)
	};
}

struct nnp_profile median_profile(struct nnp_profile array[], size_t length) {
	qsort(array, length, sizeof(struct nnp_profile), &compare_profile);
	if (length % 2 == 0) {
		return average_profile(array[length / 2 - 1], array[length / 2]);
	} else {
		return array[length / 2];
	}
}

struct nnp_profile benchmark_convolution_output(
	enum nnp_convolution_algorithm algorithm,
	size_t batch_size,
	size_t input_dimensions,
	size_t output_dimensions,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	pthreadpool_t threadpool,
	size_t max_iterations)
{
	const struct nnp_size output_size = {
		.width = input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1,
		.height = input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1
	};

	void* input = malloc(batch_size * input_dimensions * input_size.width * input_size.height * sizeof(float));
	void* kernel = malloc(input_dimensions * output_dimensions * kernel_size.width * kernel_size.height * sizeof(float));
	void* output = malloc(batch_size * output_dimensions * output_size.width * output_size.height * sizeof(float));
	void* bias = malloc(output_dimensions * sizeof(float));

	memset(input, 0, batch_size * input_dimensions * input_size.width * input_size.height * sizeof(float));
	memset(kernel, 0, input_dimensions * output_dimensions * kernel_size.width * kernel_size.height * sizeof(float));
	memset(output, 0, batch_size * output_dimensions * output_size.width * output_size.height * sizeof(float));
	memset(bias, 0, output_dimensions * sizeof(float));

	struct nnp_profile computation_profile[max_iterations];
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		nnp_convolution_output(
			algorithm,
			batch_size, input_dimensions, output_dimensions,
			input_size, input_padding,
			kernel_size,
			input, kernel, bias, output,
			threadpool,
			&computation_profile[iteration]);
	}

	free(input);
	free(kernel);
	free(output);
	free(bias);

	return median_profile(computation_profile, max_iterations);
}
