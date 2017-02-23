#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <nnpack/hwinfo.h>
#include <nnpack/activations.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN inplace_relu_context {
	nnp_inplace_relu_function relu_function;
	float* data;
	float negative_slope;
};

static void compute_inplace_relu_output(
	const struct inplace_relu_context context[restrict static 1],
	size_t block_start, size_t block_size)
{
	nnp_inplace_relu_function relu_function = context->relu_function;
	float* data                             = context->data;
	float negative_slope                    = context->negative_slope;

	relu_function(data + block_start, block_size, negative_slope);
}

struct NNP_CACHE_ALIGN outplace_relu_context {
	nnp_outplace_relu_function relu_function;
	const float* input;
	float* output;
	float negative_slope;
};

static void compute_outplace_relu_output(
	const struct outplace_relu_context context[restrict static 1],
	size_t block_start, size_t block_size)
{
	nnp_outplace_relu_function relu_function = context->relu_function;
	const float* input                       = context->input;
	float* output                            = context->output;
	float negative_slope                     = context->negative_slope;

	relu_function(input + block_start, output + block_start, block_size, negative_slope);
}

enum nnp_status nnp_relu_output(
	size_t batch_size,
	size_t channels,
	const float input[],
	float output[],
	float negative_slope,
	pthreadpool_t threadpool)
{
	enum nnp_status status = validate_relu_arguments(batch_size, channels);
	if (status != nnp_status_success) {
		return status;
	}

	size_t elements = batch_size * channels;
	const size_t simd_width = nnp_hwinfo.simd_width;

	assert(((uintptr_t) input) % sizeof(float) == 0);
	assert(((uintptr_t) output) % sizeof(float) == 0);

	const size_t prologue_elements = min((size_t) (-(((uintptr_t) output) / sizeof(float)) % simd_width), elements);
	for (size_t i = 0; i < prologue_elements; i++) {
		output[i] = relu(input[i], negative_slope);
	}
	elements -= prologue_elements;
	input += prologue_elements;
	output += prologue_elements;

	const size_t epilogue_elements = elements % simd_width;
	for (size_t i = 0; i < epilogue_elements; i++) {
		output[elements - epilogue_elements + i] =
			relu(input[elements - epilogue_elements + i], negative_slope);
	}
	elements -= epilogue_elements;

	if (input == output) {
		/* In-place transformation */
		struct inplace_relu_context inplace_relu_context = {
			.relu_function = nnp_hwinfo.activations.inplace_relu,
			.data = output,
			.negative_slope = negative_slope,
		};

		pthreadpool_compute_1d_tiled(threadpool,
			(pthreadpool_function_1d_tiled_t) compute_inplace_relu_output,
			&inplace_relu_context,
			elements, round_down(nnp_hwinfo.blocking.l1 / sizeof(float), simd_width));
	} else {
		/* Out-of-place transformation */
		struct outplace_relu_context outplace_relu_context = {
			.relu_function = nnp_hwinfo.activations.outplace_relu,
			.input = input,
			.output = output,
			.negative_slope = negative_slope,
		};

		pthreadpool_compute_1d_tiled(threadpool,
			(pthreadpool_function_1d_tiled_t) compute_outplace_relu_output,
			&outplace_relu_context,
			elements, round_down(nnp_hwinfo.blocking.l1 / sizeof(float), simd_width));
	}

	return nnp_status_success;
}
