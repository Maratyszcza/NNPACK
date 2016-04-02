#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include <nnpack.h>
#include <nnpack/system.h>
#include <nnpack/utils.h>
#include <nnpack/simd.h>

#include <nnpack/validation.h>
#include <nnpack/blas.h>

struct NNP_CACHE_ALIGN fully_connected_inference_context {
	size_t input_channels;
	const float* input;
	const float* kernel;
	float* output;
	nnp_sdotxf_function sdotxf[8];
};

static void compute_fully_connected_inference(
	const struct fully_connected_inference_context context[restrict static 1],
	size_t output_channels_subblock_start, size_t output_channels_subblock_size)
{
	const size_t input_channels      = context->input_channels;
	const float* input               = context->input;
	const float* kernel              = context->kernel;
	float* output                    = context->output;
	const nnp_sdotxf_function sdotxf = context->sdotxf[output_channels_subblock_size - 1];

	sdotxf(input, &kernel[output_channels_subblock_start * input_channels], input_channels, &output[output_channels_subblock_start], input_channels);
}

enum nnp_status nnp_fully_connected_inference(
	size_t input_channels,
	size_t output_channels,
	const float input[],
	const float kernel[],
	float output[],
	pthreadpool_t threadpool)
{
	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_fully_connected_arguments(1, input_channels, output_channels);
	if (status != nnp_status_success) {
		return status;
	}

	/* Do the computation */
	const size_t output_channels_subblock_max = 8;
	struct fully_connected_inference_context fully_connected_inference_context = {
		.input_channels = input_channels,
		.input = input,
		.kernel = kernel,
		.output = output,
		.sdotxf = {
#if NNP_ARCH_X86_64
			[0] = nnp_sdotxf1__avx2,
			[1] = nnp_sdotxf2__avx2,
			[2] = nnp_sdotxf3__avx2,
			[3] = nnp_sdotxf4__avx2,
			[4] = nnp_sdotxf5__avx2,
			[5] = nnp_sdotxf6__avx2,
			[6] = nnp_sdotxf7__avx2,
			[7] = nnp_sdotxf8__avx2,
#endif
		},
	};
	pthreadpool_compute_1d_tiled(threadpool,
		(pthreadpool_function_1d_tiled_t) compute_fully_connected_inference,
		&fully_connected_inference_context,
		output_channels, output_channels_subblock_max);

	return nnp_status_success;
}
