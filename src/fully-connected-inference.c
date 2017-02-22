#include <stddef.h>

#include <nnpack.h>
#include <nnpack/macros.h>

#include <nnpack/hwinfo.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN fully_connected_inference_context {
	size_t input_channels;
	const void* input;
	const void* kernel;
	void* output;
};

static void compute_fully_connected_inference_f32(
	const struct fully_connected_inference_context context[restrict static 1],
	size_t output_channels_subblock_start, size_t output_channels_subblock_size)
{
	const size_t input_channels      = context->input_channels;
	const float* input               = context->input;
	const float* kernel              = context->kernel;
	float* output                    = context->output;
	const nnp_sdotxf_function sdotxf = nnp_hwinfo.sdotxf.functions[output_channels_subblock_size - 1];

	sdotxf(input, &kernel[output_channels_subblock_start * input_channels],
		input_channels, &output[output_channels_subblock_start], input_channels);
}

static void compute_fully_connected_inference_f16f32(
	const struct fully_connected_inference_context context[restrict static 1],
	size_t output_channels_subblock_start, size_t output_channels_subblock_size)
{
	const size_t input_channels        = context->input_channels;
	const float* input                 = context->input;
	const uint16_t* kernel             = context->kernel;
	float* output                      = context->output;
	const nnp_shdotxf_function shdotxf = nnp_hwinfo.shdotxf.functions[output_channels_subblock_size - 1];

	shdotxf(input, &kernel[output_channels_subblock_start * input_channels],
		input_channels, &output[output_channels_subblock_start], input_channels);
}

enum nnp_status nnp_fully_connected_inference(
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const float* kernel,
	float* output,
	pthreadpool_t threadpool)
{
	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_fully_connected_arguments(1, input_channels, output_channels);
	if (status != nnp_status_success) {
		return status;
	}

	/* Do the computation */
	const size_t output_channels_subblock_max = nnp_hwinfo.sdotxf.fusion;
	struct fully_connected_inference_context fully_connected_inference_context = {
		.input_channels = input_channels,
		.input = input,
		.kernel = kernel,
		.output = output,
	};
	pthreadpool_compute_1d_tiled(threadpool,
		(pthreadpool_function_1d_tiled_t) compute_fully_connected_inference_f32,
		&fully_connected_inference_context,
		output_channels, output_channels_subblock_max);

	return nnp_status_success;
}

enum nnp_status nnp_fully_connected_inference_f16f32(
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const void* kernel,
	float* output,
	pthreadpool_t threadpool)
{
	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_fully_connected_arguments(1, input_channels, output_channels);
	if (status != nnp_status_success) {
		return status;
	}

	/* Do the computation */
	const size_t output_channels_subblock_max = nnp_hwinfo.sdotxf.fusion;
	struct fully_connected_inference_context fully_connected_inference_context = {
		.input_channels = input_channels,
		.input = input,
		.kernel = kernel,
		.output = output,
	};
	pthreadpool_compute_1d_tiled(threadpool,
		(pthreadpool_function_1d_tiled_t) compute_fully_connected_inference_f16f32,
		&fully_connected_inference_context,
		output_channels, output_channels_subblock_max);

	return nnp_status_success;
}
