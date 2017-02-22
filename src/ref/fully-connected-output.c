#include <fp16.h>

#include <nnpack.h>
#include <nnpack/reference.h>

struct fully_connected_output_context {
	size_t input_channels;
	size_t output_channels;
	const void* input_pointer;
	const void* kernel_pointer;
	void* output_pointer;
};

static void compute_fully_connected_output_f32(
	const struct fully_connected_output_context* context,
	size_t sample, size_t output_channel)
{
	const size_t input_channels = context->input_channels;
	const size_t output_channels = context->output_channels;

	const float (*input)[input_channels] = (const float(*)[input_channels]) context->input_pointer;
	const float (*kernel)[input_channels] = (const float(*)[input_channels]) context->kernel_pointer;
	float (*output)[output_channels] = (float(*)[output_channels]) context->output_pointer;

	double v = 0.0;
	for (size_t input_channel = 0; input_channel < input_channels; input_channel++) {
		v += (double) input[sample][input_channel] * (double) kernel[output_channel][input_channel];
	}
	output[sample][output_channel] = v;
}

static void compute_fully_connected_output_f16f32(
	const struct fully_connected_output_context* context,
	size_t sample, size_t output_channel)
{
	const size_t input_channels = context->input_channels;
	const size_t output_channels = context->output_channels;

	const float (*input)[input_channels] = (const float(*)[input_channels]) context->input_pointer;
	const uint16_t (*kernel)[input_channels] = (const uint16_t(*)[input_channels]) context->kernel_pointer;
	float (*output)[output_channels] = (float(*)[output_channels]) context->output_pointer;

	double v = 0.0;
	for (size_t input_channel = 0; input_channel < input_channels; input_channel++) {
		v += (double) input[sample][input_channel] *
			(double) fp16_alt_to_fp32_value(kernel[output_channel][input_channel]);
	}
	output[sample][output_channel] = v;
}

void nnp_fully_connected_output_f32__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const float* kernel,
	float* output,
	pthreadpool_t threadpool)
{
	struct fully_connected_output_context fully_connected_output_context = {
		.input_channels = input_channels,
		.output_channels = output_channels,
		.input_pointer = input,
		.kernel_pointer = kernel,
		.output_pointer = output
	};

	pthreadpool_compute_2d(threadpool,
		(pthreadpool_function_2d_t) compute_fully_connected_output_f32,
		&fully_connected_output_context,
		batch_size, output_channels);
}

void nnp_fully_connected_output_f16f32__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const void* kernel,
	float* output,
	pthreadpool_t threadpool)
{
	struct fully_connected_output_context fully_connected_output_context = {
		.input_channels = input_channels,
		.output_channels = output_channels,
		.input_pointer = input,
		.kernel_pointer = kernel,
		.output_pointer = output
	};

	pthreadpool_compute_2d(threadpool,
		(pthreadpool_function_2d_t) compute_fully_connected_output_f16f32,
		&fully_connected_output_context,
		batch_size, output_channels);
}
