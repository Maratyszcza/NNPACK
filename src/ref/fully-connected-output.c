#include <nnpack.h>
#include <nnpack/reference.h>

struct fully_connected_output_context {
	size_t input_channels;
	size_t output_channels;
	const float* input_pointer;
	const float* kernel_pointer;
	float* output_pointer;
};

static void compute_fully_connected_output(
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
		v += input[sample][input_channel] * kernel[output_channel][input_channel];
	}
	output[sample][output_channel] = v;
}

void nnp_fully_connected_output__reference(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	const float input_pointer[],
	const float kernel_pointer[],
	float output_pointer[],
	pthreadpool_t threadpool)
{
	struct fully_connected_output_context fully_connected_output_context = {
		.input_channels = input_channels,
		.output_channels = output_channels,
		.input_pointer = input_pointer,
		.kernel_pointer = kernel_pointer,
		.output_pointer = output_pointer
	};

	pthreadpool_compute_2d(threadpool,
		(pthreadpool_function_2d_t) compute_fully_connected_output,
		&fully_connected_output_context,
		batch_size, output_channels);
}
