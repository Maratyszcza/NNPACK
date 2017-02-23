#include <nnpack.h>
#include <nnpack/reference.h>
#include <nnpack/activations.h>

struct relu_output_context {
	size_t channels;
	const float* input;
	float* output;
	float negative_slope;
};

static void compute_relu_output(
	const struct relu_output_context context[restrict static 1],
	size_t sample)
{
	const size_t channels = context->channels;
	const float* input    = context->input  + sample * channels;
	float* output         = context->output + sample * channels;
	float negative_slope  = context->negative_slope;

	for (size_t channel = 0; channel < channels; channel++) {
		output[channel] = relu(input[channel], negative_slope);
	}
}

void nnp_relu_output__reference(
	size_t batch_size,
	size_t channels,
	const float input[],
	float output[],
	float negative_slope,
	pthreadpool_t threadpool)
{
	struct relu_output_context relu_output_context = {
		.channels = channels,
		.input = input,
		.output = output,
		.negative_slope = negative_slope,
	};

	pthreadpool_compute_1d(threadpool,
		(pthreadpool_function_1d_t) compute_relu_output,
		&relu_output_context,
		batch_size);
}
