#include <nnpack.h>
#include <nnpack/reference.h>
#include <nnpack/activations.h>


struct relu_input_gradient_context {
	size_t channels;
	const float* grad_output;
	const float* input;
	float* grad_input;
	float negative_slope;
};

static void compute_relu_input_gradient(
	const struct relu_input_gradient_context context[restrict static 1],
	size_t sample)
{
	const size_t channels    = context->channels;
	const float* grad_output = context->grad_output + sample * channels;
	const float* input       = context->input       + sample * channels;
	float* grad_input        = context->grad_input  + sample * channels;
	float negative_slope     = context->negative_slope;

	for (size_t channel = 0; channel < channels; channel++) {
		grad_input[channel] = grad_relu(grad_output[channel], input[channel], negative_slope);
	}
}

void nnp_relu_input_gradient__reference(
	size_t batch_size,
	size_t channels,
	const float grad_output[],
	const float input[],
	float grad_input[],
	float negative_slope,
	pthreadpool_t threadpool)
{
	struct relu_input_gradient_context relu_input_gradient_context = {
		.channels = channels,
		.grad_output = grad_output,
		.input = input,
		.grad_input = grad_input,
		.negative_slope = negative_slope,
	};

	pthreadpool_compute_1d(threadpool,
		(pthreadpool_function_1d_t) compute_relu_input_gradient,
		&relu_input_gradient_context,
		batch_size);
}
