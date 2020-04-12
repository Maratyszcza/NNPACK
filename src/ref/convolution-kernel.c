#include <nnpack.h>
#include <nnpack/reference.h>

struct convolution_kernel_gradient_context {
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size kernel_size;
	struct nnp_size output_size;
	const float* input_pointer;
	const float* grad_output_pointer;
	float* grad_kernel_pointer;
};

static void compute_convolution_kernel_gradient(
	const struct convolution_kernel_gradient_context context[restrict static 1],
	size_t output_channel, size_t input_channel)
{
	const size_t batch_size                = context->batch_size;
	const size_t input_channels            = context->input_channels;
	const size_t output_channels           = context->output_channels;
	const struct nnp_size input_size       = context->input_size;
	const struct nnp_padding input_padding = context->input_padding;
	const struct nnp_size kernel_size      = context->kernel_size;
	const struct nnp_size output_size      = context->output_size;

	const float (*input)[input_channels][input_size.height][input_size.width] =
		(const float(*)[input_channels][input_size.height][input_size.width]) context->input_pointer;
	const float (*grad_output)[output_channels][output_size.height][output_size.width] =
		(const float(*)[output_channels][output_size.height][output_size.width]) context->grad_output_pointer;
	float (*grad_kernel)[input_channels][kernel_size.height][kernel_size.width] =
		(float(*)[input_channels][kernel_size.height][kernel_size.width]) context->grad_kernel_pointer;

	for (size_t y = 0; y < kernel_size.height; y++) {
		for (size_t x = 0; x < kernel_size.width; x++) {
			double grad_kernel_yx = 0.0;
			for (size_t sample = 0; sample < batch_size; sample++) {
				for (size_t i = 0; i < output_size.height; i++) {
					const size_t s = y + i - input_padding.top;
					if (s < input_size.height) {
						for (size_t j = 0; j < output_size.width; j++) {
							const size_t t = x + j - input_padding.left;
							if (t < input_size.width) {
								grad_kernel_yx += input[sample][input_channel][s][t] * grad_output[sample][output_channel][i][j];
							}
						}
					}
				}
			}
			grad_kernel[output_channel][input_channel][y][x] = grad_kernel_yx;
		}
	}
}

struct convolution_kernel_update_context {
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size kernel_size;
	struct nnp_size output_size;
	const float* input_pointer;
	const float* grad_output_pointer;
	float* kernel_pointer;
	float scale;
};

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
	pthreadpool_t threadpool)
{
	const struct nnp_size output_size = {
		.width = input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1,
		.height = input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1
	};
	struct convolution_kernel_gradient_context convolution_kernel_gradient_context = {
		.batch_size = batch_size,
		.input_channels = input_channels,
		.output_channels = output_channels,
		.input_size = input_size,
		.input_padding = input_padding,
		.kernel_size = kernel_size,
		.output_size = output_size,
		.input_pointer = input,
		.grad_output_pointer = grad_output,
		.grad_kernel_pointer = grad_kernel,
	};

	pthreadpool_parallelize_2d(threadpool,
		(pthreadpool_function_2d_t) compute_convolution_kernel_gradient,
		&convolution_kernel_gradient_context,
		output_channels, input_channels,
		PTHREADPOOL_FLAG_DISABLE_DENORMALS);
}
