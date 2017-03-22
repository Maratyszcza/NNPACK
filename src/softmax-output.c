#include <stddef.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <nnpack/hwinfo.h>
#include <nnpack/softmax.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN inplace_softmax_context {
	nnp_inplace_softmax_function softmax_function;
	size_t channels;
	float* data;
};

static void compute_inplace_softmax_output(
	const struct inplace_softmax_context context[restrict static 1],
	size_t sample)
{
	const nnp_inplace_softmax_function softmax_function = context->softmax_function;
	const size_t channels                               = context->channels;

	float (*data)[channels] =
		(float(*)[channels]) context->data;

	softmax_function(channels, data[sample]);
}

struct NNP_CACHE_ALIGN outplace_softmax_context {
	nnp_outplace_softmax_function softmax_function;
	size_t channels;
	const float* input;
	float* output;
};

static void compute_outplace_softmax_output(
	const struct outplace_softmax_context context[restrict static 1],
	size_t sample)
{
	const nnp_outplace_softmax_function softmax_function = context->softmax_function;
	const size_t channels                                = context->channels;

	const float (*input)[channels] =
		(const float(*)[channels]) context->input;
	float (*output)[channels] =
		(float(*)[channels]) context->output;

	softmax_function(channels, input[sample], output[sample]);
}

enum nnp_status nnp_softmax_output(
	size_t batch_size,
	size_t channels,
	const float* input,
	float* output,
	pthreadpool_t threadpool)
{
	enum nnp_status status = validate_softmax_arguments(batch_size, channels);
	if (status != nnp_status_success) {
		return status;
	}

	if (input == output) {
		/* In-place softmax */
		struct inplace_softmax_context inplace_softmax_context = {
			.softmax_function = nnp_hwinfo.activations.inplace_softmax,
			.channels = channels,
			.data = output,
		};
		pthreadpool_compute_1d(threadpool,
			(pthreadpool_function_1d_t) compute_inplace_softmax_output,
			&inplace_softmax_context,
			batch_size);
	} else {
		/* Out-of-place softmax */
		struct outplace_softmax_context outplace_softmax_context = {
			.softmax_function = nnp_hwinfo.activations.outplace_softmax,
			.channels = channels,
			.input = input,
			.output = output,
		};
		pthreadpool_compute_1d(threadpool,
			(pthreadpool_function_1d_t) compute_outplace_softmax_output,
			&outplace_softmax_context,
			batch_size);
	}

	return nnp_status_success;
}
