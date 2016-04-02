#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include <nnpack.h>
#include <nnpack/pooling.h>
#include <nnpack/utils.h>

#include <nnpack/validation.h>

struct NNP_CACHE_ALIGN pooling_context {
	nnp_pooling_function pooling_function;
	const float* input_pointer;
	float* output_pointer;

	size_t channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size pooling_stride;
	struct nnp_size output_size;
	struct nnp_size input_tile;
	struct nnp_size output_tile;
};

static void compute_pooling_output(
	const struct pooling_context context[restrict static 1],
	size_t sample, size_t channel)
{
	const size_t channels                  = context->channels;
	const struct nnp_size input_size       = context->input_size;
	const struct nnp_padding input_padding = context->input_padding;
	const struct nnp_size pooling_stride   = context->pooling_stride;
	const struct nnp_size output_size      = context->output_size;
	const struct nnp_size input_tile       = context->input_tile;
	const struct nnp_size output_tile      = context->output_tile;
	const nnp_pooling_function pooling     = context->pooling_function;

	const float (*input)[channels][input_size.height][input_size.width] =
		(const float(*)[channels][input_size.height][input_size.width]) context->input_pointer;
	float (*output)[channels][output_size.height][output_size.width] =
		(float(*)[channels][output_size.height][output_size.width]) context->output_pointer;

	for (size_t y = 0; y < output_size.height; y += output_tile.height) {
		const size_t input_y = min(doz(y * pooling_stride.height, input_padding.top), input_size.height);
		const size_t input_row_offset = doz(input_padding.top, y);
		const size_t input_row_count = min(input_tile.height, doz(input_size.height, input_y));
		const size_t output_row_count = min(output_tile.height, output_size.height - y);
		for (size_t x = 0; x < output_size.width; x += output_tile.width) {
			const size_t input_x = min(doz(x * pooling_stride.width, input_padding.left), input_size.width);
			const size_t input_column_offset = doz(input_padding.left, x);
			const size_t input_column_count = min(input_tile.width, doz(input_size.width, input_x));
			const size_t output_column_count = min(output_tile.width, output_size.width - x);
			pooling(
				&input[sample][channel][input_y][input_x],
				&output[sample][channel][y][x],
				input_size.width,
				input_row_offset,
				input_row_count,
				input_column_offset,
				input_column_count,
				output_column_count);
		}
	}
}

enum nnp_status nnp_max_pooling_output(
	size_t batch_size,
	size_t channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size pooling_size,
	struct nnp_size pooling_stride,
	const float input[],
	float output[],
	pthreadpool_t threadpool)
{
	enum nnp_status status = validate_pooling_arguments(
		batch_size, channels,
		input_size, input_padding,
		pooling_size, pooling_stride);
	if (status != nnp_status_success) {
		return status;
	}

	const struct nnp_size output_size = {
		.height = divide_round_up(input_padding.top + input_size.height + input_padding.bottom - pooling_size.height, pooling_stride.height) + 1,
		.width = divide_round_up(input_padding.left + input_size.width + input_padding.right - pooling_size.width, pooling_stride.width) + 1,
	};

	struct pooling_context pooling_context = {
		.channels = channels,
		.input_pointer = input,
		.output_pointer = output,
		.input_size = input_size,
		.pooling_stride = pooling_stride,
		.output_size = output_size,
	};

	if ((pooling_stride.height != 2) || (pooling_stride.width != 2)) {
		return nnp_status_unsupported_pooling_stride;
	}

	if (pooling_size.width != pooling_size.height) {
		return nnp_status_unsupported_pooling_size;
	}
	switch (pooling_size.width) {
#if NNP_ARCH_X86_64
		case 2:
			pooling_context.pooling_function = nnp_maxpool_2x2_2x2__avx2;
			pooling_context.input_tile = (struct nnp_size) { .height = 2, .width = 16 };
			pooling_context.output_tile = (struct nnp_size) { .height = 1, .width = 8 };
			break;
#endif
		case 3:
			return nnp_status_unsupported_pooling_size;
		default:
			return nnp_status_unsupported_pooling_size;
	}

	pthreadpool_compute_2d(threadpool,
		(pthreadpool_function_2d_t) compute_pooling_output,
		&pooling_context,
		batch_size, channels);

	return nnp_status_success;
}
