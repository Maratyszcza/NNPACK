#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <fxdiv.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>
#include <nnpack/system.h>

#include <nnpack/hwinfo.h>
#include <nnpack/activations.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN kernel_transform_context {
	nnp_transform_2d_with_offset transform_function;
	const float* kernel;
	void* kernel_transform;

	size_t tuple_size;
	size_t input_channels;
	size_t input_channels_block_size;
	size_t output_channels;
	struct nnp_size kernel_size;
};

static void compute_kernel_transform(
	const struct kernel_transform_context context[restrict static 1],
	size_t output_channels_subblock_start, size_t input_channels_block_offset,
	size_t output_channels_subblock_size,  size_t input_channels_block_increment)
{
	const size_t tuple_size                 = context->tuple_size;
	const size_t input_channels             = context->input_channels;
	const size_t input_channels_block_size  = context->input_channels_block_size;
	const size_t output_channels            = context->output_channels;
	const struct nnp_size kernel_size       = context->kernel_size;

	const float (*kernel)[input_channels][kernel_size.width * kernel_size.height] =
		(const float(*)[input_channels][kernel_size.width * kernel_size.height]) context->kernel;
	void* kernel_transform                          = context->kernel_transform;
	nnp_transform_2d_with_offset transform_function = context->transform_function;

	for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform_function(
			kernel[output_channel][input_channels_block_offset],
			kernel_transform +
				(output_channels_subblock_start * input_channels_block_size + input_channels_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
			kernel_size.width,
			input_channels_block_size * output_channels * tuple_size,
			kernel_size.height, kernel_size.width, 0, 0);
	}
}

struct NNP_CACHE_ALIGN input_transform_context {
	const float* input;
	void* input_transform;
	nnp_transform_2d_with_offset transform_function;

	const size_t tuple_size;
	const size_t tiles_count;
	const struct fxdiv_divisor_size_t tiles_x_count;
	const size_t input_channels_block_start;
	const size_t input_channels_block_size;
	const struct nnp_size input_size;
	const size_t input_padding_left;
	const size_t input_padding_top;
	const struct nnp_size input_tile;
	const struct nnp_size input_tile_step;
};

static void compute_input_transform(
	const struct input_transform_context context[restrict static 1],
	size_t input_channels_block_offset, size_t tiles_subblock_start,
	size_t input_channels_block_range,  size_t tiles_subblock_size)
{
	const size_t tuple_size                         = context->tuple_size;
	const size_t tiles_count                        = context->tiles_count;
	const struct fxdiv_divisor_size_t tiles_x_count = context->tiles_x_count;
	const size_t input_channels_block_start         = context->input_channels_block_start;
	const size_t input_channels_block_size          = context->input_channels_block_size;
	const struct nnp_size input_size                = context->input_size;
	const size_t input_padding_left                 = context->input_padding_left;
	const size_t input_padding_top                  = context->input_padding_top;
	const struct nnp_size input_tile                = context->input_tile;
	const struct nnp_size input_tile_step           = context->input_tile_step;

	const float (*input)[input_size.height][input_size.width] =
		(const float(*)[input_size.height][input_size.width]) context->input;
	void* input_transform                           = context->input_transform;
	nnp_transform_2d_with_offset transform_function = context->transform_function;

	const size_t input_channel = input_channels_block_start + input_channels_block_offset;
	for (size_t tiles_subblock_offset = 0; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset += 1) {
		const size_t tile = tiles_subblock_start + tiles_subblock_offset;
		const struct fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
		const size_t tile_x = tile_xy.remainder;
		const size_t tile_y = tile_xy.quotient;

		const size_t output_x = tile_x * input_tile_step.width;
		const size_t output_y = tile_y * input_tile_step.height;

		const size_t input_x = min(doz(output_x, input_padding_left), input_size.width);
		const size_t input_y = min(doz(output_y, input_padding_top), input_size.height);

		const size_t row_offset = doz(input_padding_top, output_y);
		const size_t row_count = min(input_size.height - input_y, input_tile.height - row_offset);
		const size_t column_offset = doz(input_padding_left, output_x);
		const size_t column_count = min(input_size.width - input_x, input_tile.width - column_offset);

		transform_function(
			&input[input_channel][input_y][input_x],
			input_transform + (tiles_subblock_start * input_channels_block_size + input_channels_block_offset * tiles_subblock_size + tiles_subblock_offset) * tuple_size,
			input_size.width,
			input_channels_block_size * tiles_count * tuple_size,
			row_count, column_count, row_offset, column_offset);
	}
}

struct NNP_CACHE_ALIGN output_transform_context {
	nnp_transform_2d_with_bias transform_function;
	float* output;
	const void* output_transform;
	const float* bias;

	size_t tuple_size;
	size_t tiles_count;
	struct fxdiv_divisor_size_t tiles_x_count;
	struct fxdiv_divisor_size_t tiles_block_max;
	size_t output_channels;
	struct nnp_size output_size;
	struct nnp_size output_tile;
};

static void compute_output_transform(
	const struct output_transform_context context[restrict static 1],
	size_t output_channels_subblock_start, size_t tiles_subblock_start,
	size_t output_channels_subblock_size,  size_t tiles_subblock_size)
{
	const size_t tuple_size                           = context->tuple_size;
	const size_t tiles_count                          = context->tiles_count;
	const struct fxdiv_divisor_size_t tiles_x_count   = context->tiles_x_count;
	const struct fxdiv_divisor_size_t tiles_block_max = context->tiles_block_max;
	const size_t output_channels                      = context->output_channels;
	const struct nnp_size output_size                 = context->output_size;
	const struct nnp_size output_tile                 = context->output_tile;

	const size_t tiles_block_start = fxdiv_round_down_size_t(tiles_subblock_start, tiles_block_max);
	const size_t tiles_block_size = min(tiles_count - tiles_block_start, tiles_block_max.value);

	float (*output)[output_size.height][output_size.width] =
		(float(*)[output_size.height][output_size.width]) context->output;
	const void* output_transform                  = context->output_transform;
	const float* bias                             = context->bias;
	nnp_transform_2d_with_bias transform_function = context->transform_function;

	for (size_t tiles_subblock_offset = 0; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset += 1) {
		const size_t tile = tiles_subblock_start + tiles_subblock_offset;
		const struct fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
		const size_t tile_x = tile_xy.remainder;
		const size_t tile_y = tile_xy.quotient;

		const size_t output_x = tile_x * output_tile.width;
		const size_t output_y = tile_y * output_tile.height;

		for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
			const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
			transform_function(
				output_transform +
					(tiles_block_start * output_channels + output_channels_subblock_start * tiles_block_size + ((tiles_subblock_start - tiles_block_start) + tiles_subblock_offset) * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
				&output[output_channel][output_y][output_x],
				&bias[output_channel],
				tiles_count * output_channels * tuple_size,
				output_size.width,
				min(output_tile.height, output_size.height - output_y),
				min(output_tile.width, output_size.width - output_x));
		}
	}
}

struct NNP_CACHE_ALIGN tuple_multiplication_context {
	size_t tuple_elements;
	size_t tuple_size;
	size_t tiles_subblock_max;
	size_t input_channels_block_size;
	size_t input_channels_block_start;
	size_t output_channels;
	size_t output_channels_subblock_max;
	size_t output_channels_block_start;

	const void* input_transform;
	const void* kernel_transform;
	void* output_transform;

	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_tuple_multiplication(
	const struct tuple_multiplication_context context[restrict static 1],
	size_t tiles_block_start, size_t output_channels_subblock_start,
	size_t tiles_block_size,  size_t output_channels_subblock_size)
{
	const size_t tuple_elements               = context->tuple_elements;
	const size_t tuple_size                   = context->tuple_size;
	const size_t tiles_subblock_max           = context->tiles_subblock_max;
	const size_t input_channels_block_size    = context->input_channels_block_size;
	const size_t input_channels_block_start   = context->input_channels_block_start;
	const size_t output_channels              = context->output_channels;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;
	const size_t output_channels_block_start  = context->output_channels_block_start;

	const void* input_transform  = context->input_transform +
		tiles_block_start * input_channels_block_size * tuple_size;
	const void* kernel_transform = context->kernel_transform +
		(output_channels_block_start + output_channels_subblock_start) * input_channels_block_size * tuple_size;
	void* output_transform       = context->output_transform +
		(tiles_block_start * output_channels + (output_channels_block_start + output_channels_subblock_start) * tiles_block_size) * tuple_size;

	if (output_channels_subblock_size == output_channels_subblock_max) {
		const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
		while (tiles_block_size >= tiles_subblock_max) {
			tiles_block_size -= tiles_subblock_max;

			fast_gemm(
				input_channels_block_size, input_channels_block_start,
				input_transform, kernel_transform, output_transform,
				output_channels_subblock_size * tuple_elements);

			input_transform  += tiles_subblock_max * input_channels_block_size * tuple_size;
			output_transform += tiles_subblock_max * output_channels_subblock_size * tuple_size;
		}
	}

	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
	while (tiles_block_size != 0) {
		const size_t tiles_subblock_size = min(tiles_block_size, tiles_subblock_max);
		tiles_block_size -= tiles_subblock_size;

		full_gemm(
			tiles_subblock_size, output_channels_subblock_size,
			input_channels_block_size, input_channels_block_start,
			input_transform, kernel_transform, output_transform,
			output_channels_subblock_size * tuple_elements);

		input_transform  += tiles_subblock_max * input_channels_block_size * tuple_size;
		output_transform += tiles_subblock_max * output_channels_subblock_size * tuple_size;
	}
}

struct NNP_CACHE_ALIGN kernel_packing_context {
	const float* kernel;
	float* packed_kernel;

	size_t reduction_size;
	size_t reduction_block_start;
	size_t reduction_block_size;
};

static void compute_kernel_packing(
	const struct kernel_packing_context context[restrict static 1],
	size_t output_channels_subblock_start, size_t reduction_block_offset,
	size_t output_channels_subblock_size,  size_t reduction_block_range)
{
	const size_t reduction_size        = context->reduction_size;
	const size_t reduction_block_start = context->reduction_block_start;
	const size_t reduction_block_size  = context->reduction_block_size;

	const float* kernel  = context->kernel +
		output_channels_subblock_start * reduction_size + reduction_block_offset;
	float* packed_kernel = context->packed_kernel +
		output_channels_subblock_start * reduction_block_size + reduction_block_offset * output_channels_subblock_size;

	for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
		packed_kernel[output_channels_subblock_offset] = kernel[output_channels_subblock_offset * reduction_size];
	}
}

struct NNP_CACHE_ALIGN input_packing_context {
	const float* input;
	float* packed_input;

	size_t simd_width;
	size_t reduction_block_start;
	size_t reduction_block_size;
	size_t output_image_block_start;
	struct nnp_size input_size;
	size_t input_padding_top;
	size_t input_padding_left;
	struct fxdiv_divisor_size_t kernel_elements;
	struct fxdiv_divisor_size_t kernel_width;
	struct fxdiv_divisor_size_t output_width;
	struct nnp_size output_subsampling;
};

static void compute_input_packing(
	const struct input_packing_context context[restrict static 1],
	size_t reduction_block_offset, size_t output_image_subblock_start,
	size_t reduction_block_range,  size_t output_image_subblock_size)
{
	const size_t simd_width                           = context->simd_width;
	const size_t reduction_block_start                = context->reduction_block_start;
	const size_t reduction_block_size                 = context->reduction_block_size;
	const size_t output_image_block_start             = context->output_image_block_start;
	const struct nnp_size input_size                  = context->input_size;
	const size_t input_padding_top                    = context->input_padding_top;
	const size_t input_padding_left                   = context->input_padding_left;
	const struct fxdiv_divisor_size_t kernel_elements = context->kernel_elements;
	const struct fxdiv_divisor_size_t kernel_width    = context->kernel_width;
	const struct fxdiv_divisor_size_t output_width    = context->output_width;
	const struct nnp_size output_subsampling          = context->output_subsampling;

	const float (*input)[input_size.height][input_size.width] =
		(const float(*)[input_size.height][input_size.width]) context->input;
	float* packed_input = context->packed_input;

	const size_t output_image_subblock_stride = round_up_by_power_of_2(output_image_subblock_size, simd_width);

	const size_t reduction_index = reduction_block_start + reduction_block_offset;
	const struct fxdiv_result_size_t reduction_index_divmod = fxdiv_divide_size_t(reduction_index, kernel_elements);
	const size_t input_channel = reduction_index_divmod.quotient;
	const struct fxdiv_result_size_t kernel_xy = fxdiv_divide_size_t(reduction_index_divmod.remainder, kernel_width);
	const size_t kernel_y = kernel_xy.quotient;
	const size_t kernel_x = kernel_xy.remainder;

	for (size_t output_image_subblock_offset = 0; output_image_subblock_offset < output_image_subblock_size; output_image_subblock_offset += 1) {
		const size_t output_image_index = output_image_block_start + output_image_subblock_start + output_image_subblock_offset;
		const struct fxdiv_result_size_t output_xy = fxdiv_divide_size_t(output_image_index, output_width);
		const size_t output_y = output_xy.quotient;
		const size_t output_x = output_xy.remainder;

		const size_t input_y = output_y * output_subsampling.height + kernel_y - input_padding_top;
		const size_t input_x = output_x * output_subsampling.width  + kernel_x - input_padding_left;

		const size_t packed_index = output_image_subblock_start * reduction_block_size +
			reduction_block_offset * output_image_subblock_stride + output_image_subblock_offset;
		if ((input_x < input_size.width) && (input_y < input_size.height)) {
			packed_input[packed_index] = input[input_channel][input_y][input_x];
		} else {
			packed_input[packed_index] = 0.0f;
		}
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context {
	const float* packed_kernel;
	const float* packed_input;
	float* output;

	size_t reduction_block_start;
	size_t reduction_block_size;
	size_t output_image_size;
	size_t output_image_block_start;
	size_t output_image_subblock_max;
	size_t output_channels_subblock_max;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context context[restrict static 1],
	size_t output_channels_block_start, size_t output_image_subblock_start,
	size_t output_channels_block_size,  size_t output_image_subblock_size)
{
	const size_t reduction_block_start        = context->reduction_block_start;
	const size_t reduction_block_size         = context->reduction_block_size;
	const size_t output_image_size            = context->output_image_size;
	const size_t output_image_block_start     = context->output_image_block_start;
	const size_t output_image_subblock_max    = context->output_image_subblock_max;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;

	const float* packed_kernel = context->packed_kernel +
		output_channels_block_start * reduction_block_size;
	const float* packed_input  = context->packed_input +
		output_image_subblock_start * reduction_block_size;
	float* output              = context->output +
		output_channels_block_start * output_image_size + output_image_block_start + output_image_subblock_start;

	if (output_image_subblock_size == output_image_subblock_max) {
		const nnp_fast_sgemm_function fast_gemm = nnp_hwinfo.sgemm.only_mr_x_nr;
		while (output_channels_block_size >= output_channels_subblock_max) {
			output_channels_block_size -= output_channels_subblock_max;

			fast_gemm(
				reduction_block_size, reduction_block_start,
				packed_kernel, packed_input, output,
				output_image_size);

			packed_kernel += reduction_block_size * output_channels_subblock_max;
			output        += output_image_size    * output_channels_subblock_max;
		}
	}

	const nnp_full_sgemm_function full_gemm = nnp_hwinfo.sgemm.upto_mr_x_nr;
	while (output_channels_block_size != 0) {
		const size_t output_channels_subblock_size = min(output_channels_block_size, output_channels_subblock_max);
		output_channels_block_size -= output_channels_subblock_size;

		full_gemm(
			output_channels_subblock_size, output_image_subblock_size,
			reduction_block_size, reduction_block_start,
			packed_kernel, packed_input, output,
			output_image_size);

		packed_kernel += reduction_block_size * output_channels_subblock_max;
		output        += output_image_size    * output_channels_subblock_max;
	}
}

struct NNP_CACHE_ALIGN direct_convolution_context {
	const float* input;
	const float* kernel;
	float* output;

	size_t image_elements;
	size_t input_channels;
	size_t input_channels_block_max;
	size_t output_channels_block_max;

	nnp_fast_conv_function fast_conv;
	nnp_full_conv_function full_conv;
};

static void compute_direct_convolution(
	const struct direct_convolution_context context[restrict static 1],
	size_t output_channels_block_start, size_t output_channels_block_size)
{
	const size_t image_elements            = context->image_elements;
	const size_t input_channels            = context->input_channels;
	const size_t input_channels_block_max  = context->input_channels_block_max;
	const size_t output_channels_block_max = context->output_channels_block_max;

	const float* input  = context->input;
	const float* kernel = context->kernel + output_channels_block_start * input_channels;
	float* output       = context->output + output_channels_block_start * image_elements;

	memset(output, 0, sizeof(float) * output_channels_block_size * image_elements);

	size_t input_channels_unprocessed = input_channels;
	if (output_channels_block_size == output_channels_block_max) {
		const nnp_fast_conv_function fast_conv = context->fast_conv;
		while (input_channels_unprocessed >= input_channels_block_max) {
			input_channels_unprocessed -= input_channels_block_max;

			fast_conv(
				input_channels, image_elements,
				input, kernel, output);

			input  += input_channels_block_max * image_elements;
			kernel += input_channels_block_max;
		}
	}

	const nnp_full_conv_function full_conv = context->full_conv;
	while (input_channels_unprocessed != 0) {
		const size_t input_channels_block_size = min(input_channels_unprocessed, input_channels_block_max);
		input_channels_unprocessed -= input_channels_block_size;

		full_conv(
			input_channels_block_size, output_channels_block_size,
			input_channels, image_elements,
			input, kernel, output);

		input  += input_channels_block_max * image_elements;
		kernel += input_channels_block_max;
	}
}

static enum nnp_status compute_fast_convolution_inference(
	const bool fourier_transform,
	const enum nnp_convolution_transform_strategy transform_strategy,
	const size_t transform_element_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size tile_size,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_size,
	const struct nnp_size output_subsampling,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	void* workspace_buffer,
	size_t* workspace_size,
	const nnp_transform_2d_with_offset input_transform_function,
	const nnp_transform_2d_with_offset kernel_transform_function,
	const nnp_transform_2d_with_bias output_transform_function,
	pthreadpool_t threadpool,
	struct nnp_profile* profile)
{
	void* memory_block = NULL;
	size_t memory_size = 0;
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = (fourier_transform ? simd_width * 2 : simd_width);
	const size_t tuple_size = tuple_elements * transform_element_size;
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size output_tile_size = {
		.width = (tile_size.width - kernel_size.width) / output_subsampling.width + 1,
		.height = (tile_size.height - kernel_size.height) / output_subsampling.height + 1
	};
	const struct nnp_size tile_step = {
		.width = tile_size.width - kernel_size.width + 1,
		.height = tile_size.height - kernel_size.height + 1
	};

	const size_t tiles_y_count = divide_round_up(output_size.height, output_tile_size.height);
	const size_t tiles_x_count = divide_round_up(output_size.width, output_tile_size.width);
	const size_t tiles_count = tiles_x_count * tiles_y_count;

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / tuple_size;
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / tuple_size;
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / tuple_size;

	const size_t tiles_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t output_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t input_channels_block_max =
		round_down(cache_elements_l1 / (tiles_subblock_max + output_channels_subblock_max), 2);
	const size_t tiles_block_max =
		round_down(cache_elements_l2 / input_channels_block_max, tiles_subblock_max);
	const size_t output_channels_block_max =
		round_down(cache_elements_l3 / input_channels_block_max, output_channels_subblock_max);

	const size_t transform_tile_size = tile_elements * transform_element_size;
	const size_t input_transform_size = tiles_count * min(input_channels, input_channels_block_max) * transform_tile_size;
	const size_t output_transform_size = tiles_count * output_channels * transform_tile_size;
	switch (transform_strategy) {
		case nnp_convolution_transform_strategy_compute:
		case nnp_convolution_transform_strategy_reuse:
		{
			memory_size = input_transform_size + output_transform_size;
			const size_t kernel_transform_size = output_channels * min(input_channels, input_channels_block_max) * transform_tile_size;
			if (transform_strategy == nnp_convolution_transform_strategy_compute) {
				memory_size += kernel_transform_size;
			}
			if (workspace_buffer == NULL) {
				if (workspace_size == NULL) {
					memory_block = allocate_memory(memory_size);
					if (memory_block == NULL) {
						return nnp_status_out_of_memory;
					}
				} else {
					*workspace_size = memory_size;
					return nnp_status_success;
				}
			} else {
				if (*workspace_size < memory_size) {
					return nnp_status_insufficient_buffer;
				}
				memory_block = workspace_buffer;
			}

			void* input_transform = memory_block;
			void* output_transform = memory_block + input_transform_size;
			void* kernel_transform = memory_block + input_transform_size + output_transform_size;

			for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
				const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

				if (transform_strategy == nnp_convolution_transform_strategy_compute) {
					NNP_KERNEL_TRANSFORM_START(profile)
					struct kernel_transform_context kernel_transform_context = {
						.transform_function = kernel_transform_function,
						.kernel = kernel + input_channels_block_start * kernel_size.height * kernel_size.width,
						.kernel_transform = kernel_transform,
						.tuple_size = tuple_size,
						.input_channels = input_channels,
						.input_channels_block_size = input_channels_block_size,
						.output_channels = output_channels,
						.kernel_size = kernel_size,
					};
					pthreadpool_parallelize_2d_tile_2d(threadpool,
						(pthreadpool_task_2d_tile_2d_t) compute_kernel_transform,
						&kernel_transform_context,
						output_channels,              input_channels_block_size,
						output_channels_subblock_max, 1,
						PTHREADPOOL_FLAG_DISABLE_DENORMALS);
					NNP_KERNEL_TRANSFORM_END(profile)
				} else {
					kernel_transform = (void*) kernel + input_channels_block_start * output_channels * transform_tile_size;
				}

				NNP_INPUT_TRANSFORM_START(profile)
				struct input_transform_context input_transform_context = {
					.input = input,
					.input_transform = input_transform,
					.transform_function = input_transform_function,
					.tuple_size = tuple_size,
					.tiles_count = tiles_count,
					.tiles_x_count = fxdiv_init_size_t(tiles_x_count),
					.input_channels_block_start = input_channels_block_start,
					.input_channels_block_size = input_channels_block_size,
					.input_size = input_size,
					.input_padding_left = input_padding.left,
					.input_padding_top = input_padding.top,
					.input_tile = tile_size,
					.input_tile_step = tile_step,
				};
				pthreadpool_parallelize_2d_tile_2d(threadpool,
					(pthreadpool_task_2d_tile_2d_t) compute_input_transform,
					&input_transform_context,
					input_channels_block_size, tiles_count,
					1,                         tiles_subblock_max,
					PTHREADPOOL_FLAG_DISABLE_DENORMALS);
				NNP_INPUT_TRANSFORM_END(profile)

				NNP_BLOCK_MULTIPLICATION_START(profile)
				for (size_t tuple_index = 0; tuple_index < tuple_count; tuple_index += 1) {
					nnp_full_tuple_gemm_function full_gemm_function;
					nnp_fast_tuple_gemm_function fast_gemm_function;
					if (fourier_transform) {
						if (tuple_index < NNP_COMPLEX_TUPLE_INDEX) {
							fast_gemm_function = nnp_hwinfo.cxgemm.s4cX_conjb_only_mr_x_nr;
							full_gemm_function = nnp_hwinfo.cxgemm.s4cX_conjb_upto_mr_x_nr;
						} else {
							fast_gemm_function = nnp_hwinfo.cxgemm.cX_conjb_only_mr_x_nr;
							full_gemm_function = nnp_hwinfo.cxgemm.cX_conjb_upto_mr_x_nr;
						}
					} else {
						if NNP_LIKELY(transform_element_size == sizeof(float)) {
							fast_gemm_function = nnp_hwinfo.sxgemm.only_mr_x_nr;
							full_gemm_function = nnp_hwinfo.sxgemm.upto_mr_x_nr;
						} else {
							#if NNP_BACKEND_ARM
								fast_gemm_function = nnp_hwinfo.hxgemm.only_mr_x_nr;
								full_gemm_function = nnp_hwinfo.hxgemm.upto_mr_x_nr;
							#endif /* NNP_BACKEND_ARM */
						}
					}
					for (size_t output_channels_block_start = 0; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max) {
						const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
						struct tuple_multiplication_context tuple_multiplication_context = {
							.tuple_elements = tuple_elements,
							.tuple_size = tuple_size,
							.tiles_subblock_max = tiles_subblock_max,
							.input_channels_block_start = input_channels_block_start,
							.input_channels_block_size = input_channels_block_size,
							.output_channels = output_channels,
							.output_channels_subblock_max = output_channels_subblock_max,
							.output_channels_block_start = output_channels_block_start,
							.input_transform = input_transform +
								tuple_index * tiles_count * input_channels_block_size * tuple_size,
							.kernel_transform = kernel_transform +
								tuple_index * output_channels * input_channels_block_size * tuple_size,
							.output_transform = output_transform +
								tuple_index * tiles_count * output_channels * tuple_size,
							.fast_gemm = fast_gemm_function,
							.full_gemm = full_gemm_function,
						};
						pthreadpool_parallelize_2d_tile_2d(threadpool,
							(pthreadpool_task_2d_tile_2d_t) compute_tuple_multiplication,
							&tuple_multiplication_context,
							tiles_count,     output_channels_block_size,
							tiles_block_max, output_channels_subblock_max,
							PTHREADPOOL_FLAG_DISABLE_DENORMALS);
					}
				}
				NNP_BLOCK_MULTIPLICATION_END(profile)
			}
			NNP_OUTPUT_TRANSFORM_START(profile)
			struct output_transform_context output_transform_context = {
				.transform_function = output_transform_function,
				.output = output,
				.output_transform = output_transform,
				.bias = bias,
				.tuple_size = tuple_size,
				.tiles_count = tiles_count,
				.tiles_x_count = fxdiv_init_size_t(tiles_x_count),
				.tiles_block_max = fxdiv_init_size_t(tiles_block_max),
				.output_channels = output_channels,
				.output_size = output_size,
				.output_tile = output_tile_size,
			};
			pthreadpool_parallelize_2d_tile_2d(threadpool,
				(pthreadpool_task_2d_tile_2d_t) compute_output_transform,
				&output_transform_context,
				output_channels,              tiles_count,
				output_channels_subblock_max, tiles_subblock_max,
				PTHREADPOOL_FLAG_DISABLE_DENORMALS);
			NNP_OUTPUT_TRANSFORM_END(profile)
			break;
		}
		case nnp_convolution_transform_strategy_precompute:
		{
			const size_t kernel_transform_size = output_channels * input_channels * transform_tile_size;
			if (workspace_buffer == NULL) {
				*workspace_size = kernel_transform_size;
				return nnp_status_success;
			} else {
				if (*workspace_size < kernel_transform_size) {
					return nnp_status_insufficient_buffer;
				}
				memory_block = workspace_buffer;
			}

			for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
				const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

				NNP_KERNEL_TRANSFORM_START(profile)
				struct kernel_transform_context kernel_transform_context = {
					.transform_function = kernel_transform_function,
					.kernel = kernel + input_channels_block_start * kernel_size.height * kernel_size.width,
					.kernel_transform = (void*) workspace_buffer + input_channels_block_start * output_channels * transform_tile_size,
					.tuple_size = tuple_size,
					.input_channels = input_channels,
					.input_channels_block_size = input_channels_block_size,
					.output_channels = output_channels,
					.kernel_size = kernel_size,
				};
				pthreadpool_parallelize_2d_tile_2d(threadpool,
					(pthreadpool_task_2d_tile_2d_t) compute_kernel_transform,
					&kernel_transform_context,
					output_channels,              input_channels_block_size,
					output_channels_subblock_max, 1,
					PTHREADPOOL_FLAG_DISABLE_DENORMALS);
				NNP_KERNEL_TRANSFORM_END(profile)
			}
			break;
		}
		default:
			return nnp_status_invalid_transform_strategy;
	}

	if (memory_block != workspace_buffer) {
		release_memory(memory_block, memory_size);
	}
	return nnp_status_success;
}

static enum nnp_status compute_gemm_convolution_inference(
	const enum nnp_convolution_transform_strategy transform_strategy,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_size,
	const struct nnp_size output_subsampling,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	void* workspace_buffer,
	size_t* workspace_size,
	enum nnp_activation activation,
	pthreadpool_t threadpool,
	struct nnp_profile* profile)
{
	enum nnp_status status = nnp_status_success;
	void* memory_block = NULL;
	size_t memory_size = 0;
	const size_t simd_width = nnp_hwinfo.simd_width;

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / sizeof(float);
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / sizeof(float);
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / sizeof(float);

	const size_t output_channels_subblock_max = nnp_hwinfo.sgemm.mr;
	const size_t output_image_subblock_max = nnp_hwinfo.sgemm.nr;

	const size_t reduction_size = input_channels * kernel_size.height * kernel_size.width;
	const size_t output_image_size = output_size.height * output_size.width;
	const size_t reduction_block_max =
		round_down(cache_elements_l1 / (output_channels_subblock_max + output_image_subblock_max), 2);
	const size_t output_channels_block_max =
		round_down(cache_elements_l2 / reduction_block_max, output_channels_subblock_max);
	const size_t output_image_block_max =
		round_down(cache_elements_l3 / reduction_block_max, output_image_subblock_max);

	switch (transform_strategy) {
		case nnp_convolution_transform_strategy_compute:
		case nnp_convolution_transform_strategy_reuse:
		{
			const size_t packed_kernel_size = output_channels *
				min(reduction_block_max, reduction_size) * sizeof(float);
			const size_t packed_input_size = min(output_image_block_max, round_up(output_image_size, simd_width)) *
				min(reduction_block_max, reduction_size) * sizeof(float);
			memory_size = packed_kernel_size + packed_input_size;
			if (workspace_buffer == NULL) {
				if (workspace_size == NULL) {
					memory_block = allocate_memory(memory_size);
					if (memory_block == NULL) {
						return nnp_status_out_of_memory;
					}
				} else {
					*workspace_size = memory_size;
					return nnp_status_success;
				}
			} else {
				if (*workspace_size < memory_size) {
					return nnp_status_insufficient_buffer;
				}
				memory_block = workspace_buffer;
			}

			float* packed_input = memory_block;
			float* packed_kernel = memory_block + packed_input_size;

			for (size_t reduction_block_start = 0; reduction_block_start < reduction_size; reduction_block_start += reduction_block_max) {
				const size_t reduction_block_size = min(reduction_size - reduction_block_start, reduction_block_max);

				if (transform_strategy == nnp_convolution_transform_strategy_compute) {
					/* Pack kernel into memory block */
					NNP_KERNEL_TRANSFORM_START(profile)
					struct kernel_packing_context kernel_packing_context = {
						.kernel = kernel + reduction_block_start,
						.packed_kernel = packed_kernel,
						.reduction_size = reduction_size,
						.reduction_block_start = reduction_block_start,
						.reduction_block_size = reduction_block_size,
					};
					pthreadpool_parallelize_2d_tile_2d(threadpool,
						(pthreadpool_task_2d_tile_2d_t) compute_kernel_packing,
						&kernel_packing_context,
						output_channels,              reduction_block_size,
						output_channels_subblock_max, 1,
						PTHREADPOOL_FLAG_DISABLE_DENORMALS);
					NNP_KERNEL_TRANSFORM_END(profile)
				} else {
					packed_kernel = (void*) kernel + output_channels * reduction_block_start * sizeof(float);
				}

				const struct fxdiv_divisor_size_t kernel_elements_divisor = fxdiv_init_size_t(kernel_size.height * kernel_size.width);
				const struct fxdiv_divisor_size_t kernel_width_divisor = fxdiv_init_size_t(kernel_size.width);
				const struct fxdiv_divisor_size_t output_width_divisor = fxdiv_init_size_t(output_size.width);
				for (size_t output_image_block_start = 0; output_image_block_start < output_image_size; output_image_block_start += output_image_block_max) {
					const size_t output_image_block_size = min(output_image_size - output_image_block_start, output_image_block_max);

					/* Pack image into L3 block */
					NNP_INPUT_TRANSFORM_START(profile)
					struct input_packing_context input_packing_context = {
						.input = input,
						.packed_input = packed_input,
						.simd_width = simd_width,
						.reduction_block_start = reduction_block_start,
						.reduction_block_size = reduction_block_size,
						.output_image_block_start = output_image_block_start,
						.input_size = input_size,
						.input_padding_top = input_padding.top,
						.input_padding_left = input_padding.left,
						.kernel_elements = kernel_elements_divisor,
						.kernel_width = kernel_width_divisor,
						.output_width = output_width_divisor,
						.output_subsampling = output_subsampling,
					};
					pthreadpool_parallelize_2d_tile_2d(threadpool,
						(pthreadpool_task_2d_tile_2d_t) compute_input_packing,
						&input_packing_context,
						reduction_block_size, output_image_block_size,
						1,                    output_image_subblock_max,
						PTHREADPOOL_FLAG_DISABLE_DENORMALS);
					NNP_INPUT_TRANSFORM_END(profile)

					NNP_BLOCK_MULTIPLICATION_START(profile)
					struct matrix_multiplication_context matrix_multiplication_context = {
						.packed_kernel = packed_kernel,
						.packed_input = packed_input,
						.output = output,
						.reduction_block_start = reduction_block_start,
						.reduction_block_size = reduction_block_size,
						.output_image_size = output_image_size,
						.output_image_block_start = output_image_block_start,
						.output_image_subblock_max = output_image_subblock_max,
						.output_channels_subblock_max = output_channels_subblock_max,
					};
					pthreadpool_parallelize_2d_tile_2d(threadpool,
						(pthreadpool_task_2d_tile_2d_t) compute_matrix_multiplication,
						&matrix_multiplication_context,
						output_channels,           output_image_block_size,
						output_channels_block_max, output_image_subblock_max,
						PTHREADPOOL_FLAG_DISABLE_DENORMALS);
					NNP_BLOCK_MULTIPLICATION_END(profile)
				}
			}
			/* Add bias */
			NNP_OUTPUT_TRANSFORM_START(profile)
			switch (activation) {
				case nnp_activation_identity:
					for (size_t output_channel = 0; output_channel < output_channels; output_channel += 1) {
						const float bias_value = bias[output_channel];
						for (size_t index = 0; index < output_image_size; index += 1) {
							output[output_channel * output_image_size + index] += bias_value;
						}
					}
					break;
				case nnp_activation_relu:
					for (size_t output_channel = 0; output_channel < output_channels; output_channel += 1) {
						const float bias_value = bias[output_channel];
						for (size_t index = 0; index < output_image_size; index += 1) {
							output[output_channel * output_image_size + index] =
								relu(output[output_channel * output_image_size + index] + bias_value, 0.0f);
						}
					}
					break;
				default:
					NNP_UNREACHABLE;
			}
			NNP_OUTPUT_TRANSFORM_END(profile)
			break;
		}
		case nnp_convolution_transform_strategy_precompute:
		{
			const size_t packed_kernel_size = output_channels * reduction_size * sizeof(float);
			if (workspace_buffer == NULL) {
				*workspace_size = packed_kernel_size;
				return nnp_status_success;
			} else {
				if (*workspace_size < packed_kernel_size) {
					return nnp_status_insufficient_buffer;
				}
				memory_block = workspace_buffer;
			}

			for (size_t reduction_block_start = 0; reduction_block_start < reduction_size; reduction_block_start += reduction_block_max) {
				const size_t reduction_block_size = min(reduction_size - reduction_block_start, reduction_block_max);

				/* Pack kernel into memory block */
				NNP_KERNEL_TRANSFORM_START(profile)
				struct kernel_packing_context kernel_packing_context = {
					.kernel = kernel + reduction_block_start,
					.packed_kernel = (void*) workspace_buffer + output_channels * reduction_block_start * sizeof(float),
					.reduction_size = reduction_size,
					.reduction_block_start = reduction_block_start,
					.reduction_block_size = reduction_block_size,
				};
				pthreadpool_parallelize_2d_tile_2d(threadpool,
					(pthreadpool_task_2d_tile_2d_t) compute_kernel_packing,
					&kernel_packing_context,
					output_channels,              reduction_block_size,
					output_channels_subblock_max, 1,
					PTHREADPOOL_FLAG_DISABLE_DENORMALS);
				NNP_KERNEL_TRANSFORM_END(profile)
			}
			break;
		}
		default:
			return nnp_status_invalid_transform_strategy;
	}

	if (memory_block != workspace_buffer) {
		release_memory(memory_block, memory_size);
	}
	return status;
}

static enum nnp_status compute_direct_convolution_inference(
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size image_size,
	const struct nnp_size kernel_size,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	void* workspace_buffer,
	size_t* workspace_size,
	enum nnp_activation activation,
	pthreadpool_t threadpool,
	struct nnp_profile* profile)
{
	const size_t image_elements = image_size.height * image_size.width;

	if (workspace_buffer == NULL && workspace_size != NULL) {
		*workspace_size = 0;
		return nnp_status_success;
	}

	NNP_BLOCK_MULTIPLICATION_START(profile)
	struct direct_convolution_context direct_convolution_context = {
		.input = input,
		.kernel = kernel,
		.output = output,
		.image_elements = image_elements,
		.input_channels = input_channels,
		.input_channels_block_max = nnp_hwinfo.conv1x1.mr,
		.output_channels_block_max = nnp_hwinfo.conv1x1.nr,
		.fast_conv = nnp_hwinfo.conv1x1.only_mr_x_nr,
		.full_conv = nnp_hwinfo.conv1x1.upto_mr_x_nr,
	};
	pthreadpool_parallelize_1d_tile_1d(threadpool,
		(pthreadpool_task_1d_tile_1d_t) compute_direct_convolution,
		&direct_convolution_context,
		output_channels, nnp_hwinfo.conv1x1.nr,
		PTHREADPOOL_FLAG_DISABLE_DENORMALS);
	NNP_BLOCK_MULTIPLICATION_END(profile)

	/* Add bias */
	NNP_OUTPUT_TRANSFORM_START(profile)
	switch (activation) {
		case nnp_activation_identity:
			for (size_t output_channel = 0; output_channel < output_channels; output_channel += 1) {
				const float bias_value = bias[output_channel];
				for (size_t index = 0; index < image_elements; index += 1) {
					output[output_channel * image_elements + index] += bias_value;
				}
			}
			break;
		case nnp_activation_relu:
			for (size_t output_channel = 0; output_channel < output_channels; output_channel += 1) {
				const float bias_value = bias[output_channel];
				for (size_t index = 0; index < image_elements; index += 1) {
					output[output_channel * image_elements + index] =
						relu(output[output_channel * image_elements + index] + bias_value, 0.0f);
				}
			}
			break;
		default:
			NNP_UNREACHABLE;
	}
	NNP_OUTPUT_TRANSFORM_END(profile)

	return nnp_status_success;
}

static inline enum nnp_convolution_algorithm select_algorithm(
	struct nnp_size kernel_size,
	struct nnp_size output_subsampling,
	struct nnp_size output_size)
{
	if (max(output_subsampling.height, output_subsampling.width) == 1) {
		/* Stride-1 convolution: consider fast convolution algorithm and direct 1x1 */
		if (max(kernel_size.height, kernel_size.width) == 1) {
			return nnp_convolution_algorithm_direct;
		} else if (kernel_size.height == 3 && kernel_size.width == 3) {
			return nnp_convolution_algorithm_wt8x8;
		} else if (min(kernel_size.height, kernel_size.width) >= 2) {
			/* Consider FFT-based fast convolution */
			if (max(kernel_size.height, kernel_size.width) <= 8) {
				/* Decide between FFT 8x8 and FFT 16x16 */
				const size_t tile_count_8x8 =
					divide_round_up(output_size.height, 8 - kernel_size.height + 1) *
					divide_round_up(output_size.width, 8 - kernel_size.width + 1);
				const size_t tile_count_16x16 =
					divide_round_up(output_size.height, 16 - kernel_size.height + 1) *
					divide_round_up(output_size.width, 16 - kernel_size.width + 1);
				if (tile_count_8x8 <= 4 * tile_count_16x16) {
					/* 8x8 tiles are more efficient */
					return nnp_convolution_algorithm_ft8x8;
				} else {
					return nnp_convolution_algorithm_ft16x16;
				}
			} else if (max(kernel_size.height, kernel_size.width) <= 16) {
				return nnp_convolution_algorithm_ft16x16;
			}
		}
	}

	/* Fall-back algorithm */
	return nnp_convolution_algorithm_implicit_gemm;
}

enum nnp_status nnp_convolution_inference(
	enum nnp_convolution_algorithm algorithm,
	enum nnp_convolution_transform_strategy transform_strategy,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	struct nnp_size output_subsampling,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	void* workspace_buffer,
	size_t* workspace_size,
	enum nnp_activation activation,
	const void* activation_parameters,
	pthreadpool_t threadpool,
	struct nnp_profile* profile)
{
	NNP_TOTAL_START(profile)

	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_convolution_arguments(
		1, input_channels, output_channels,
		input_size, input_padding, kernel_size, output_subsampling,
		activation, activation_parameters);
	if (status != nnp_status_success) {
		goto cleanup;
	}

	if (activation_parameters != NULL) {
		status = nnp_status_unsupported_activation_parameters;
		goto cleanup;
	}

	const struct nnp_size output_size = {
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};

	if (algorithm == nnp_convolution_algorithm_auto) {
		algorithm = select_algorithm(kernel_size, output_subsampling, output_size);
	}

	struct nnp_size tile_size;
	size_t transform_element_size;
	bool fourier_transform;
	nnp_transform_2d_with_offset input_transform_function = NULL;
	nnp_transform_2d_with_offset kernel_transform_function = NULL;
	nnp_transform_2d_with_bias output_transform_function = NULL;
	switch (algorithm) {
		case nnp_convolution_algorithm_wt8x8_fp16:
			#if NNP_BACKEND_ARM
				if (kernel_size.height != 3 || kernel_size.width != 3) {
					status = nnp_status_unsupported_algorithm;
					goto cleanup;
				}
				if (max(output_subsampling.height, output_subsampling.width) > 1) {
					status = nnp_status_unsupported_algorithm;
					goto cleanup;
				}
				tile_size = (struct nnp_size) { .height = 8, .width = 8 };
				transform_element_size = sizeof(uint16_t);
				fourier_transform = false;

				input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_fp16_with_offset;
				kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3_fp16;
				switch (activation) {
					case nnp_activation_identity:
						output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias;
						break;
					case nnp_activation_relu:
						output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias_with_relu;
						break;
					default:
						NNP_UNREACHABLE;
				}
				if (input_transform_function != NULL && kernel_transform_function != NULL && output_transform_function != NULL) {
					break;
				}
			#endif
			/*
			 * Fallthrough otherwise. The rationale here is that only some backends have fp16 storage natively implemented
			 * (e.g. ARM NEON + VFP_FP16 currently), while configuration is (currently) fairly platform-independent.
			 * Thus silently falling back to the baseline Winograd implementation is reasonable.
			 */
		case nnp_convolution_algorithm_wt8x8:
			if (kernel_size.height != 3 || kernel_size.width != 3) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			transform_element_size = sizeof(float);
			fourier_transform = false;

			input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;
			switch (activation) {
				case nnp_activation_identity:
					if (output_subsampling.height == 1 && output_subsampling.width == 1) {
						output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
					} else if (output_subsampling.height == 2 && output_subsampling.width == 2) {
						output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3s2_with_bias;
					}
					break;
				case nnp_activation_relu:
					if (output_subsampling.height == 1 && output_subsampling.width == 1) {
						output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu;
					} else if (output_subsampling.height == 2 && output_subsampling.width == 2) {
						output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3s2_with_bias_with_relu;
					}
					break;
				default:
					NNP_UNREACHABLE;
			}
			break;
		case nnp_convolution_algorithm_ft8x8:
			if (max(kernel_size.height, kernel_size.width) > 8) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			if (max(output_subsampling.height, output_subsampling.width) > 1) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			transform_element_size = sizeof(float);
			fourier_transform = true;

			input_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			switch (activation) {
				case nnp_activation_identity:
					output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias;
					break;
				case nnp_activation_relu:
					output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu;
					break;
				default:
					NNP_UNREACHABLE;
			}
			break;
		case nnp_convolution_algorithm_ft16x16:
			if (max(kernel_size.height, kernel_size.width) > 16) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			if (max(output_subsampling.height, output_subsampling.width) > 1) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			tile_size = (struct nnp_size) { .height = 16, .width = 16 };
			transform_element_size = sizeof(float);
			fourier_transform = true;

			input_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			switch (activation) {
				case nnp_activation_identity:
					output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias;
					break;
				case nnp_activation_relu:
					output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu;
					break;
				default:
					NNP_UNREACHABLE;
			}
			break;
		case nnp_convolution_algorithm_implicit_gemm:
			break;
		case nnp_convolution_algorithm_direct:
			if (max(kernel_size.height, kernel_size.width) > 1) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			if (max(output_subsampling.height, output_subsampling.width) > 1) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			break;
		case nnp_convolution_algorithm_auto:
			NNP_UNREACHABLE;
		default:
			status = nnp_status_invalid_algorithm;
			goto cleanup;
	}

	switch (algorithm) {
		case nnp_convolution_algorithm_wt8x8:
		case nnp_convolution_algorithm_wt8x8_fp16:
		case nnp_convolution_algorithm_ft8x8:
		case nnp_convolution_algorithm_ft16x16:
			if (input_transform_function == NULL || kernel_transform_function == NULL || output_transform_function == NULL) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			status = compute_fast_convolution_inference(
				fourier_transform, transform_strategy, transform_element_size,
				input_channels, output_channels,
				tile_size, input_size, input_padding, kernel_size, output_size, output_subsampling,
				input, kernel, bias, output, workspace_buffer, workspace_size,
				input_transform_function, kernel_transform_function, output_transform_function,
				threadpool, profile);
			break;
		case nnp_convolution_algorithm_implicit_gemm:
			status = compute_gemm_convolution_inference(
				transform_strategy,
				input_channels, output_channels,
				input_size, input_padding, kernel_size, output_size, output_subsampling,
				input, kernel, bias, output, workspace_buffer, workspace_size,
				activation,
				threadpool, profile);
			break;
		case nnp_convolution_algorithm_direct:
			if (transform_strategy != nnp_convolution_transform_strategy_compute) {
				status = nnp_status_unsupported_transform_strategy;
				goto cleanup;
			}
			status = compute_direct_convolution_inference(
				input_channels, output_channels, input_size, kernel_size,
				input, kernel, bias, output, workspace_buffer, workspace_size,
				activation,
				threadpool, profile);
			break;
		case nnp_convolution_algorithm_auto:
			NNP_UNREACHABLE;
	}

cleanup:
	NNP_TOTAL_END(profile)
	return status;
}
