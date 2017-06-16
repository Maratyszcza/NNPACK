#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>
#include <nnpack/system.h>

#include <nnpack/hwinfo.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN kernel_transform_context {
	nnp_transform_2d_with_offset transform_function;
	const float* kernel;
	float* kernel_transform;

	size_t tuple_elements;
	size_t output_channels;
	size_t input_channels;
	size_t input_channels_block_max;
	struct nnp_size kernel_size;
};

static void compute_kernel_transform(
	const struct kernel_transform_context context[restrict static 1],
	size_t input_channel,       size_t output_channels_subblock_start,
	size_t input_channel_range, size_t output_channels_subblock_size)
{
	const size_t tuple_elements           = context->tuple_elements;
	const size_t output_channels          = context->output_channels;
	const size_t input_channels           = context->input_channels;
	const size_t input_channels_block_max = context->input_channels_block_max;
	const struct nnp_size kernel_size     = context->kernel_size;

	const float (*kernel)[input_channels][kernel_size.width * kernel_size.height] =
		(const float(*)[input_channels][kernel_size.width * kernel_size.height]) context->kernel;
	float* kernel_transform                         = context->kernel_transform;
	nnp_transform_2d_with_offset transform_function = context->transform_function;

	const size_t input_channels_block_start = round_down(input_channel, input_channels_block_max);
	const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);
	const size_t input_channels_block_offset = input_channel - input_channels_block_start;

	for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform_function(
			kernel[output_channel][input_channel],
			kernel_transform +
				(input_channels_block_start * output_channels + output_channels_subblock_start * input_channels_block_size + input_channels_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
			kernel_size.width,
			output_channels * input_channels * tuple_elements * sizeof(float),
			kernel_size.height, kernel_size.width, 0, 0);
	}
}

struct NNP_CACHE_ALIGN input_transform_context {
	nnp_transform_2d_with_offset transform_function;
	const float* input;
	float* input_transform;

	size_t tuple_elements;
	size_t batch_size;
	size_t input_channels;
	size_t input_channels_block_max;
	struct nnp_size input_size;
	size_t row_offset;
	size_t row_count;
	size_t column_offset;
	size_t column_count;
};

static void compute_input_transform(
	const struct input_transform_context context[restrict static 1],
	size_t input_channel,       size_t batch_subblock_start,
	size_t input_channel_range, size_t batch_subblock_size)
{
	const size_t tuple_elements           = context->tuple_elements;
	const size_t batch_size               = context->batch_size;
	const size_t input_channels           = context->input_channels;
	const size_t input_channels_block_max = context->input_channels_block_max;
	const struct nnp_size input_size      = context->input_size;
	const size_t row_offset               = context->row_offset;
	const size_t row_count                = context->row_count;
	const size_t column_offset            = context->column_offset;
	const size_t column_count             = context->column_count;

	const float (*input)[input_channels][input_size.width * input_size.height] =
		(const float(*)[input_channels][input_size.width * input_size.height]) context->input;
	float* input_transform                          = context->input_transform;
	nnp_transform_2d_with_offset transform_function = context->transform_function;

	const size_t input_channels_block_start = round_down(input_channel, input_channels_block_max);
	const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);
	const size_t input_channels_block_offset = input_channel - input_channels_block_start;

	for (size_t batch_subblock_offset = 0; batch_subblock_offset < batch_subblock_size; batch_subblock_offset += 1) {
		const size_t sample = batch_subblock_start + batch_subblock_offset;
		transform_function(
			input[sample][input_channel],
			input_transform +
				(input_channels_block_start * batch_size + batch_subblock_start * input_channels_block_size + input_channels_block_offset * batch_subblock_size + batch_subblock_offset) * tuple_elements,
			input_size.width,
			batch_size * input_channels * tuple_elements * sizeof(float),
			row_count, column_count, row_offset, column_offset);
	}
}

struct NNP_CACHE_ALIGN output_transform_context {
	nnp_transform_2d_with_bias transform_function;
	float* output;
	const float* output_transform;
	const float* bias;

	size_t tuple_elements;
	size_t output_channels;
	size_t batch_size;
	size_t batch_block_max;
	struct nnp_size output_size;
	size_t row_offset;
	size_t row_count;
	size_t column_offset;
	size_t column_count;
};

static void compute_output_transform(
	const struct output_transform_context context[restrict static 1],
	size_t sample,       size_t output_channels_subblock_start,
	size_t sample_range, size_t output_channels_subblock_size)
{
	const size_t tuple_elements       = context->tuple_elements;
	const size_t batch_size           = context->batch_size;
	const size_t output_channels      = context->output_channels;
	const size_t batch_block_max      = context->batch_block_max;
	const struct nnp_size output_size = context->output_size;
	const size_t row_offset           = context->row_offset;
	const size_t row_count            = context->row_count;
	const size_t column_offset        = context->column_offset;
	const size_t column_count         = context->column_count;

	float (*output)[output_channels][output_size.width * output_size.height] =
		(float(*)[output_channels][output_size.width * output_size.height]) context->output;
	const float* output_transform                 = context->output_transform;
	const float* bias                             = context->bias;
	nnp_transform_2d_with_bias transform_function = context->transform_function;

	const size_t batch_block_start = round_down(sample, batch_block_max);
	const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);
	const size_t batch_block_offset = sample - batch_block_start;

	for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform_function(
			output_transform +
				(batch_block_start * output_channels + output_channels_subblock_start * batch_block_size + batch_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
			output[sample][output_channel],
			&bias[output_channel],
			batch_size * output_channels * tuple_elements * sizeof(float),
			output_size.width,
			row_count, column_count);
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context {
	size_t tuple_elements;
	size_t batch_block_size;
	size_t input_channels_block_start;
	size_t input_channels_block_size;
	size_t batch_subblock_max;
	size_t output_channels_subblock_max;

	const float* input_transform;
	const float* kernel_transform;
	float* output_transform;

	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context context[restrict static 1],
	size_t output_channels_block_start, size_t batch_subblock_start,
	size_t output_channels_block_size,  size_t batch_subblock_size)
{
	const size_t tuple_elements               = context->tuple_elements;
	const size_t batch_block_size             = context->batch_block_size;
	const size_t input_channels_block_start   = context->input_channels_block_start;
	const size_t input_channels_block_size    = context->input_channels_block_size;
	const size_t batch_subblock_max           = context->batch_subblock_max;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;

	const float* input_transform  = context->input_transform +
		(batch_subblock_start * input_channels_block_size * tuple_elements);
	const float* kernel_transform = context->kernel_transform +
		(output_channels_block_start * input_channels_block_size * tuple_elements);
	float* output_transform       = context->output_transform +
		(output_channels_block_start * batch_block_size * tuple_elements);

	if (batch_subblock_size == batch_subblock_max) {
		const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
		while (output_channels_block_size >= output_channels_subblock_max) {
			output_channels_block_size -= output_channels_subblock_max;

			fast_gemm(
				input_channels_block_size, input_channels_block_start,
				input_transform,
				kernel_transform,
				output_transform + (batch_subblock_start * output_channels_subblock_max * tuple_elements),
				output_channels_subblock_max * tuple_elements);

			kernel_transform += input_channels_block_size * output_channels_subblock_max * tuple_elements;
			output_transform += batch_block_size          * output_channels_subblock_max * tuple_elements;
		}
	}

	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
	while (output_channels_block_size != 0) {
		const size_t output_channels_subblock_size = min(output_channels_block_size, output_channels_subblock_max);
		output_channels_block_size -= output_channels_subblock_size;

		full_gemm(
			batch_subblock_size, output_channels_subblock_size,
			input_channels_block_size, input_channels_block_start,
			input_transform,
			kernel_transform,
			output_transform + (batch_subblock_start * output_channels_subblock_size * tuple_elements),
			output_channels_subblock_size * tuple_elements);

		kernel_transform += input_channels_block_size * output_channels_subblock_max * tuple_elements;
		output_transform += batch_block_size          * output_channels_subblock_max * tuple_elements;
	}
}

static enum nnp_status compute_fast_convolution_output(
	bool fourier_transform,
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size tile_size,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	struct nnp_size output_size,
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
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = (fourier_transform ? simd_width * 2 : simd_width);
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size output_tile_size = {
		.height = tile_size.height - kernel_size.height + 1,
		.width = tile_size.width - kernel_size.width + 1
	};

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t batch_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t output_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t input_channels_block_max =
		round_down(cache_elements_l1 / (batch_subblock_max + output_channels_subblock_max), 2);
	const size_t batch_block_max =
		round_down(cache_elements_l3 / input_channels_block_max, batch_subblock_max);
	const size_t output_channels_block_max =
		round_down(cache_elements_l2 / input_channels_block_max, output_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t kernel_transform_size = output_channels * input_channels * tile_elements * sizeof(float);
	const size_t input_transform_size = batch_size * input_channels * tile_elements * sizeof(float);
	const size_t output_transform_size = batch_size * output_channels * tile_elements * sizeof(float);
	const size_t memory_size = kernel_transform_size + input_transform_size + output_transform_size;

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

	float* input_transform = memory_block;
	float* output_transform = memory_block + input_transform_size;
	float* kernel_transform = memory_block + input_transform_size + output_transform_size;

	NNP_KERNEL_TRANSFORM_START(profile)
	struct kernel_transform_context kernel_transform_context = {
		.transform_function = kernel_transform_function,
		.kernel = kernel,
		.kernel_transform = kernel_transform,
		.tuple_elements = tuple_elements,
		.output_channels = output_channels,
		.input_channels = input_channels,
		.input_channels_block_max = input_channels_block_max,
		.kernel_size = kernel_size,
	};
	pthreadpool_compute_2d_tiled(threadpool,
		(pthreadpool_function_2d_tiled_t) compute_kernel_transform,
		&kernel_transform_context,
		input_channels, output_channels,
		1,              output_channels_subblock_max);
	NNP_KERNEL_TRANSFORM_END(profile)

	for (size_t y = 0; y < output_size.height; y += output_tile_size.height) {
		const size_t input_y = min(doz(y, input_padding.top), input_size.height);
		for (size_t x = 0; x < output_size.width; x += output_tile_size.width) {
			const size_t input_x = min(doz(x, input_padding.left), input_size.width);

			NNP_INPUT_TRANSFORM_START(profile)
			struct input_transform_context input_transform_context = {
				.transform_function = input_transform_function,
				.input = input + input_y * input_size.width + input_x,
				.input_transform = input_transform,
				.tuple_elements = tuple_elements,
				.batch_size = batch_size,
				.input_channels = input_channels,
				.input_channels_block_max = input_channels_block_max,
				.input_size = input_size,
				.row_offset = doz(input_padding.top, y),
				.row_count = min(input_size.height - input_y,
					tile_size.height - input_transform_context.row_offset),
				.column_offset = doz(input_padding.left, x),
				.column_count = min(input_size.width - input_x,
					tile_size.width - input_transform_context.column_offset),
			};
			pthreadpool_compute_2d_tiled(threadpool,
				(pthreadpool_function_2d_tiled_t) compute_input_transform,
				&input_transform_context,
				input_channels, batch_size,
				1, batch_subblock_max);
			NNP_INPUT_TRANSFORM_END(profile)

			NNP_BLOCK_MULTIPLICATION_START(profile)
			for (size_t tuple_index = 0; tuple_index < tuple_count; tuple_index += 1) {
				for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
					const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);
					for (size_t batch_block_start = 0; batch_block_start < batch_size; batch_block_start += batch_block_max) {
						const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);
						struct matrix_multiplication_context matrix_multiplication_context = {
							.tuple_elements = tuple_elements,
							.batch_block_size = batch_block_size,
							.input_channels_block_start = input_channels_block_start,
							.input_channels_block_size = input_channels_block_size,
							.batch_subblock_max = batch_subblock_max,
							.output_channels_subblock_max = output_channels_subblock_max,
							.input_transform = input_transform +
								tuple_index * tuple_elements * batch_size * input_channels +
								input_channels_block_start * batch_size * tuple_elements +
								batch_block_start * input_channels_block_size * tuple_elements,
							.kernel_transform = kernel_transform +
								tuple_index * tuple_elements * output_channels * input_channels +
								input_channels_block_start * output_channels * tuple_elements,
							.output_transform = output_transform + tuple_index * tuple_elements * batch_size * output_channels +
								batch_block_start * output_channels * tuple_elements,
						};
						if (fourier_transform) {
							if (tuple_index < NNP_COMPLEX_TUPLE_INDEX) {
								matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_only_mr_x_nr;
								matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_upto_mr_x_nr;
							} else {
								matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.cX_conjb_only_mr_x_nr;
								matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.cX_conjb_upto_mr_x_nr;
							}
						} else {
							matrix_multiplication_context.fast_gemm = nnp_hwinfo.sxgemm.only_mr_x_nr;
							matrix_multiplication_context.full_gemm = nnp_hwinfo.sxgemm.upto_mr_x_nr;
						}
						pthreadpool_compute_2d_tiled(threadpool,
							(pthreadpool_function_2d_tiled_t) compute_matrix_multiplication,
							&matrix_multiplication_context,
							output_channels,           batch_block_size,
							output_channels_block_max, batch_subblock_max);
					}
				}
			}
			NNP_BLOCK_MULTIPLICATION_END(profile)

			NNP_OUTPUT_TRANSFORM_START(profile)
			struct output_transform_context output_transform_context = {
				.transform_function = output_transform_function,
				.output = output + y * output_size.width + x,
				.output_transform = output_transform,
				.bias = bias,
				.tuple_elements = tuple_elements,
				.output_channels = output_channels,
				.batch_size = batch_size,
				.batch_block_max = batch_block_max,
				.output_size = output_size,
				.row_count = min(output_tile_size.height, output_size.height - y),
				.column_count = min(output_tile_size.width, output_size.width - x),
			};
			pthreadpool_compute_2d_tiled(threadpool,
				(pthreadpool_function_2d_tiled_t) compute_output_transform,
				&output_transform_context,
				batch_size, output_channels,
				1,          output_channels_subblock_max);
			NNP_OUTPUT_TRANSFORM_END(profile)
		}
	}

	if (memory_block != workspace_buffer) {
		release_memory(memory_block, memory_size);
	}
	return nnp_status_success;
}

enum nnp_status nnp_convolution_output(
	enum nnp_convolution_algorithm algorithm,
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
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
		batch_size, input_channels, output_channels,
		input_size, input_padding, kernel_size, (struct nnp_size) { 1, 1 },
		activation, activation_parameters);
	if (status != nnp_status_success) {
		goto cleanup;
	}

	if (activation_parameters != NULL) {
		status = nnp_status_unsupported_activation_parameters;
		goto cleanup;
	}

	const struct nnp_size output_size = {
		.width = input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1,
		.height = input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1
	};

	/* If requested, choose optimal convolution algorithm */
	if (algorithm == nnp_convolution_algorithm_auto) {
		if (max(kernel_size.width, kernel_size.height) > 8) {
			algorithm = nnp_convolution_algorithm_ft16x16;
		} else {
			const size_t tile_count_8x8 =
				divide_round_up(output_size.height, 8 - kernel_size.height + 1) *
				divide_round_up(output_size.width, 8 - kernel_size.width + 1);
			const size_t tile_count_16x16 =
				divide_round_up(output_size.height, 16 - kernel_size.height + 1) *
				divide_round_up(output_size.width, 16 - kernel_size.width + 1);
			if (tile_count_8x8 <= 4 * tile_count_16x16) {
				/* 8x8 tiles are more efficient */
				if ((kernel_size.height == 3) && (kernel_size.width == 3)) {
					algorithm = nnp_convolution_algorithm_wt8x8;
				} else {
					algorithm = nnp_convolution_algorithm_ft8x8;
				}
			} else {
				algorithm = nnp_convolution_algorithm_ft16x16;
			}
		}
	}

	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	struct nnp_size tile_size;
	bool fourier_transform;
	nnp_transform_2d_with_offset input_transform_function;
	nnp_transform_2d_with_offset kernel_transform_function;
	nnp_transform_2d_with_bias output_transform_function;
	switch (algorithm) {
		case nnp_convolution_algorithm_ft8x8:
			input_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			switch (activation) {
				case nnp_activation_relu:
					output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu;
					break;
				case nnp_activation_identity:
					output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias;
					break;
				default:
					NNP_UNREACHABLE;
			}
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			fourier_transform = true;
			break;
		case nnp_convolution_algorithm_ft16x16:
			input_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			switch (activation) {
				case nnp_activation_relu:
					output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu;
					break;
				case nnp_activation_identity:
					output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias;
					break;
				default:
					NNP_UNREACHABLE;
			}
			tile_size = (struct nnp_size) { .height = 16, .width = 16 };
			fourier_transform = true;
			break;
		case nnp_convolution_algorithm_wt8x8:
		case nnp_convolution_algorithm_wt8x8_fp16:
			if ((kernel_size.height != 3) || (kernel_size.width != 3)) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;
			output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
			switch (activation) {
				case nnp_activation_relu:
					output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu;
					break;
				case nnp_activation_identity:
					output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
					break;
				default:
					NNP_UNREACHABLE;
			}
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			fourier_transform = false;
			break;
		case nnp_convolution_algorithm_implicit_gemm:
		case nnp_convolution_algorithm_direct:
			status = nnp_status_unsupported_algorithm;
			goto cleanup;
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
			if (kernel_size.height > tile_size.height || kernel_size.width > tile_size.width) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			status = compute_fast_convolution_output(
				fourier_transform,
				batch_size, input_channels, output_channels,
				tile_size, input_size, input_padding, kernel_size, output_size,
				input, kernel, bias, output, workspace_buffer, workspace_size,
				input_transform_function, kernel_transform_function, output_transform_function,
				threadpool, profile);
			break;
		case nnp_convolution_algorithm_implicit_gemm:
		case nnp_convolution_algorithm_direct:
		case nnp_convolution_algorithm_auto:
			NNP_UNREACHABLE;
	}

cleanup:
	NNP_TOTAL_END(profile)
	return status;
}
