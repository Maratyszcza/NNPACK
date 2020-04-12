#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>
#include <nnpack/system.h>

#include <nnpack/hwinfo.h>
#include <nnpack/validation.h>


struct NNP_CACHE_ALIGN input_transform_context {
	size_t tuple_elements;
	size_t input_elements;
	size_t batch_block_size;
	size_t input_channels;
	size_t input_stride;
	uint32_t row_offset;
	uint32_t column_offset;
	uint32_t row_count;
	uint32_t column_count;
	const float* input;
	float* input_transform;
	nnp_transform_2d_with_offset transform_function;
};

static void compute_input_transform(
	const struct input_transform_context context[restrict static 1],
	size_t batch_block_offset,       size_t input_channels_subblock_start,
	size_t batch_block_offset_range, size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t input_elements                  = context->input_elements;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t input_channels                  = context->input_channels;
	const size_t input_stride                    = context->input_stride;
	const uint32_t row_count                     = context->row_count;
	const uint32_t column_count                  = context->column_count;
	const uint32_t row_offset                    = context->row_offset;
	const uint32_t column_offset                 = context->column_offset;
	const float* input                           = context->input;
	float* input_transform                       = context->input_transform;
	const nnp_transform_2d_with_offset transform = context->transform_function;

	for (size_t input_channels_subblock_offset = 0; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset += 1) {
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform(
			input +
				(batch_block_offset * input_channels + input_channel) * input_elements,
			input_transform +
				(input_channels_subblock_start * batch_block_size + batch_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			input_stride,
			batch_block_size * input_channels * tuple_elements * sizeof(float),
			row_count, column_count, row_offset, column_offset);
	}
}

struct NNP_CACHE_ALIGN grad_output_transform_context {
	size_t tuple_elements;
	size_t output_elements;
	size_t batch_block_size;
	size_t output_channels;
	size_t grad_output_stride;
	uint32_t row_count;
	uint32_t column_count;
	const float* grad_output;
	float* grad_output_transform;
	nnp_transform_2d_with_offset transform_function;
};

static void compute_grad_output_transform(
	const struct grad_output_transform_context context[restrict static 1],
	size_t batch_block_offset,       size_t output_channels_subblock_start,
	size_t batch_block_offset_range, size_t output_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t output_elements                 = context->output_elements;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t output_channels                 = context->output_channels;
	const size_t grad_output_stride              = context->grad_output_stride;
	const uint32_t row_count                     = context->row_count;
	const uint32_t column_count                  = context->column_count;
	const float* grad_output                     = context->grad_output;
	float* grad_output_transform                 = context->grad_output_transform;
	const nnp_transform_2d_with_offset transform = context->transform_function;

	for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform(
			grad_output +
				(batch_block_offset * output_channels + output_channel) * output_elements,
			grad_output_transform +
				(output_channels_subblock_start * batch_block_size + batch_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
			grad_output_stride,
			batch_block_size * output_channels * tuple_elements * sizeof(float),
			row_count, column_count, 0, 0);
	}
}

struct NNP_CACHE_ALIGN grad_kernel_transform_context {
	size_t tuple_elements;
	size_t input_channels;
	size_t output_channels;
	size_t output_channels_block_max;
	struct nnp_size kernel_size;
	const float* grad_kernel_transform;
	float* grad_kernel;
	nnp_transform_2d_with_offset transform_function;
};

static void compute_grad_kernel_transform(
	const struct grad_kernel_transform_context context[restrict static 1],
	size_t output_channel,       size_t input_channels_subblock_start,
	size_t output_channel_range, size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t input_channels                  = context->input_channels;
	const size_t output_channels                 = context->output_channels;
	const size_t output_channels_block_max       = context->output_channels_block_max;
	const struct nnp_size kernel_size            = context->kernel_size;
	const float* grad_kernel_transform           = context->grad_kernel_transform;
	float* grad_kernel                           = context->grad_kernel;
	const nnp_transform_2d_with_offset transform = context->transform_function;

	const size_t output_channels_block_start  = round_down(output_channel, output_channels_block_max);
	const size_t output_channels_block_size   = min(output_channels - output_channels_block_start, output_channels_block_max);
	const size_t output_channels_block_offset = output_channel - output_channels_block_start;
	const size_t kernel_elements = kernel_size.height * kernel_size.width;

	for (size_t input_channels_subblock_offset = 0; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset += 1) {
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform(
			grad_kernel_transform +
				(output_channels_block_start * input_channels + input_channels_subblock_start * output_channels_block_size + output_channels_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			grad_kernel +
				(output_channel * input_channels + input_channel) * kernel_elements,
			output_channels * input_channels * tuple_elements * sizeof(float),
			kernel_size.width,
			kernel_size.height, kernel_size.width, 0, 0);
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context {
	size_t tuple_elements;
	size_t batch_size;
	size_t batch_block_size;
	size_t batch_block_update;
	size_t input_channels;
	size_t input_channels_block_start;
	size_t input_channels_block_size;
	size_t input_channels_subblock_max;
	size_t output_channels;
	size_t output_channels_subblock_max;
	const float* grad_output_transform;
	const float* input_transform;
	float* grad_kernel_transform;

	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context context[restrict static 1],
	size_t output_channels_block_start, size_t input_channels_subblock_start,
	size_t output_channels_block_size,  size_t input_channels_subblock_size)
{
	const size_t tuple_elements               = context->tuple_elements;
	const size_t batch_size                   = context->batch_size;
	const size_t batch_block_size             = context->batch_block_size;
	const size_t batch_block_update           = context->batch_block_update;
	const size_t input_channels               = context->input_channels;
	const size_t input_channels_block_start   = context->input_channels_block_start;
	const size_t input_channels_block_size    = context->input_channels_block_size;
	const size_t input_channels_subblock_max  = context->input_channels_subblock_max;
	const size_t output_channels              = context->output_channels;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;

	const float* grad_output_transform = context->grad_output_transform +
		output_channels_block_start * batch_block_size * tuple_elements;
	const float* input_transform       = context->input_transform +
		(input_channels_block_start + input_channels_subblock_start) * batch_block_size * tuple_elements;
	float* grad_kernel_transform       = context->grad_kernel_transform +
		(output_channels_block_start * input_channels + (input_channels_block_start + input_channels_subblock_start) * output_channels_block_size) * tuple_elements;

	if (input_channels_subblock_size == input_channels_subblock_max) {
		const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
		while (output_channels_block_size >= output_channels_subblock_max) {
			output_channels_block_size -= output_channels_subblock_max;

			fast_gemm(
				batch_block_size, batch_block_update,
				input_transform,
				grad_output_transform,
				grad_kernel_transform,
				input_channels_subblock_size * tuple_elements);

			grad_output_transform += output_channels_subblock_max * batch_block_size * tuple_elements;
			grad_kernel_transform += output_channels_subblock_max * input_channels_subblock_size * tuple_elements;
		}
	}

	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
	while (output_channels_block_size != 0) {
		const size_t output_channels_subblock_size = min(output_channels_block_size, output_channels_subblock_max);
		output_channels_block_size -= output_channels_subblock_size;

		full_gemm(
			input_channels_subblock_size, output_channels_subblock_size,
			batch_block_size, batch_block_update,
			input_transform,
			grad_output_transform,
			grad_kernel_transform,
			input_channels_subblock_size * tuple_elements);

		grad_output_transform += output_channels_subblock_max * batch_block_size * tuple_elements;
		grad_kernel_transform += output_channels_subblock_max * input_channels_subblock_size * tuple_elements;
	}
}

static enum nnp_status compute_fast_convolution_kernel_gradient(
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size tile_size,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	struct nnp_size output_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
	void* workspace_buffer,
	size_t* workspace_size,
	nnp_transform_2d_with_offset input_transform_function,
	nnp_transform_2d_with_offset grad_output_transform_function,
	nnp_transform_2d_with_offset grad_kernel_transform_function,
	pthreadpool_t threadpool,
	struct nnp_profile* profile)
{
	void* memory_block = NULL;
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = simd_width * 2;
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size output_tile = {
		.height = tile_size.height - kernel_size.height + 1,
		.width = tile_size.width - kernel_size.width + 1
	};

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t input_channels_subblock_max = nnp_hwinfo.cxgemm.mr;
	const size_t output_channels_subblock_max = nnp_hwinfo.cxgemm.nr;

	const size_t batch_block_max =
		round_down(cache_elements_l1 / (input_channels_subblock_max + output_channels_subblock_max), 2);
	const size_t input_channels_block_max =
		round_down(cache_elements_l3 / batch_block_max, input_channels_subblock_max);
	const size_t output_channels_block_max =
		round_down(cache_elements_l2 / batch_block_max, output_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t input_transform_size = min(batch_size, batch_block_max) * input_channels * tile_elements * sizeof(float);
	const size_t grad_output_transform_size = min(batch_size, batch_block_max) * output_channels * tile_elements * sizeof(float);
	const size_t grad_kernel_transform_size = output_channels * input_channels * tile_elements * sizeof(float);
	const size_t memory_size = input_transform_size + grad_output_transform_size + grad_kernel_transform_size;

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
	float* grad_output_transform = memory_block + input_transform_size;
	float* grad_kernel_transform = memory_block + input_transform_size + grad_output_transform_size;

	for (size_t y = 0; y < output_size.height; y += output_tile.height) {
		const size_t input_y = min(doz(y, input_padding.top), input_size.height);
		for (size_t x = 0; x < output_size.width; x += output_tile.width) {
			const size_t input_x = min(doz(x, input_padding.left), input_size.width);

			for (size_t batch_block_start = 0; batch_block_start < batch_size; batch_block_start += batch_block_max) {
				const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);

				/* Input transform */
				NNP_INPUT_TRANSFORM_START(profile)
				struct input_transform_context input_transform_context = {
					.tuple_elements = tuple_elements,
					.input_elements = input_size.height * input_size.width,
					.batch_block_size = batch_block_size,
					.input_channels = input_channels,
					.input_stride = input_size.width,
					.row_offset = doz(input_padding.top, y),
					.column_offset = doz(input_padding.left, x),
					.row_count = min(input_size.height - input_y,
						tile_size.height - input_transform_context.row_offset),
					.column_count = min(input_size.width - input_x,
						tile_size.width - input_transform_context.column_offset),
					.input = input + (batch_block_start * input_channels * input_size.height + input_y) * input_size.width + input_x,
					.input_transform = input_transform,
					.transform_function = input_transform_function,
				};
				pthreadpool_parallelize_2d_tile_2d(threadpool,
					(pthreadpool_task_2d_tile_2d_t) compute_input_transform,
					&input_transform_context,
					batch_block_size, input_channels,
					1,                input_channels_subblock_max,
					PTHREADPOOL_FLAG_DISABLE_DENORMALS);
				NNP_INPUT_TRANSFORM_END(profile)

				/* Grad output transform */
				NNP_OUTPUT_TRANSFORM_START(profile)
				struct grad_output_transform_context grad_output_transform_context = {
					.tuple_elements = tuple_elements,
					.output_elements = output_size.height * output_size.width,
					.batch_block_size = batch_block_size,
					.output_channels = output_channels,
					.grad_output_stride = output_size.width,
					.row_count = min(output_tile.height, output_size.height - y),
					.column_count = min(output_tile.width, output_size.width - x),
					.grad_output = grad_output + (batch_block_start * output_channels * output_size.height + y) * output_size.width + x,
					.grad_output_transform = grad_output_transform,
					.transform_function = grad_output_transform_function,
				};
				pthreadpool_parallelize_2d_tile_2d(threadpool,
					(pthreadpool_task_2d_tile_2d_t) compute_grad_output_transform,
					&grad_output_transform_context,
					batch_block_size, output_channels,
					1,                output_channels_subblock_max,
					PTHREADPOOL_FLAG_DISABLE_DENORMALS);
				NNP_OUTPUT_TRANSFORM_END(profile)

				NNP_BLOCK_MULTIPLICATION_START(profile)
				for (size_t tuple_index = 0; tuple_index < tuple_count; tuple_index += 1) {
					for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
						const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

						struct matrix_multiplication_context matrix_multiplication_context = {
							.tuple_elements = tuple_elements,
							.batch_size = batch_size,
							.batch_block_size = batch_block_size,
							.batch_block_update = batch_block_start | x | y,
							.input_channels = input_channels,
							.input_channels_block_start = input_channels_block_start,
							.input_channels_block_size = input_channels_block_size,
							.input_channels_subblock_max = input_channels_subblock_max,
							.output_channels = output_channels,
							.output_channels_subblock_max = output_channels_subblock_max,
							.grad_output_transform = grad_output_transform +
								tuple_index * tuple_elements * batch_block_size * output_channels,
							.input_transform = input_transform +
								tuple_index * tuple_elements * batch_block_size * input_channels,
							.grad_kernel_transform = grad_kernel_transform +
								tuple_index * tuple_elements * output_channels * input_channels,
						};
						if (tuple_index < NNP_COMPLEX_TUPLE_INDEX) {
							matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_transc_only_mr_x_nr;
							matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_transc_upto_mr_x_nr;
						} else {
							matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.cX_conjb_transc_only_mr_x_nr;
							matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.cX_conjb_transc_upto_mr_x_nr;
						}
						pthreadpool_parallelize_2d_tile_2d(threadpool,
							(pthreadpool_task_2d_tile_2d_t) compute_matrix_multiplication,
							&matrix_multiplication_context,
							output_channels,           input_channels_block_size,
							output_channels_block_max, input_channels_subblock_max,
							PTHREADPOOL_FLAG_DISABLE_DENORMALS);
					}
				}
				NNP_BLOCK_MULTIPLICATION_END(profile)


			}
		}
	}
	/* Grad kernel transform */
	NNP_KERNEL_TRANSFORM_START(profile)
	struct grad_kernel_transform_context grad_kernel_transform_context = {
		.tuple_elements = tuple_elements,
		.input_channels = input_channels,
		.output_channels = output_channels,
		.output_channels_block_max = output_channels_block_max,
		.kernel_size = kernel_size,
		.grad_kernel = grad_kernel,
		.grad_kernel_transform = grad_kernel_transform,
		.transform_function = grad_kernel_transform_function,
	};
	pthreadpool_parallelize_2d_tile_2d(threadpool,
		(pthreadpool_task_2d_tile_2d_t) compute_grad_kernel_transform,
		&grad_kernel_transform_context,
		output_channels, input_channels,
		1,               input_channels_subblock_max,
		PTHREADPOOL_FLAG_DISABLE_DENORMALS);
	NNP_KERNEL_TRANSFORM_END(profile)

	if (memory_block != workspace_buffer) {
		release_memory(memory_block, memory_size);
	}
	return nnp_status_success;
}

enum nnp_status nnp_convolution_kernel_gradient(
	enum nnp_convolution_algorithm algorithm,
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
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

	if (activation != nnp_activation_identity) {
		status = nnp_status_unsupported_activation;
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
				algorithm = nnp_convolution_algorithm_ft8x8;
			} else {
				algorithm = nnp_convolution_algorithm_ft16x16;
			}
		}
	}

	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	struct nnp_size tile_size;
	nnp_transform_2d_with_offset input_transform_function;
	nnp_transform_2d_with_offset grad_output_transform_function;
	nnp_transform_2d_with_offset grad_kernel_transform_function;
	switch (algorithm) {
		case nnp_convolution_algorithm_ft8x8:
			input_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			grad_output_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			grad_kernel_transform_function = nnp_hwinfo.transforms.ifft8x8_with_offset;
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			break;
		case nnp_convolution_algorithm_ft16x16:
			input_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			grad_output_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			grad_kernel_transform_function = nnp_hwinfo.transforms.ifft16x16_with_offset;
			tile_size = (struct nnp_size) { .height = 16, .width = 16 };
			break;
		case nnp_convolution_algorithm_wt8x8:
		case nnp_convolution_algorithm_wt8x8_fp16:
			/*
			 * Winograd transform is not supported for this operation:
			 * it needs F(5x5, 4x4) transform and presently we implement only F(3x3, 6x6)
			 */
			status = nnp_status_unsupported_algorithm;
			goto cleanup;
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
		case nnp_convolution_algorithm_ft8x8:
		case nnp_convolution_algorithm_ft16x16:
			if (kernel_size.height > tile_size.height || kernel_size.width > tile_size.width) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			status = compute_fast_convolution_kernel_gradient(
				batch_size, input_channels, output_channels,
				tile_size, input_size, input_padding, kernel_size, output_size,
				input, grad_output, grad_kernel, workspace_buffer, workspace_size,
				input_transform_function, grad_output_transform_function, grad_kernel_transform_function,
				threadpool, profile);
			break;
		case nnp_convolution_algorithm_wt8x8:
		case nnp_convolution_algorithm_wt8x8_fp16:
		case nnp_convolution_algorithm_implicit_gemm:
		case nnp_convolution_algorithm_direct:
		case nnp_convolution_algorithm_auto:
			NNP_UNREACHABLE;
	}

cleanup:
	NNP_TOTAL_END(profile)
	return status;
}
