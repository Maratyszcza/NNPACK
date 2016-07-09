#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include <fxdiv.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>
#include <nnpack/system.h>
#include <nnpack/hwinfo.h>
#include <nnpack/simd.h>

#include <nnpack/validation.h>
#include <nnpack/transform.h>

enum nnp_status nnp_convolution_inference(
	enum nnp_convolution_algorithm algorithm,
	enum nnp_convolution_transform_strategy transform_strategy,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	const float input_pointer[],
	const float kernel_pointer[],
	const float bias[],
	float output_pointer[],
	pthreadpool_t threadpool,
	struct nnp_profile* profile)
{
	void* memory_block = NULL;
	NNP_TOTAL_START(profile)

	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_convolution_arguments(
		1, input_channels, output_channels,
		input_size, input_padding, kernel_size);
	if (status != nnp_status_success) {
		goto cleanup;
	}

	const struct nnp_size output_size = {
		.width = input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1,
		.height = input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1
	};

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

	const size_t simd_width = nnp_hwinfo.simd_width;
	struct nnp_size tile_size;
	bool fourier_transform;
	nnp_transform_2d input_transform_function = NULL;
	nnp_transform_2d kernel_transform_function = NULL;
	nnp_transform_2d_with_bias output_transform_function = NULL;
	nnp_blockmac block_mac_function = NULL;
	switch (algorithm) {
		case nnp_convolution_algorithm_wt8x8:
			if ((kernel_size.height != 3) || (kernel_size.width != 3)) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;
			output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
			fourier_transform = false;
			break;
		case nnp_convolution_algorithm_ft8x8:
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
			input_transform_function = nnp_hwinfo.transforms.fft8x8_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft8x8_and_stream;
			output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias;
			fourier_transform = true;
			break;
		case nnp_convolution_algorithm_ft16x16:
			tile_size = (struct nnp_size) { .height = 16, .width = 16 };
			input_transform_function = nnp_hwinfo.transforms.fft16x16_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft16x16_and_stream;
			output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias;
			fourier_transform = true;
			break;
		case nnp_convolution_algorithm_auto:
			NNP_UNREACHABLE;
		default:
			status = nnp_status_invalid_algorithm;
			goto cleanup;
	}

	const size_t tuple_elements = (fourier_transform ? simd_width * 2 : simd_width);
	const size_t tuple_size = tuple_elements * sizeof(float);
	const size_t tile_elements = tile_size.height * tile_size.width;

	const struct nnp_size input_tile = {
		.width = tile_size.width,
		.height = tile_size.height
	};

	const struct nnp_size output_tile = {
		.width = input_tile.width - kernel_size.width + 1,
		.height = input_tile.height - kernel_size.height + 1
	};

	const size_t tiles_y_count = divide_round_up(output_size.height, output_tile.height);
	const struct fxdiv_divisor_size_t tiles_x_count = fxdiv_init_size_t(
		divide_round_up(output_size.width, output_tile.width));
	const size_t tiles_count = tiles_x_count.value * tiles_y_count;

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t tiles_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t output_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t input_channels_block_max =
		round_down(cache_elements_l1 / (tiles_subblock_max + output_channels_subblock_max), 2);
	const size_t tiles_block_max =
		round_down(cache_elements_l2 / input_channels_block_max, tiles_subblock_max);
	const size_t output_channels_block_max =
		round_down(cache_elements_l3 / input_channels_block_max, output_channels_subblock_max);

	const size_t transform_tile_size = tile_elements * sizeof(float);
	const size_t input_transform_size = tiles_count * min(input_channels, input_channels_block_max) * transform_tile_size;
	const size_t kernel_transform_size = output_channels * min(input_channels, input_channels_block_max) * transform_tile_size;
	const size_t output_transform_size = tiles_count * output_channels * transform_tile_size;
	const size_t memory_size = input_transform_size + kernel_transform_size + output_transform_size;

	memory_block = allocate_memory(memory_size);
	if (memory_block == NULL) {
		status = nnp_status_out_of_memory;
		goto cleanup;
	}

	float* input_transform = memory_block;
	float* kernel_transform = memory_block + input_transform_size;
	float* output_transform = memory_block + input_transform_size + kernel_transform_size;

	{
		const float (*input)[input_size.width][input_size.height] =
			(const float(*)[input_size.width][input_size.height]) input_pointer;
		const float (*kernel)[input_channels][kernel_size.width * kernel_size.height] =
			(const float(*)[input_channels][kernel_size.width * kernel_size.height]) kernel_pointer;
		float (*output)[output_size.width][output_size.height] =
			(float(*)[output_size.width][output_size.height]) output_pointer;

		switch (transform_strategy) {
			case nnp_convolution_transform_strategy_tuple_based:
				for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
					const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

					NNP_KERNEL_TRANSFORM_START(profile)
					for (size_t output_channels_subblock_start = 0; output_channels_subblock_start < output_channels; output_channels_subblock_start += output_channels_subblock_max) {
						const size_t output_channels_subblock_size = min(output_channels - output_channels_subblock_start, output_channels_subblock_max);
						for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
							const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
							for (size_t input_channels_block_offset = 0; input_channels_block_offset < input_channels_block_size; input_channels_block_offset += 1) {
								const size_t input_channel = input_channels_block_start + input_channels_block_offset;
								kernel_transform_function(
									kernel[output_channel][input_channel],
									kernel_transform +
										(output_channels_subblock_start * input_channels_block_size + input_channels_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
									kernel_size.width,
									input_channels_block_size * output_channels * tuple_size,
									kernel_size.height, kernel_size.width, 0, 0);
							}
						}
					}
					NNP_KERNEL_TRANSFORM_END(profile)

					NNP_INPUT_TRANSFORM_START(profile)
					for (size_t tiles_block_start = 0; tiles_block_start < tiles_count; tiles_block_start += tiles_block_max) {
						const size_t tiles_block_size = min(tiles_count - tiles_block_start, tiles_block_max);
						for (size_t tiles_subblock_start = 0; tiles_subblock_start < tiles_block_size; tiles_subblock_start += tiles_subblock_max) {
							const size_t tiles_subblock_size = min(tiles_block_size - tiles_subblock_start, tiles_subblock_max);
							for (size_t tiles_subblock_offset = 0; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset += 1) {
								const size_t tile = tiles_block_start + tiles_subblock_start + tiles_subblock_offset;
								const struct fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
								const size_t tile_x = tile_xy.remainder;
								const size_t tile_y = tile_xy.quotient;

								const size_t output_x = tile_x * output_tile.width;
								const size_t output_y = tile_y * output_tile.height;

								const size_t input_x = min(doz(output_x, input_padding.left), input_size.width);
								const size_t input_y = min(doz(output_y, input_padding.top), input_size.height);

								const size_t row_offset = doz(input_padding.top, output_y);
								const size_t row_count = min(input_size.height - input_y, input_tile.height - row_offset);
								const size_t column_offset = doz(input_padding.left, output_x);
								const size_t column_count = min(input_size.width - input_x, input_tile.width - column_offset);

								for (size_t input_channels_block_offset = 0; input_channels_block_offset < input_channels_block_size; input_channels_block_offset++) {
									const size_t input_channel = input_channels_block_start + input_channels_block_offset;
									input_transform_function(
										&input[input_channel][input_y][input_x],
										input_transform + ((tiles_block_start + tiles_subblock_start) * input_channels_block_size + input_channels_block_offset * tiles_subblock_size + tiles_subblock_offset) * tuple_elements,
										input_size.width,
										input_channels_block_size * tiles_count * tuple_size,
										row_count, column_count, row_offset, column_offset);
								}
							}
						}
					}
					NNP_INPUT_TRANSFORM_END(profile)

					NNP_BLOCK_MULTIPLICATION_START(profile)
					for (size_t tuple_index = 0; tuple_index < tile_elements / tuple_elements; tuple_index += 1) {
						nnp_full_tuple_gemm_function full_gemm_function;
						if (fourier_transform) {
							if (tuple_index == 0) {
								full_gemm_function = nnp_hwinfo.cxgemm.s4cX_conjb_upto_mr_x_nr;
							} else {
								full_gemm_function = nnp_hwinfo.cxgemm.cX_conjb_upto_mr_x_nr;
							}
						} else {
							full_gemm_function = nnp_hwinfo.sxgemm.upto_mr_x_nr;
						}
						for (size_t tiles_block_start = 0; tiles_block_start < tiles_count; tiles_block_start += tiles_block_max) {
							const size_t tiles_block_size = min(tiles_count - tiles_block_start, tiles_block_max);
							for (size_t output_channels_block_start = 0; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max) {
								const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
								for (size_t output_channels_subblock_start = 0; output_channels_subblock_start < output_channels_block_size; output_channels_subblock_start += output_channels_subblock_max) {
									const size_t output_channels_subblock_size = min(output_channels_block_size - output_channels_subblock_start, output_channels_subblock_max);
									for (size_t tiles_subblock_start = 0; tiles_subblock_start < tiles_block_size; tiles_subblock_start += tiles_subblock_max) {
										const size_t tiles_subblock_size = min(tiles_block_size - tiles_subblock_start, tiles_subblock_max);
										full_gemm_function(
											tiles_subblock_size, output_channels_subblock_size,
											input_channels_block_size, input_channels_block_start,
											input_transform +
												tuple_index * tiles_count * input_channels_block_size * tuple_elements +
												((tiles_block_start + tiles_subblock_start) * input_channels_block_size) * tuple_elements,
											kernel_transform +
												tuple_index * output_channels * input_channels_block_size * tuple_elements +
												((output_channels_block_start + output_channels_subblock_start) * input_channels_block_size) * tuple_elements,
											output_transform +
												tuple_index * tiles_count * output_channels * tuple_elements +
												(tiles_block_start * output_channels + (output_channels_block_start + output_channels_subblock_start) * tiles_block_size + tiles_subblock_start * output_channels_subblock_size) * tuple_elements,
											output_channels_subblock_size * tuple_elements);
									}
								}
							}
						}
					}
					NNP_BLOCK_MULTIPLICATION_END(profile)
				}
				NNP_OUTPUT_TRANSFORM_START(profile)
				for (size_t output_channels_block_start = 0; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max) {
					const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
					for (size_t output_channels_subblock_start = 0; output_channels_subblock_start < output_channels_block_size; output_channels_subblock_start += output_channels_subblock_max) {
						const size_t output_channels_subblock_size = min(output_channels_block_size - output_channels_subblock_start, output_channels_subblock_max);
						for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1) {
							const size_t output_channel = output_channels_block_start + output_channels_subblock_start + output_channels_subblock_offset;
							for (size_t tiles_block_start = 0; tiles_block_start < tiles_count; tiles_block_start += tiles_block_max) {
								const size_t tiles_block_size = min(tiles_count - tiles_block_start, tiles_block_max);
								for (size_t tiles_subblock_start = 0; tiles_subblock_start < tiles_block_size; tiles_subblock_start += tiles_subblock_max) {
									const size_t tiles_subblock_size = min(tiles_block_size - tiles_subblock_start, tiles_subblock_max);
									for (size_t tiles_subblock_offset = 0; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset += 1) {
										const size_t tile = tiles_block_start + tiles_subblock_start + tiles_subblock_offset;
										const struct fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
										const size_t tile_x = tile_xy.remainder;
										const size_t tile_y = tile_xy.quotient;

										const size_t output_x = tile_x * output_tile.width;
										const size_t output_y = tile_y * output_tile.height;

										output_transform_function(
											output_transform +
												(tiles_block_start * output_channels + (output_channels_block_start + output_channels_subblock_start) * tiles_block_size + (tiles_subblock_start + tiles_subblock_offset) * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
											&output[output_channel][output_y][output_x],
											&bias[output_channel],
											tiles_count * output_channels * tuple_size,
											output_size.width,
											min(output_tile.height, output_size.height - output_y),
											min(output_tile.width, output_size.width - output_x));
									}
								}
							}
						}
					}
				}
				NNP_OUTPUT_TRANSFORM_END(profile)
				break;
			case nnp_convolution_transform_strategy_block_based:
			case nnp_convolution_transform_strategy_precomputed:
				status = nnp_status_unsupported_transform_strategy;
				break;
			default:
				status = nnp_status_invalid_transform_strategy;
				break;
		}
	}

cleanup:
	release_memory(memory_block, memory_size);
	NNP_TOTAL_END(profile)
	return status;
}
