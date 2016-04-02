#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

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
	enum nnp_convolution_kernel_transform_strategy kernel_transform_strategy,
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
	void (*input_transform_function)(const float[], float[], size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t) = NULL;
	void (*kernel_transform_function)(const float[], float[], size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t) = NULL;
	void (*kernel_fourier_transform_and_macc_function)(const float[], float[], const float[], size_t, uint32_t, uint32_t, uint32_t, uint32_t) = NULL;
	void (*kernel_winograd_transform_and_mac_function)(const float[], float[], const float[], size_t) = NULL;
	void (*macc_function)(float[], const float[], const float[]) = NULL;
	void (*output_transform_function)(const float[], float[], const float[], size_t, size_t, uint32_t, uint32_t) = NULL;
	switch (algorithm) {
		case nnp_convolution_algorithm_wt8x8:
			if ((kernel_size.height != 3) || (kernel_size.width != 3)) {
				status = nnp_status_unsupported_algorithm;
				goto cleanup;
			}
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
		#if NNP_ARCH_X86_64
			input_transform_function = nnp_iwt8x8_3x3_and_store__avx2;
			kernel_transform_function = nnp_kwt8x8_3x3_and_stream__avx2;
			kernel_winograd_transform_and_mac_function = nnp_kwt8x8_3x3_and_mac__avx2;
			macc_function = nnp_s8x8gemm__fma3;
			output_transform_function = nnp_owt8x8_3x3_with_bias__avx2;
		#endif
			fourier_transform = false;
			break;
		case nnp_convolution_algorithm_ft8x8:
			tile_size = (struct nnp_size) { .height = 8, .width = 8 };
		#if NNP_ARCH_X86_64
			input_transform_function = nnp_fft8x8_and_store__avx2;
			kernel_transform_function = nnp_fft8x8_and_stream__avx2;
			kernel_fourier_transform_and_macc_function = nnp_fft8x8_and_macc__avx2;
			macc_function = nnp_ft8x8gemmc__fma3;
			output_transform_function = nnp_ifft8x8_with_bias__avx2;
		#endif
			fourier_transform = true;
			break;
		case nnp_convolution_algorithm_ft16x16:
			tile_size = (struct nnp_size) { .height = 16, .width = 16 };
		#if NNP_ARCH_X86_64
			input_transform_function = nnp_fft16x16_and_store__avx2;
			kernel_transform_function = nnp_fft16x16_and_stream__avx2;
			kernel_fourier_transform_and_macc_function = nnp_fft16x16_and_macc__avx2;
			macc_function = nnp_ft16x16gemmc__fma3;
			output_transform_function = nnp_ifft16x16_with_bias__avx2;
		#endif
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

	const size_t transform_tile_size = tile_elements * sizeof(float);
	const size_t input_transform_size = input_channels * transform_tile_size;
	const size_t kernel_transform_size = output_channels * input_channels * transform_tile_size;
	const size_t output_transform_size = output_channels * transform_tile_size;
	size_t memory_size = input_transform_size + output_transform_size;
	if (kernel_transform_strategy == nnp_convolution_kernel_transform_strategy_reuse) {
		memory_size += kernel_transform_size;
	}

	memory_block = allocate_memory(memory_size);
	if (memory_block == NULL) {
		status = nnp_status_out_of_memory;
		goto cleanup;
	}

	float* input_transform = memory_block;
	float* output_transform = memory_block + input_transform_size;
	float* kernel_transform = NULL;
	if (kernel_transform_strategy == nnp_convolution_kernel_transform_strategy_reuse) {
		kernel_transform = memory_block + input_transform_size + output_transform_size;
	}

	const size_t input_channels_block_max = 16 / (tile_elements / 64);
	const size_t output_channels_block_max = 16 / (tile_elements / 64);

	{
		const float (*input)[input_size.width * input_size.height] =
			(const float(*)[input_size.width * input_size.height]) input_pointer;
		const float (*kernel)[input_channels][kernel_size.width * kernel_size.height] =
			(const float(*)[input_channels][kernel_size.width * kernel_size.height]) kernel_pointer;
		float (*output)[output_size.width * output_size.height] =
			(float(*)[output_size.width * output_size.height]) output_pointer;

		switch (kernel_transform_strategy) {
			case nnp_convolution_kernel_transform_strategy_recompute:
				for (size_t y = 0; y < output_size.height; y += output_tile.height) {
					const size_t input_y = min(doz(y, input_padding.top), input_size.height);
					for (size_t x = 0; x < output_size.width; x += output_tile.width) {
						const size_t input_x = min(doz(x, input_padding.left), input_size.width);

						{
							NNP_INPUT_TRANSFORM_START(profile)
							for (size_t input_channel = 0; input_channel < input_channels; input_channel++) {
								input_transform_function(
									&input[input_channel][input_y * input_size.width + input_x],
									input_transform + input_channel * tile_elements,
									input_size.width,
									tuple_size,
									min(input_tile.height, doz(input_size.height, input_y)),
									min(input_tile.width, doz(input_size.width, input_x)),
									doz(input_padding.top, y),
									doz(input_padding.left, x));
							}
							NNP_INPUT_TRANSFORM_END(profile)
						}

						for (size_t output_channel = 0; output_channel < output_channels; output_channel++) {
							memset(output_transform + output_channel * tile_elements, 0, tile_elements * sizeof(float));

							NNP_KERNEL_TRANSFORM_START(profile)
							for (size_t input_channel = 0; input_channel < input_channels; input_channel++) {
								if (fourier_transform) {
									kernel_fourier_transform_and_macc_function(
										kernel[output_channel][input_channel],
										output_transform + output_channel * tile_elements,
										input_transform + input_channel * tile_elements,
										kernel_size.width,
										kernel_size.height, kernel_size.width, 0, 0);
								} else {
									kernel_winograd_transform_and_mac_function(
										kernel[output_channel][input_channel],
										output_transform + output_channel * tile_elements,
										input_transform + input_channel * tile_elements,
										kernel_size.width);
								}
							}
							NNP_KERNEL_TRANSFORM_END(profile)

							NNP_OUTPUT_TRANSFORM_START(profile)
							output_transform_function(
								output_transform + output_channel * tile_elements,
								&output[output_channel][y * output_size.width + x],
								&bias[output_channel],
								tuple_size,
								output_size.width,
								min(output_tile.height, output_size.height - y),
								min(output_tile.width, output_size.width - x));
							NNP_OUTPUT_TRANSFORM_END(profile)
						}
					}
				}
				break;
			case nnp_convolution_kernel_transform_strategy_reuse:
				{
					for (size_t output_channel = 0; output_channel < output_channels; output_channel++) {
						for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
							const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);
							for (size_t input_channels_block_offset = 0; input_channels_block_offset < input_channels_block_size; input_channels_block_offset++) {
								const size_t input_channel = input_channels_block_start + input_channels_block_offset;
								kernel_transform_function(
									kernel[output_channel][input_channel],
									kernel_transform + (input_channels_block_start * output_channels + output_channel * input_channels_block_size + input_channels_block_offset) * tile_elements,
									kernel_size.width,
									tuple_size,
									kernel_size.height, kernel_size.width, 0, 0);
							}
						}
					}
				}
				for (size_t y = 0; y < output_size.height; y += output_tile.height) {
					const size_t input_y = min(doz(y, input_padding.top), input_size.height);
					for (size_t x = 0; x < output_size.width; x += output_tile.width) {
						const size_t input_x = min(doz(x, input_padding.left), input_size.width);

						for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
							const size_t input_channels_block_end = min(input_channels_block_start + input_channels_block_max, input_channels);
							for (size_t input_channel = input_channels_block_start; input_channel < input_channels_block_end; input_channel++) {
								input_transform_function(
									&input[input_channel][input_y * input_size.width + input_x],
									input_transform + input_channel * tile_elements,
									input_size.width,
									tuple_size,
									min(input_tile.height, doz(input_size.height, input_y)),
									min(input_tile.width, doz(input_size.width, input_x)),
									doz(input_padding.top, y),
									doz(input_padding.left, x));
							}
						}

						for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) {
							const size_t input_channels_block_size = min(input_channels_block_max, input_channels - input_channels_block_start);
							for (size_t output_channels_block_start = 0; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max) {
								const size_t output_channels_block_end = min(output_channels_block_start + output_channels_block_max, output_channels);
								for (size_t output_channel = output_channels_block_start; output_channel < output_channels_block_end; output_channel++) {
									if (input_channels_block_start == 0) {
										memset(output_transform + output_channel * tile_elements, 0, tile_elements * sizeof(float));
									}

									for (size_t input_channels_block_offset = 0; input_channels_block_offset < input_channels_block_size; input_channels_block_offset++) {
										macc_function(
											output_transform + output_channel * tile_elements,
											input_transform + (input_channels_block_start + input_channels_block_offset) * tile_elements,
											kernel_transform + (input_channels_block_start * output_channels + output_channel * input_channels_block_size + input_channels_block_offset) * tile_elements);
									}

									if (input_channels_block_start + input_channels_block_size == input_channels) {
										output_transform_function(
											output_transform + output_channel * tile_elements,
											&output[output_channel][y * output_size.width + x],
											&bias[output_channel],
											tuple_size,
											output_size.width,
											min(output_tile.height, output_size.height - y),
											min(output_tile.width, output_size.width - x));
									}
								}
							}
						}
					}
				}
				break;
			case nnp_convolution_kernel_transform_strategy_precomputed:
				NNP_UNREACHABLE;
				break;
		}
	}

cleanup:
	release_memory(memory_block, memory_size);
	NNP_TOTAL_END(profile)
	return status;
}
