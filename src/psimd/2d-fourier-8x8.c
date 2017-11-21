#include <stdint.h>
#include <stddef.h>

#include <psimd.h>

#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <psimd/fft/real.h>
#include <psimd/fft/soa.h>
#include <psimd/fft/dualreal.h>


union NNP_SIMD_ALIGN block8x8 {
	float as_float[8][8];
	psimd_f32 as_psimd_f32[8][2];
};


void nnp_fft8x8_with_offset__psimd(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	const uint32_t simd_width = 4;
	const uint32_t block_size = 8;
	transform_stride /= sizeof(float);

	union block8x8 block;
	if (column_count >= simd_width) {
		const psimd_f32 zero = psimd_zero_f32();
		float *restrict block_data = &block.as_float[0][0];
		const uint32_t column_end = column_offset + column_count;
		if (column_offset != 0 && column_end != block_size) {
			for (uint32_t row = 0; row < block_size; row++) {
				psimd_store_f32(block_data,              zero);
				psimd_store_f32(block_data + simd_width, zero);
				block_data += block_size;
			}
		} else {
			if (column_offset != 0) {
				for (uint32_t row = 0; row < block_size; row++) {
					psimd_store_f32(block_data, zero);
					block_data += block_size;
				}
			} else if (column_end != block_size) {
				for (uint32_t row = 0; row < block_size; row++) {
					psimd_store_f32(block_data + simd_width, zero);
					block_data += block_size;
				}
			}
		}

		const float *restrict row0 = data;
		const float *restrict row4 = data + doz(block_size / 2, row_offset) * data_stride;
		float* restrict output = &block.as_float[0][column_offset];
		do {
			const uint32_t column_block = min(column_count, simd_width);
			row0 += column_block;
			row4 += column_block;
			output += column_block;

			psimd_fft8_real_f32(
				row0 - simd_width, row4 - simd_width, data_stride, row_offset, row_count,
				output - simd_width, block_size);

			column_count -= column_block;
		} while (column_count != 0);
	} else {
		const psimd_f32 zero = psimd_zero_f32();
		float *restrict block_data = &block.as_float[0][0];
		for (uint32_t row = 0; row < block_size; row++) {
			psimd_store_f32(block_data,              zero);
			psimd_store_f32(block_data + simd_width, zero);
			block_data += block_size;
		}

		for (size_t row = 0; row < row_count; row++) {
			for (size_t column = 0; column < column_count; column++) {
				block.as_float[row_offset + row][column_offset + column] = data[row * data_stride + column];
			}
		}

		const uint32_t column = min(column_offset, block_size - simd_width);
		psimd_fft8_real_f32(
			&block.as_float[row_offset][column], &block.as_float[max(row_offset, block_size / 2)][column],
			block_size, row_offset, row_count,
			&block.as_float[0][column], block_size);
	}

	psimd_fft8_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1]);
	for (size_t row = 2; row < block_size; row += 2) {
		psimd_fft8_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1]);
	}

	for (size_t row = 0; row < block_size; row += 2) {
		for (size_t column = 0; column < 2; column += 1) {
			psimd_store_f32(transform,              block.as_psimd_f32[row][column]);
			psimd_store_f32(transform + simd_width, block.as_psimd_f32[row + 1][column]);
			transform += transform_stride;
		}
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft8x8_with_offset__psimd(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	union block8x8 block;
	for (size_t row = 0; row < 8; row += 2) {
		for (size_t column = 0; column < 2; column += 1) {
			block.as_psimd_f32[row][column] = psimd_load_f32(transform + 0);
			block.as_psimd_f32[row + 1][column] = psimd_load_f32(transform + 4);
			transform += transform_stride;
		}
	}

	psimd_ifft8_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1]);
	for (size_t row = 2; row < 8; row += 2) {
		psimd_ifft8_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1]);
	}

	psimd_ifft8_real_f32(
		block.as_psimd_f32[0][0], block.as_psimd_f32[1][0], block.as_psimd_f32[2][0], block.as_psimd_f32[3][0],
		block.as_psimd_f32[4][0], block.as_psimd_f32[5][0], block.as_psimd_f32[6][0], block.as_psimd_f32[7][0],
		&block.as_float[0][0], &block.as_float[4][0], 8);
	const uint32_t column_end = column_offset + column_count;
	if (column_end > 4) {
		psimd_ifft8_real_f32(
			block.as_psimd_f32[0][1], block.as_psimd_f32[1][1], block.as_psimd_f32[2][1], block.as_psimd_f32[3][1],
			block.as_psimd_f32[4][1], block.as_psimd_f32[5][1], block.as_psimd_f32[6][1], block.as_psimd_f32[7][1],
			&block.as_float[0][4], &block.as_float[4][4], 8);
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row_offset + row][column_offset + column];
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft8x8_with_bias__psimd(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	union block8x8 block;
	for (size_t row = 0; row < 8; row += 2) {
		for (size_t column = 0; column < 2; column += 1) {
			block.as_psimd_f32[row][column] = psimd_load_f32(transform + 0);
			block.as_psimd_f32[row + 1][column] = psimd_load_f32(transform + 4);
			transform += transform_stride;
		}
	}

	block.as_float[0][0] += (*bias) * 64.0f;

	psimd_ifft8_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1]);
	for (size_t row = 2; row < 8; row += 2) {
		psimd_ifft8_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1]);
	}

	psimd_ifft8_real_f32(
		block.as_psimd_f32[0][0], block.as_psimd_f32[1][0], block.as_psimd_f32[2][0], block.as_psimd_f32[3][0],
		block.as_psimd_f32[4][0], block.as_psimd_f32[5][0], block.as_psimd_f32[6][0], block.as_psimd_f32[7][0],
		&block.as_float[0][0], &block.as_float[4][0], 8);
	if (column_count > 4) {
		psimd_ifft8_real_f32(
			block.as_psimd_f32[0][1], block.as_psimd_f32[1][1], block.as_psimd_f32[2][1], block.as_psimd_f32[3][1],
			block.as_psimd_f32[4][1], block.as_psimd_f32[5][1], block.as_psimd_f32[6][1], block.as_psimd_f32[7][1],
			&block.as_float[0][4], &block.as_float[4][4], 8);
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row][column];
		}
	}
}

void nnp_ifft8x8_with_bias_with_relu__psimd(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	union block8x8 block;
	for (size_t row = 0; row < 8; row += 2) {
		for (size_t column = 0; column < 2; column += 1) {
			block.as_psimd_f32[row][column] = psimd_load_f32(transform + 0);
			block.as_psimd_f32[row + 1][column] = psimd_load_f32(transform + 4);
			transform += transform_stride;
		}
	}

	block.as_float[0][0] += (*bias) * 64.0f;

	psimd_ifft8_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1]);
	for (size_t row = 2; row < 8; row += 2) {
		psimd_ifft8_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1]);
	}

	psimd_ifft8_real_f32(
		block.as_psimd_f32[0][0], block.as_psimd_f32[1][0], block.as_psimd_f32[2][0], block.as_psimd_f32[3][0],
		block.as_psimd_f32[4][0], block.as_psimd_f32[5][0], block.as_psimd_f32[6][0], block.as_psimd_f32[7][0],
		&block.as_float[0][0], &block.as_float[4][0], 8);
	if (column_count > 4) {
		psimd_ifft8_real_f32(
			block.as_psimd_f32[0][1], block.as_psimd_f32[1][1], block.as_psimd_f32[2][1], block.as_psimd_f32[3][1],
			block.as_psimd_f32[4][1], block.as_psimd_f32[5][1], block.as_psimd_f32[6][1], block.as_psimd_f32[7][1],
			&block.as_float[0][4], &block.as_float[4][4], 8);
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block.as_float[row][column], 0.0f);
		}
	}
}
