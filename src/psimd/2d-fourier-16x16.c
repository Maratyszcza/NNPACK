#include <stdint.h>
#include <stddef.h>

#include <psimd.h>

#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <psimd/fft/real.h>
#include <psimd/fft/soa.h>
#include <psimd/fft/dualreal.h>


union NNP_SIMD_ALIGN block16x16 {
	float as_float[16][16];
	psimd_f32 as_psimd_f32[16][4];
};


void nnp_fft16x16_with_offset__psimd(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	const uint32_t simd_width = 4;
	const uint32_t block_size = 16;
	transform_stride /= sizeof(float);

	union block16x16 block;
	if (column_count >= simd_width) {
		const psimd_f32 zero = psimd_zero_f32();
		const uint32_t column_end = column_offset + column_count;
		const uint32_t zero_column_start = column_end & (-simd_width);
		if (column_offset + simd_width >= zero_column_start) {
			float *restrict block_data = &block.as_float[0][0];
			for (uint32_t i = 0; i < block_size * block_size / (2 * simd_width); i++) {
				psimd_store_f32(block_data,              zero);
				psimd_store_f32(block_data + simd_width, zero);
				block_data += 2 * simd_width;
			}
		} else {
			for (uint32_t row = 0; row < block_size; row++) {
				for (uint32_t column = 0; column < column_end; column += simd_width) {
					psimd_store_f32(&block.as_float[row][column], zero);
				}
				for (uint32_t column = zero_column_start; column < block_size; column += simd_width) {
					psimd_store_f32(&block.as_float[row][column], zero);
				}
			}
		}

		const float *restrict row0 = data;
		const float *restrict row8 = data + doz(block_size / 2, row_offset) * data_stride;
		float* restrict output = &block.as_float[0][column_offset];
		do {
			const uint32_t column_block = min(column_count, simd_width);
			row0 += column_block;
			row8 += column_block;
			output += column_block;

			psimd_fft16_real_f32(
				row0 - simd_width, row8 - simd_width, data_stride, row_offset, row_count,
				output - simd_width, block_size);

			column_count -= column_block;
		} while (column_count != 0);
	} else {
		const psimd_f32 zero = psimd_zero_f32();
		float *restrict block_data = &block.as_float[0][0];
		for (uint32_t i = 0; i < block_size * block_size / (2 * simd_width); i++) {
			psimd_store_f32(block_data,              zero);
			psimd_store_f32(block_data + simd_width, zero);
			block_data += 2 * simd_width;
		}

		for (size_t row = 0; row < row_count; row++) {
			for (size_t column = 0; column < column_count; column++) {
				block.as_float[row_offset + row][column_offset + column] = data[row * data_stride + column];
			}
		}

		const uint32_t column = min(column_offset, block_size - simd_width);
		psimd_fft16_real_f32(
			&block.as_float[row_offset][column], &block.as_float[max(row_offset, block_size / 2)][column],
			block_size, row_offset, row_count,
			&block.as_float[0][column], block_size);
	}

	psimd_fft16_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1], &block.as_psimd_f32[0][2], &block.as_psimd_f32[0][3],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1], &block.as_psimd_f32[1][2], &block.as_psimd_f32[1][3]);

	for (size_t row = 2; row < block_size; row += 2) {
		psimd_fft16_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1], &block.as_psimd_f32[row    ][2], &block.as_psimd_f32[row    ][3],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1], &block.as_psimd_f32[row + 1][2], &block.as_psimd_f32[row + 1][3]);
	}

	for (size_t row = 0; row < block_size; row += 2) {
		for (size_t column = 0; column < block_size / simd_width; column += 1) {
			psimd_store_f32(transform ,             block.as_psimd_f32[row][column]);
			psimd_store_f32(transform + simd_width, block.as_psimd_f32[row + 1][column]);
			transform += transform_stride;
		}
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft16x16_with_offset__psimd(
	const float transform[],
	float data[],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	union block16x16 block;
	for (size_t row = 0; row < 16; row += 2) {
		for (size_t column = 0; column < 4; column += 1) {
			block.as_psimd_f32[row][column] = psimd_load_f32(transform + 0);
			block.as_psimd_f32[row + 1][column] = psimd_load_f32(transform + 4);
			transform += transform_stride;
		}
	}

	psimd_ifft16_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1], &block.as_psimd_f32[0][2], &block.as_psimd_f32[0][3],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1], &block.as_psimd_f32[1][2], &block.as_psimd_f32[1][3]);
	for (size_t row = 2; row < 16; row += 2) {
		psimd_ifft16_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1], &block.as_psimd_f32[row    ][2], &block.as_psimd_f32[row    ][3],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1], &block.as_psimd_f32[row + 1][2], &block.as_psimd_f32[row + 1][3]);
	}

	psimd_ifft16_real_f32(
		block.as_psimd_f32[ 0][0], block.as_psimd_f32[ 1][0], block.as_psimd_f32[ 2][0], block.as_psimd_f32[ 3][0],
		block.as_psimd_f32[ 4][0], block.as_psimd_f32[ 5][0], block.as_psimd_f32[ 6][0], block.as_psimd_f32[ 7][0],
		block.as_psimd_f32[ 8][0], block.as_psimd_f32[ 9][0], block.as_psimd_f32[10][0], block.as_psimd_f32[11][0],
		block.as_psimd_f32[12][0], block.as_psimd_f32[13][0], block.as_psimd_f32[14][0], block.as_psimd_f32[15][0],
		&block.as_float[0][0], &block.as_float[8][0], 16);
	const uint32_t column_end = column_offset + column_count;
	if (column_end > 4) {
		psimd_ifft16_real_f32(
			block.as_psimd_f32[ 0][1], block.as_psimd_f32[ 1][1], block.as_psimd_f32[ 2][1], block.as_psimd_f32[ 3][1],
			block.as_psimd_f32[ 4][1], block.as_psimd_f32[ 5][1], block.as_psimd_f32[ 6][1], block.as_psimd_f32[ 7][1],
			block.as_psimd_f32[ 8][1], block.as_psimd_f32[ 9][1], block.as_psimd_f32[10][1], block.as_psimd_f32[11][1],
			block.as_psimd_f32[12][1], block.as_psimd_f32[13][1], block.as_psimd_f32[14][1], block.as_psimd_f32[15][1],
			&block.as_float[0][4], &block.as_float[8][4], 16);
		if (column_end > 8) {
			psimd_ifft16_real_f32(
				block.as_psimd_f32[ 0][2], block.as_psimd_f32[ 1][2], block.as_psimd_f32[ 2][2], block.as_psimd_f32[ 3][2],
				block.as_psimd_f32[ 4][2], block.as_psimd_f32[ 5][2], block.as_psimd_f32[ 6][2], block.as_psimd_f32[ 7][2],
				block.as_psimd_f32[ 8][2], block.as_psimd_f32[ 9][2], block.as_psimd_f32[10][2], block.as_psimd_f32[11][2],
				block.as_psimd_f32[12][2], block.as_psimd_f32[13][2], block.as_psimd_f32[14][2], block.as_psimd_f32[15][2],
				&block.as_float[0][8], &block.as_float[8][8], 16);
			if (column_end > 12) {
				psimd_ifft16_real_f32(
					block.as_psimd_f32[ 0][3], block.as_psimd_f32[ 1][3], block.as_psimd_f32[ 2][3], block.as_psimd_f32[ 3][3],
					block.as_psimd_f32[ 4][3], block.as_psimd_f32[ 5][3], block.as_psimd_f32[ 6][3], block.as_psimd_f32[ 7][3],
					block.as_psimd_f32[ 8][3], block.as_psimd_f32[ 9][3], block.as_psimd_f32[10][3], block.as_psimd_f32[11][3],
					block.as_psimd_f32[12][3], block.as_psimd_f32[13][3], block.as_psimd_f32[14][3], block.as_psimd_f32[15][3],
					&block.as_float[0][12], &block.as_float[8][12], 16);
			}
		}
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row_offset + row][column_offset + column];
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft16x16_with_bias__psimd(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	union block16x16 block;
	for (size_t row = 0; row < 16; row += 2) {
		for (size_t column = 0; column < 4; column += 1) {
			block.as_psimd_f32[row][column] = psimd_load_f32(transform + 0);
			block.as_psimd_f32[row + 1][column] = psimd_load_f32(transform + 4);
			transform += transform_stride;
		}
	}

	block.as_float[0][0] += (*bias) * 256.0f;

	psimd_ifft16_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1], &block.as_psimd_f32[0][2], &block.as_psimd_f32[0][3],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1], &block.as_psimd_f32[1][2], &block.as_psimd_f32[1][3]);
	for (size_t row = 2; row < 16; row += 2) {
		psimd_ifft16_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1], &block.as_psimd_f32[row    ][2], &block.as_psimd_f32[row    ][3],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1], &block.as_psimd_f32[row + 1][2], &block.as_psimd_f32[row + 1][3]);
	}

	psimd_ifft16_real_f32(
		block.as_psimd_f32[ 0][0], block.as_psimd_f32[ 1][0], block.as_psimd_f32[ 2][0], block.as_psimd_f32[ 3][0],
		block.as_psimd_f32[ 4][0], block.as_psimd_f32[ 5][0], block.as_psimd_f32[ 6][0], block.as_psimd_f32[ 7][0],
		block.as_psimd_f32[ 8][0], block.as_psimd_f32[ 9][0], block.as_psimd_f32[10][0], block.as_psimd_f32[11][0],
		block.as_psimd_f32[12][0], block.as_psimd_f32[13][0], block.as_psimd_f32[14][0], block.as_psimd_f32[15][0],
		&block.as_float[0][0], &block.as_float[8][0], 16);
	if (column_count > 4) {
		psimd_ifft16_real_f32(
			block.as_psimd_f32[ 0][1], block.as_psimd_f32[ 1][1], block.as_psimd_f32[ 2][1], block.as_psimd_f32[ 3][1],
			block.as_psimd_f32[ 4][1], block.as_psimd_f32[ 5][1], block.as_psimd_f32[ 6][1], block.as_psimd_f32[ 7][1],
			block.as_psimd_f32[ 8][1], block.as_psimd_f32[ 9][1], block.as_psimd_f32[10][1], block.as_psimd_f32[11][1],
			block.as_psimd_f32[12][1], block.as_psimd_f32[13][1], block.as_psimd_f32[14][1], block.as_psimd_f32[15][1],
			&block.as_float[0][4], &block.as_float[8][4], 16);
		if (column_count > 8) {
			psimd_ifft16_real_f32(
				block.as_psimd_f32[ 0][2], block.as_psimd_f32[ 1][2], block.as_psimd_f32[ 2][2], block.as_psimd_f32[ 3][2],
				block.as_psimd_f32[ 4][2], block.as_psimd_f32[ 5][2], block.as_psimd_f32[ 6][2], block.as_psimd_f32[ 7][2],
				block.as_psimd_f32[ 8][2], block.as_psimd_f32[ 9][2], block.as_psimd_f32[10][2], block.as_psimd_f32[11][2],
				block.as_psimd_f32[12][2], block.as_psimd_f32[13][2], block.as_psimd_f32[14][2], block.as_psimd_f32[15][2],
				&block.as_float[0][8], &block.as_float[8][8], 16);
			if (column_count > 12) {
				psimd_ifft16_real_f32(
					block.as_psimd_f32[ 0][3], block.as_psimd_f32[ 1][3], block.as_psimd_f32[ 2][3], block.as_psimd_f32[ 3][3],
					block.as_psimd_f32[ 4][3], block.as_psimd_f32[ 5][3], block.as_psimd_f32[ 6][3], block.as_psimd_f32[ 7][3],
					block.as_psimd_f32[ 8][3], block.as_psimd_f32[ 9][3], block.as_psimd_f32[10][3], block.as_psimd_f32[11][3],
					block.as_psimd_f32[12][3], block.as_psimd_f32[13][3], block.as_psimd_f32[14][3], block.as_psimd_f32[15][3],
					&block.as_float[0][12], &block.as_float[8][12], 16);
			}
		}
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row][column];
		}
	}
}

void nnp_ifft16x16_with_bias_with_relu__psimd(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	union block16x16 block;
	for (size_t row = 0; row < 16; row += 2) {
		for (size_t column = 0; column < 4; column += 1) {
			block.as_psimd_f32[row][column] = psimd_load_f32(transform + 0);
			block.as_psimd_f32[row + 1][column] = psimd_load_f32(transform + 4);
			transform += transform_stride;
		}
	}

	block.as_float[0][0] += (*bias) * 256.0f;

	psimd_ifft16_dualreal_f32(
		&block.as_psimd_f32[0][0], &block.as_psimd_f32[0][1], &block.as_psimd_f32[0][2], &block.as_psimd_f32[0][3],
		&block.as_psimd_f32[1][0], &block.as_psimd_f32[1][1], &block.as_psimd_f32[1][2], &block.as_psimd_f32[1][3]);
	for (size_t row = 2; row < 16; row += 2) {
		psimd_ifft16_soa_f32(
			&block.as_psimd_f32[row    ][0], &block.as_psimd_f32[row    ][1], &block.as_psimd_f32[row    ][2], &block.as_psimd_f32[row    ][3],
			&block.as_psimd_f32[row + 1][0], &block.as_psimd_f32[row + 1][1], &block.as_psimd_f32[row + 1][2], &block.as_psimd_f32[row + 1][3]);
	}

	psimd_ifft16_real_f32(
		block.as_psimd_f32[ 0][0], block.as_psimd_f32[ 1][0], block.as_psimd_f32[ 2][0], block.as_psimd_f32[ 3][0],
		block.as_psimd_f32[ 4][0], block.as_psimd_f32[ 5][0], block.as_psimd_f32[ 6][0], block.as_psimd_f32[ 7][0],
		block.as_psimd_f32[ 8][0], block.as_psimd_f32[ 9][0], block.as_psimd_f32[10][0], block.as_psimd_f32[11][0],
		block.as_psimd_f32[12][0], block.as_psimd_f32[13][0], block.as_psimd_f32[14][0], block.as_psimd_f32[15][0],
		&block.as_float[0][0], &block.as_float[8][0], 16);
	if (column_count > 4) {
		psimd_ifft16_real_f32(
			block.as_psimd_f32[ 0][1], block.as_psimd_f32[ 1][1], block.as_psimd_f32[ 2][1], block.as_psimd_f32[ 3][1],
			block.as_psimd_f32[ 4][1], block.as_psimd_f32[ 5][1], block.as_psimd_f32[ 6][1], block.as_psimd_f32[ 7][1],
			block.as_psimd_f32[ 8][1], block.as_psimd_f32[ 9][1], block.as_psimd_f32[10][1], block.as_psimd_f32[11][1],
			block.as_psimd_f32[12][1], block.as_psimd_f32[13][1], block.as_psimd_f32[14][1], block.as_psimd_f32[15][1],
			&block.as_float[0][4], &block.as_float[8][4], 16);
		if (column_count > 8) {
			psimd_ifft16_real_f32(
				block.as_psimd_f32[ 0][2], block.as_psimd_f32[ 1][2], block.as_psimd_f32[ 2][2], block.as_psimd_f32[ 3][2],
				block.as_psimd_f32[ 4][2], block.as_psimd_f32[ 5][2], block.as_psimd_f32[ 6][2], block.as_psimd_f32[ 7][2],
				block.as_psimd_f32[ 8][2], block.as_psimd_f32[ 9][2], block.as_psimd_f32[10][2], block.as_psimd_f32[11][2],
				block.as_psimd_f32[12][2], block.as_psimd_f32[13][2], block.as_psimd_f32[14][2], block.as_psimd_f32[15][2],
				&block.as_float[0][8], &block.as_float[8][8], 16);
			if (column_count > 12) {
				psimd_ifft16_real_f32(
					block.as_psimd_f32[ 0][3], block.as_psimd_f32[ 1][3], block.as_psimd_f32[ 2][3], block.as_psimd_f32[ 3][3],
					block.as_psimd_f32[ 4][3], block.as_psimd_f32[ 5][3], block.as_psimd_f32[ 6][3], block.as_psimd_f32[ 7][3],
					block.as_psimd_f32[ 8][3], block.as_psimd_f32[ 9][3], block.as_psimd_f32[10][3], block.as_psimd_f32[11][3],
					block.as_psimd_f32[12][3], block.as_psimd_f32[13][3], block.as_psimd_f32[14][3], block.as_psimd_f32[15][3],
					&block.as_float[0][12], &block.as_float[8][12], 16);
			}
		}
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block.as_float[row][column], 0.0f);
		}
	}
}
