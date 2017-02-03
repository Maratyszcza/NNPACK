#include <stdint.h>
#include <stddef.h>

#include <nnpack/utils.h>

#include <psimd/fft/real.h>
#include <psimd/fft/soa.h>
#include <psimd/fft/dualreal.h>


union NNP_SIMD_ALIGN block8x8 {
	float as_float[8][8];
	v4f as_v4f[8][2];
};


void nnp_fft8x8__psimd(
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
		const v4f zero = v4f_zero();
		float *restrict block_data = &block.as_float[0][0];
		const uint32_t column_end = column_offset + column_count;
		if (column_offset != 0 && column_end != block_size) {
			for (uint32_t row = 0; row < block_size; row++) {
				v4f_st(block_data,              zero);
				v4f_st(block_data + simd_width, zero);
				block_data += block_size;
			}
		} else {
			if (column_offset != 0) {
				for (uint32_t row = 0; row < block_size; row++) {
					v4f_st(block_data, zero);
					block_data += block_size;
				}
			} else if (column_end != block_size) {
				for (uint32_t row = 0; row < block_size; row++) {
					v4f_st(block_data + simd_width, zero);
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

			v4f_fft8_real(
				row0 - simd_width, row4 - simd_width, data_stride, row_offset, row_count,
				output - simd_width, block_size);

			column_count -= column_block;
		} while (column_count != 0);
	} else {
		const v4f zero = v4f_zero();
		float *restrict block_data = &block.as_float[0][0];
		for (uint32_t row = 0; row < block_size; row++) {
			v4f_st(block_data,              zero);
			v4f_st(block_data + simd_width, zero);
			block_data += block_size;
		}

		for (size_t row = 0; row < row_count; row++) {
			for (size_t column = 0; column < column_count; column++) {
				block.as_float[row_offset + row][column_offset + column] = data[row * data_stride + column];
			}
		}

		const uint32_t column = min(column_offset, block_size - simd_width);
		v4f_fft8_real(
			&block.as_float[row_offset][column], &block.as_float[max(row_offset, block_size / 2)][column],
			block_size, row_offset, row_count,
			&block.as_float[0][column], block_size);
	}

	v4f_fft8_dualreal(
		&block.as_v4f[0][0], &block.as_v4f[0][1],
		&block.as_v4f[1][0], &block.as_v4f[1][1]);
	for (size_t row = 2; row < block_size; row += 2) {
		v4f_fft8_soa(
			&block.as_v4f[row    ][0], &block.as_v4f[row    ][1],
			&block.as_v4f[row + 1][0], &block.as_v4f[row + 1][1]);
	}

	for (size_t row = 0; row < block_size; row += 2) {
		for (size_t column = 0; column < 2; column += 1) {
			v4f_st(transform,              block.as_v4f[row][column]);
			v4f_st(transform + simd_width, block.as_v4f[row + 1][column]);
			transform += transform_stride;
		}
	}
}

void nnp_ifft8x8__psimd(
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
			block.as_v4f[row][column] = v4f_ld(transform + 0);
			block.as_v4f[row + 1][column] = v4f_ld(transform + 4);
			transform += transform_stride;
		}
	}

	v4f_ifft8_dualreal(
		&block.as_v4f[0][0], &block.as_v4f[0][1],
		&block.as_v4f[1][0], &block.as_v4f[1][1]);
	for (size_t row = 2; row < 8; row += 2) {
		v4f_ifft8_soa(
			&block.as_v4f[row    ][0], &block.as_v4f[row    ][1],
			&block.as_v4f[row + 1][0], &block.as_v4f[row + 1][1]);
	}

	v4f_ifft8_real(
		block.as_v4f[0][0], block.as_v4f[1][0], block.as_v4f[2][0], block.as_v4f[3][0],
		block.as_v4f[4][0], block.as_v4f[5][0], block.as_v4f[6][0], block.as_v4f[7][0],
		&block.as_float[0][0], &block.as_float[4][0], 8);
	const uint32_t column_end = column_offset + column_count;
	if (column_end > 4) {
		v4f_ifft8_real(
			block.as_v4f[0][1], block.as_v4f[1][1], block.as_v4f[2][1], block.as_v4f[3][1],
			block.as_v4f[4][1], block.as_v4f[5][1], block.as_v4f[6][1], block.as_v4f[7][1],
			&block.as_float[0][4], &block.as_float[4][4], 8);
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row_offset + row][column_offset + column];
		}
	}
}

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
			block.as_v4f[row][column] = v4f_ld(transform + 0);
			block.as_v4f[row + 1][column] = v4f_ld(transform + 4);
			transform += transform_stride;
		}
	}

	v4f_ifft8_dualreal(
		&block.as_v4f[0][0], &block.as_v4f[0][1],
		&block.as_v4f[1][0], &block.as_v4f[1][1]);
	for (size_t row = 2; row < 8; row += 2) {
		v4f_ifft8_soa(
			&block.as_v4f[row    ][0], &block.as_v4f[row    ][1],
			&block.as_v4f[row + 1][0], &block.as_v4f[row + 1][1]);
	}

	v4f_ifft8_real(
		block.as_v4f[0][0], block.as_v4f[1][0], block.as_v4f[2][0], block.as_v4f[3][0],
		block.as_v4f[4][0], block.as_v4f[5][0], block.as_v4f[6][0], block.as_v4f[7][0],
		&block.as_float[0][0], &block.as_float[4][0], 8);
	if (column_count > 4) {
		v4f_ifft8_real(
			block.as_v4f[0][1], block.as_v4f[1][1], block.as_v4f[2][1], block.as_v4f[3][1],
			block.as_v4f[4][1], block.as_v4f[5][1], block.as_v4f[6][1], block.as_v4f[7][1],
			&block.as_float[0][4], &block.as_float[4][4], 8);
	}

	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row][column] + (*bias);
		}
	}
}
