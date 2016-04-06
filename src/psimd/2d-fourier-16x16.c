#include <stdint.h>
#include <stddef.h>
#include <complex.h>

#include <psimd/fft/real.h>
#include <psimd/fft/soa.h>
#include <psimd/fft/dualreal.h>


union NNP_SIMD_ALIGN block16x16 {
	float as_float[16][16];
	v4f as_v4f[16][4];
};


void nnp_fft16x16__psimd(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	union block16x16 block = { 0 };
	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			block.as_float[row_offset + row][column_offset + column] = data[row * data_stride + column];
		}
	}

	v4f_fft16_real(
		&block.as_float[0][0], &block.as_float[8][0], 16,
		&block.as_v4f[ 0][0], &block.as_v4f[ 1][0], &block.as_v4f[ 2][0], &block.as_v4f[ 3][0],
		&block.as_v4f[ 4][0], &block.as_v4f[ 5][0], &block.as_v4f[ 6][0], &block.as_v4f[ 7][0],
		&block.as_v4f[ 8][0], &block.as_v4f[ 9][0], &block.as_v4f[10][0], &block.as_v4f[11][0],
		&block.as_v4f[12][0], &block.as_v4f[13][0], &block.as_v4f[14][0], &block.as_v4f[15][0]);
	const uint32_t column_end = column_offset + column_count;
	if (column_end > 4) {
		v4f_fft16_real(
			&block.as_float[0][4], &block.as_float[8][4], 16,
			&block.as_v4f[ 0][1], &block.as_v4f[ 1][1], &block.as_v4f[ 2][1], &block.as_v4f[ 3][1],
			&block.as_v4f[ 4][1], &block.as_v4f[ 5][1], &block.as_v4f[ 6][1], &block.as_v4f[ 7][1],
			&block.as_v4f[ 8][1], &block.as_v4f[ 9][1], &block.as_v4f[10][1], &block.as_v4f[11][1],
			&block.as_v4f[12][1], &block.as_v4f[13][1], &block.as_v4f[14][1], &block.as_v4f[15][1]);
		if (column_end > 8) {
			v4f_fft16_real(
				&block.as_float[0][8], &block.as_float[8][8], 16,
				&block.as_v4f[ 0][2], &block.as_v4f[ 1][2], &block.as_v4f[ 2][2], &block.as_v4f[ 3][2],
				&block.as_v4f[ 4][2], &block.as_v4f[ 5][2], &block.as_v4f[ 6][2], &block.as_v4f[ 7][2],
				&block.as_v4f[ 8][2], &block.as_v4f[ 9][2], &block.as_v4f[10][2], &block.as_v4f[11][2],
				&block.as_v4f[12][2], &block.as_v4f[13][2], &block.as_v4f[14][2], &block.as_v4f[15][2]);
			if (column_end > 12) {
				v4f_fft16_real(
					&block.as_float[0][12], &block.as_float[8][12], 16,
					&block.as_v4f[ 0][3], &block.as_v4f[ 1][3], &block.as_v4f[ 2][3], &block.as_v4f[ 3][3],
					&block.as_v4f[ 4][3], &block.as_v4f[ 5][3], &block.as_v4f[ 6][3], &block.as_v4f[ 7][3],
					&block.as_v4f[ 8][3], &block.as_v4f[ 9][3], &block.as_v4f[10][3], &block.as_v4f[11][3],
					&block.as_v4f[12][3], &block.as_v4f[13][3], &block.as_v4f[14][3], &block.as_v4f[15][3]);
			}
		}
	}

	v4f_fft16_dualreal(
		&block.as_v4f[0][0], &block.as_v4f[0][1], &block.as_v4f[0][2], &block.as_v4f[0][3],
		&block.as_v4f[1][0], &block.as_v4f[1][1], &block.as_v4f[1][2], &block.as_v4f[1][3]);

	for (size_t row = 2; row < 16; row += 2) {
		v4f_fft16_soa(
			&block.as_v4f[row    ][0], &block.as_v4f[row    ][1], &block.as_v4f[row    ][2], &block.as_v4f[row    ][3],
			&block.as_v4f[row + 1][0], &block.as_v4f[row + 1][1], &block.as_v4f[row + 1][2], &block.as_v4f[row + 1][3]);
	}

	for (size_t row = 0; row < 16; row += 2) {
		for (size_t column = 0; column < 4; column += 1) {
			v4f_st(transform + 0, block.as_v4f[row][column]);
			v4f_st(transform + 4, block.as_v4f[row + 1][column]);
			transform += transform_stride;
		}
	}
}

void nnp_ifft16x16__psimd(
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
			block.as_v4f[row][column] = v4f_ld(transform + 0);
			block.as_v4f[row + 1][column] = v4f_ld(transform + 4);
			transform += transform_stride;
		}
	}

	v4f_ifft16_dualreal(
		&block.as_v4f[0][0], &block.as_v4f[0][1], &block.as_v4f[0][2], &block.as_v4f[0][3],
		&block.as_v4f[1][0], &block.as_v4f[1][1], &block.as_v4f[1][2], &block.as_v4f[1][3]);
	for (size_t row = 2; row < 16; row += 2) {
		v4f_ifft16_soa(
			&block.as_v4f[row    ][0], &block.as_v4f[row    ][1], &block.as_v4f[row    ][2], &block.as_v4f[row    ][3],
			&block.as_v4f[row + 1][0], &block.as_v4f[row + 1][1], &block.as_v4f[row + 1][2], &block.as_v4f[row + 1][3]);
	}

	v4f_ifft16_real(
		block.as_v4f[ 0][0], block.as_v4f[ 1][0], block.as_v4f[ 2][0], block.as_v4f[ 3][0],
		block.as_v4f[ 4][0], block.as_v4f[ 5][0], block.as_v4f[ 6][0], block.as_v4f[ 7][0],
		block.as_v4f[ 8][0], block.as_v4f[ 9][0], block.as_v4f[10][0], block.as_v4f[11][0],
		block.as_v4f[12][0], block.as_v4f[13][0], block.as_v4f[14][0], block.as_v4f[15][0],
		&block.as_float[0][0], &block.as_float[8][0], 16);
	const uint32_t column_end = column_offset + column_count;
	if (column_end > 4) {
		v4f_ifft16_real(
			block.as_v4f[ 0][1], block.as_v4f[ 1][1], block.as_v4f[ 2][1], block.as_v4f[ 3][1],
			block.as_v4f[ 4][1], block.as_v4f[ 5][1], block.as_v4f[ 6][1], block.as_v4f[ 7][1],
			block.as_v4f[ 8][1], block.as_v4f[ 9][1], block.as_v4f[10][1], block.as_v4f[11][1],
			block.as_v4f[12][1], block.as_v4f[13][1], block.as_v4f[14][1], block.as_v4f[15][1],
			&block.as_float[0][4], &block.as_float[8][4], 16);
		if (column_end > 8) {
			v4f_ifft16_real(
				block.as_v4f[ 0][2], block.as_v4f[ 1][2], block.as_v4f[ 2][2], block.as_v4f[ 3][2],
				block.as_v4f[ 4][2], block.as_v4f[ 5][2], block.as_v4f[ 6][2], block.as_v4f[ 7][2],
				block.as_v4f[ 8][2], block.as_v4f[ 9][2], block.as_v4f[10][2], block.as_v4f[11][2],
				block.as_v4f[12][2], block.as_v4f[13][2], block.as_v4f[14][2], block.as_v4f[15][2],
				&block.as_float[0][8], &block.as_float[8][8], 16);
			if (column_end > 12) {
				v4f_ifft16_real(
					block.as_v4f[ 0][3], block.as_v4f[ 1][3], block.as_v4f[ 2][3], block.as_v4f[ 3][3],
					block.as_v4f[ 4][3], block.as_v4f[ 5][3], block.as_v4f[ 6][3], block.as_v4f[ 7][3],
					block.as_v4f[ 8][3], block.as_v4f[ 9][3], block.as_v4f[10][3], block.as_v4f[11][3],
					block.as_v4f[12][3], block.as_v4f[13][3], block.as_v4f[14][3], block.as_v4f[15][3],
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

void nnp_ifft16x16_with_bias__psimd(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias_ptr[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	union block16x16 block;
	for (size_t row = 0; row < 16; row += 2) {
		for (size_t column = 0; column < 4; column += 1) {
			block.as_v4f[row][column] = v4f_ld(transform + 0);
			block.as_v4f[row + 1][column] = v4f_ld(transform + 4);
			transform += transform_stride;
		}
	}

	v4f_ifft16_dualreal(
		&block.as_v4f[0][0], &block.as_v4f[0][1], &block.as_v4f[0][2], &block.as_v4f[0][3],
		&block.as_v4f[1][0], &block.as_v4f[1][1], &block.as_v4f[1][2], &block.as_v4f[1][3]);
	for (size_t row = 2; row < 16; row += 2) {
		v4f_ifft16_soa(
			&block.as_v4f[row    ][0], &block.as_v4f[row    ][1], &block.as_v4f[row    ][2], &block.as_v4f[row    ][3],
			&block.as_v4f[row + 1][0], &block.as_v4f[row + 1][1], &block.as_v4f[row + 1][2], &block.as_v4f[row + 1][3]);
	}

	v4f_ifft16_real(
		block.as_v4f[ 0][0], block.as_v4f[ 1][0], block.as_v4f[ 2][0], block.as_v4f[ 3][0],
		block.as_v4f[ 4][0], block.as_v4f[ 5][0], block.as_v4f[ 6][0], block.as_v4f[ 7][0],
		block.as_v4f[ 8][0], block.as_v4f[ 9][0], block.as_v4f[10][0], block.as_v4f[11][0],
		block.as_v4f[12][0], block.as_v4f[13][0], block.as_v4f[14][0], block.as_v4f[15][0],
		&block.as_float[0][0], &block.as_float[8][0], 16);
	if (column_count > 4) {
		v4f_ifft16_real(
			block.as_v4f[ 0][1], block.as_v4f[ 1][1], block.as_v4f[ 2][1], block.as_v4f[ 3][1],
			block.as_v4f[ 4][1], block.as_v4f[ 5][1], block.as_v4f[ 6][1], block.as_v4f[ 7][1],
			block.as_v4f[ 8][1], block.as_v4f[ 9][1], block.as_v4f[10][1], block.as_v4f[11][1],
			block.as_v4f[12][1], block.as_v4f[13][1], block.as_v4f[14][1], block.as_v4f[15][1],
			&block.as_float[0][4], &block.as_float[8][4], 16);
		if (column_count > 8) {
			v4f_ifft16_real(
				block.as_v4f[ 0][2], block.as_v4f[ 1][2], block.as_v4f[ 2][2], block.as_v4f[ 3][2],
				block.as_v4f[ 4][2], block.as_v4f[ 5][2], block.as_v4f[ 6][2], block.as_v4f[ 7][2],
				block.as_v4f[ 8][2], block.as_v4f[ 9][2], block.as_v4f[10][2], block.as_v4f[11][2],
				block.as_v4f[12][2], block.as_v4f[13][2], block.as_v4f[14][2], block.as_v4f[15][2],
				&block.as_float[0][8], &block.as_float[8][8], 16);
			if (column_count > 12) {
				v4f_ifft16_real(
					block.as_v4f[ 0][3], block.as_v4f[ 1][3], block.as_v4f[ 2][3], block.as_v4f[ 3][3],
					block.as_v4f[ 4][3], block.as_v4f[ 5][3], block.as_v4f[ 6][3], block.as_v4f[ 7][3],
					block.as_v4f[ 8][3], block.as_v4f[ 9][3], block.as_v4f[10][3], block.as_v4f[11][3],
					block.as_v4f[12][3], block.as_v4f[13][3], block.as_v4f[14][3], block.as_v4f[15][3],
					&block.as_float[0][12], &block.as_float[8][12], 16);
			}
		}
	}

	const float bias = *bias_ptr;
	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block.as_float[row][column] + bias;
		}
	}
}
