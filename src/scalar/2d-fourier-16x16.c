#include <stdint.h>
#include <stddef.h>

#include <scalar/fft/real.h>
#include <scalar/fft/soa.h>
#include <scalar/fft/dualreal.h>

#include <nnpack/utils.h>
#include <nnpack/activations.h>


#define BLOCK_SIZE 16


void nnp_fft16x16_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	const uint32_t simd_width = 1;
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];
	if (column_offset != 0) {
		for (uint32_t row = 0; row < BLOCK_SIZE; row++) {
			for (uint32_t column = 0; column < column_offset; column++) {
				block[row][column] = 0.0f;
			}
		}
	}

	const uint32_t column_end = column_offset + column_count;
	if (column_end != BLOCK_SIZE) {
		for (uint32_t row = 0; row < BLOCK_SIZE; row++) {
			for (uint32_t column = column_end; column < BLOCK_SIZE; column++) {
				block[row][column] = 0.0f;
			}
		}
	}

	const float *restrict row0 = data;
	const float *restrict row8 = data + doz(BLOCK_SIZE / 2, row_offset) * data_stride;
	float* restrict output = &block[0][column_offset];
	for (uint32_t column = column_offset; column < column_end; column++) {
		scalar_fft16_real(row0, row8, data_stride,
			row_offset, row_count,
			&block[0][column], BLOCK_SIZE);

		row0 += 1;
		row8 += 1;
		output += 1;
	}

	{
		float x0, y0, x1r, y1r, x2r, y2r, x3r, y3r, x4r, y4r, x5r, y5r, x6r, y6r, x7r, y7r;
		float x8, y8, x1i, y1i, x2i, y2i, x3i, y3i, x4i, y4i, x5i, y5i, x6i, y6i, x7i, y7i;
		scalar_fft16_dualreal(
			&block[0][0],
			&x0, &y0, &x1r, &y1r, &x2r, &y2r, &x3r, &y3r, &x4r, &y4r, &x5r, &y5r, &x6r, &y6r, &x7r, &y7r,
			&x8, &y8, &x1i, &y1i, &x2i, &y2i, &x3i, &y3i, &x4i, &y4i, &x5i, &y5i, &x6i, &y6i, &x7i, &y7i);
		transform[0] = x0;
		transform[1] = x8;
		transform += transform_stride;
		transform[0] = y0;
		transform[1] = y8;
		transform += transform_stride;	
		transform[0] = x1r;
		transform[1] = x1i;
		transform += transform_stride;	
		transform[0] = y1r;
		transform[1] = y1i;
		transform += transform_stride;	
		transform[0] = x2r;
		transform[1] = x2i;
		transform += transform_stride;	
		transform[0] = y2r;
		transform[1] = y2i;
		transform += transform_stride;	
		transform[0] = x3r;
		transform[1] = x3i;
		transform += transform_stride;	
		transform[0] = y3r;
		transform[1] = y3i;
		transform += transform_stride;	
		transform[0] = x4r;
		transform[1] = x4i;
		transform += transform_stride;	
		transform[0] = y4r;
		transform[1] = y4i;
		transform += transform_stride;	
		transform[0] = x5r;
		transform[1] = x5i;
		transform += transform_stride;	
		transform[0] = y5r;
		transform[1] = y5i;
		transform += transform_stride;	
		transform[0] = x6r;
		transform[1] = x6i;
		transform += transform_stride;	
		transform[0] = y6r;
		transform[1] = y6i;
		transform += transform_stride;	
		transform[0] = x7r;
		transform[1] = x7i;
		transform += transform_stride;	
		transform[0] = y7r;
		transform[1] = y7i;
		transform += transform_stride;	
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		float f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r, f13r, f14r, f15r;
		float f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i, f8i, f9i, f10i, f11i, f12i, f13i, f14i, f15i;
		scalar_fft16_soa(
			&block[row][0],
			&f0r, &f1r, &f2r, &f3r, &f4r, &f5r, &f6r, &f7r, &f8r, &f9r, &f10r, &f11r, &f12r, &f13r, &f14r, &f15r,
			&f0i, &f1i, &f2i, &f3i, &f4i, &f5i, &f6i, &f7i, &f8i, &f9i, &f10i, &f11i, &f12i, &f13i, &f14i, &f15i);
		transform[0] = f0r;
		transform[1] = f0i;
		transform += transform_stride;	
		transform[0] = f1r;
		transform[1] = f1i;
		transform += transform_stride;	
		transform[0] = f2r;
		transform[1] = f2i;
		transform += transform_stride;	
		transform[0] = f3r;
		transform[1] = f3i;
		transform += transform_stride;	
		transform[0] = f4r;
		transform[1] = f4i;
		transform += transform_stride;	
		transform[0] = f5r;
		transform[1] = f5i;
		transform += transform_stride;	
		transform[0] = f6r;
		transform[1] = f6i;
		transform += transform_stride;	
		transform[0] = f7r;
		transform[1] = f7i;
		transform += transform_stride;	
		transform[0] = f8r;
		transform[1] = f8i;
		transform += transform_stride;	
		transform[0] = f9r;
		transform[1] = f9i;
		transform += transform_stride;	
		transform[0] = f10r;
		transform[1] = f10i;
		transform += transform_stride;	
		transform[0] = f11r;
		transform[1] = f11i;
		transform += transform_stride;	
		transform[0] = f12r;
		transform[1] = f12i;
		transform += transform_stride;	
		transform[0] = f13r;
		transform[1] = f13i;
		transform += transform_stride;	
		transform[0] = f14r;
		transform[1] = f14i;
		transform += transform_stride;	
		transform[0] = f15r;
		transform[1] = f15i;
		transform += transform_stride;	
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft16x16_with_offset__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];
	{
		const float x0 = transform[0];
		const float x8 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y8 = transform[1];
		transform += transform_stride;
		const float x1r = transform[0];
		const float x1i = transform[1];
		transform += transform_stride;
		const float y1r = transform[0];
		const float y1i = transform[1];
		transform += transform_stride;
		const float x2r = transform[0];
		const float x2i = transform[1];
		transform += transform_stride;
		const float y2r = transform[0];
		const float y2i = transform[1];
		transform += transform_stride;
		const float x3r = transform[0];
		const float x3i = transform[1];
		transform += transform_stride;
		const float y3r = transform[0];
		const float y3i = transform[1];
		transform += transform_stride;
		const float x4r = transform[0];
		const float x4i = transform[1];
		transform += transform_stride;
		const float y4r = transform[0];
		const float y4i = transform[1];
		transform += transform_stride;
		const float x5r = transform[0];
		const float x5i = transform[1];
		transform += transform_stride;
		const float y5r = transform[0];
		const float y5i = transform[1];
		transform += transform_stride;
		const float x6r = transform[0];
		const float x6i = transform[1];
		transform += transform_stride;
		const float y6r = transform[0];
		const float y6i = transform[1];
		transform += transform_stride;
		const float x7r = transform[0];
		const float x7i = transform[1];
		transform += transform_stride;
		const float y7r = transform[0];
		const float y7i = transform[1];
		transform += transform_stride;
		scalar_ifft16_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r, x4r, y4r, x5r, y5r, x6r, y6r, x7r, y7r,
			x8, y8, x1i, y1i, x2i, y2i, x3i, y3i, x4i, y4i, x5i, y5i, x6i, y6i, x7i, y7i,
			&block[0][0]);
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		const float f0r = transform[0];
		const float f0i = transform[1];
		transform += transform_stride;
		const float f1r = transform[0];
		const float f1i = transform[1];
		transform += transform_stride;
		const float f2r = transform[0];
		const float f2i = transform[1];
		transform += transform_stride;
		const float f3r = transform[0];
		const float f3i = transform[1];
		transform += transform_stride;
		const float f4r = transform[0];
		const float f4i = transform[1];
		transform += transform_stride;
		const float f5r = transform[0];
		const float f5i = transform[1];
		transform += transform_stride;
		const float f6r = transform[0];
		const float f6i = transform[1];
		transform += transform_stride;
		const float f7r = transform[0];
		const float f7i = transform[1];
		transform += transform_stride;
		const float f8r = transform[0];
		const float f8i = transform[1];
		transform += transform_stride;
		const float f9r = transform[0];
		const float f9i = transform[1];
		transform += transform_stride;
		const float f10r = transform[0];
		const float f10i = transform[1];
		transform += transform_stride;
		const float f11r = transform[0];
		const float f11i = transform[1];
		transform += transform_stride;
		const float f12r = transform[0];
		const float f12i = transform[1];
		transform += transform_stride;
		const float f13r = transform[0];
		const float f13i = transform[1];
		transform += transform_stride;
		const float f14r = transform[0];
		const float f14i = transform[1];
		transform += transform_stride;
		const float f15r = transform[0];
		const float f15i = transform[1];
		transform += transform_stride;
		scalar_ifft16_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r, f13r, f14r, f15r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i, f8i, f9i, f10i, f11i, f12i, f13i, f14i, f15i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[ 0][column];
		const float f8  = block[ 1][column];
		const float f1r = block[ 2][column];
		const float f1i = block[ 3][column];
		const float f2r = block[ 4][column];
		const float f2i = block[ 5][column];
		const float f3r = block[ 6][column];
		const float f3i = block[ 7][column];
		const float f4r = block[ 8][column];
		const float f4i = block[ 9][column];
		const float f5r = block[10][column];
		const float f5i = block[11][column];
		const float f6r = block[12][column];
		const float f6i = block[13][column];
		const float f7r = block[14][column];
		const float f7i = block[15][column];
		scalar_ifft16_real(
			f0, f8, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i, f6r, f6i, f7r, f7i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row_offset + row][column_offset + column];
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft16x16_with_bias__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];

	const float bias_value = *bias;
	{
		const float x0 = transform[0] + bias_value * 256.0f;
		const float x8 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y8 = transform[1];
		transform += transform_stride;
		const float x1r = transform[0];
		const float x1i = transform[1];
		transform += transform_stride;
		const float y1r = transform[0];
		const float y1i = transform[1];
		transform += transform_stride;
		const float x2r = transform[0];
		const float x2i = transform[1];
		transform += transform_stride;
		const float y2r = transform[0];
		const float y2i = transform[1];
		transform += transform_stride;
		const float x3r = transform[0];
		const float x3i = transform[1];
		transform += transform_stride;
		const float y3r = transform[0];
		const float y3i = transform[1];
		transform += transform_stride;
		const float x4r = transform[0];
		const float x4i = transform[1];
		transform += transform_stride;
		const float y4r = transform[0];
		const float y4i = transform[1];
		transform += transform_stride;
		const float x5r = transform[0];
		const float x5i = transform[1];
		transform += transform_stride;
		const float y5r = transform[0];
		const float y5i = transform[1];
		transform += transform_stride;
		const float x6r = transform[0];
		const float x6i = transform[1];
		transform += transform_stride;
		const float y6r = transform[0];
		const float y6i = transform[1];
		transform += transform_stride;
		const float x7r = transform[0];
		const float x7i = transform[1];
		transform += transform_stride;
		const float y7r = transform[0];
		const float y7i = transform[1];
		transform += transform_stride;
		scalar_ifft16_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r, x4r, y4r, x5r, y5r, x6r, y6r, x7r, y7r,
			x8, y8, x1i, y1i, x2i, y2i, x3i, y3i, x4i, y4i, x5i, y5i, x6i, y6i, x7i, y7i,
			&block[0][0]);
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		const float f0r = transform[0];
		const float f0i = transform[1];
		transform += transform_stride;
		const float f1r = transform[0];
		const float f1i = transform[1];
		transform += transform_stride;
		const float f2r = transform[0];
		const float f2i = transform[1];
		transform += transform_stride;
		const float f3r = transform[0];
		const float f3i = transform[1];
		transform += transform_stride;
		const float f4r = transform[0];
		const float f4i = transform[1];
		transform += transform_stride;
		const float f5r = transform[0];
		const float f5i = transform[1];
		transform += transform_stride;
		const float f6r = transform[0];
		const float f6i = transform[1];
		transform += transform_stride;
		const float f7r = transform[0];
		const float f7i = transform[1];
		transform += transform_stride;
		const float f8r = transform[0];
		const float f8i = transform[1];
		transform += transform_stride;
		const float f9r = transform[0];
		const float f9i = transform[1];
		transform += transform_stride;
		const float f10r = transform[0];
		const float f10i = transform[1];
		transform += transform_stride;
		const float f11r = transform[0];
		const float f11i = transform[1];
		transform += transform_stride;
		const float f12r = transform[0];
		const float f12i = transform[1];
		transform += transform_stride;
		const float f13r = transform[0];
		const float f13i = transform[1];
		transform += transform_stride;
		const float f14r = transform[0];
		const float f14i = transform[1];
		transform += transform_stride;
		const float f15r = transform[0];
		const float f15i = transform[1];
		transform += transform_stride;
		scalar_ifft16_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r, f13r, f14r, f15r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i, f8i, f9i, f10i, f11i, f12i, f13i, f14i, f15i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[ 0][column];
		const float f8  = block[ 1][column];
		const float f1r = block[ 2][column];
		const float f1i = block[ 3][column];
		const float f2r = block[ 4][column];
		const float f2i = block[ 5][column];
		const float f3r = block[ 6][column];
		const float f3i = block[ 7][column];
		const float f4r = block[ 8][column];
		const float f4i = block[ 9][column];
		const float f5r = block[10][column];
		const float f5i = block[11][column];
		const float f6r = block[12][column];
		const float f6i = block[13][column];
		const float f7r = block[14][column];
		const float f7i = block[15][column];
		scalar_ifft16_real(
			f0, f8, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i, f6r, f6i, f7r, f7i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row][column];
		}
	}
}

void nnp_ifft16x16_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];

	const float bias_value = *bias;
	{
		const float x0 = transform[0] + bias_value * 256.0f;
		const float x8 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y8 = transform[1];
		transform += transform_stride;
		const float x1r = transform[0];
		const float x1i = transform[1];
		transform += transform_stride;
		const float y1r = transform[0];
		const float y1i = transform[1];
		transform += transform_stride;
		const float x2r = transform[0];
		const float x2i = transform[1];
		transform += transform_stride;
		const float y2r = transform[0];
		const float y2i = transform[1];
		transform += transform_stride;
		const float x3r = transform[0];
		const float x3i = transform[1];
		transform += transform_stride;
		const float y3r = transform[0];
		const float y3i = transform[1];
		transform += transform_stride;
		const float x4r = transform[0];
		const float x4i = transform[1];
		transform += transform_stride;
		const float y4r = transform[0];
		const float y4i = transform[1];
		transform += transform_stride;
		const float x5r = transform[0];
		const float x5i = transform[1];
		transform += transform_stride;
		const float y5r = transform[0];
		const float y5i = transform[1];
		transform += transform_stride;
		const float x6r = transform[0];
		const float x6i = transform[1];
		transform += transform_stride;
		const float y6r = transform[0];
		const float y6i = transform[1];
		transform += transform_stride;
		const float x7r = transform[0];
		const float x7i = transform[1];
		transform += transform_stride;
		const float y7r = transform[0];
		const float y7i = transform[1];
		transform += transform_stride;
		scalar_ifft16_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r, x4r, y4r, x5r, y5r, x6r, y6r, x7r, y7r,
			x8, y8, x1i, y1i, x2i, y2i, x3i, y3i, x4i, y4i, x5i, y5i, x6i, y6i, x7i, y7i,
			&block[0][0]);
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		const float f0r = transform[0];
		const float f0i = transform[1];
		transform += transform_stride;
		const float f1r = transform[0];
		const float f1i = transform[1];
		transform += transform_stride;
		const float f2r = transform[0];
		const float f2i = transform[1];
		transform += transform_stride;
		const float f3r = transform[0];
		const float f3i = transform[1];
		transform += transform_stride;
		const float f4r = transform[0];
		const float f4i = transform[1];
		transform += transform_stride;
		const float f5r = transform[0];
		const float f5i = transform[1];
		transform += transform_stride;
		const float f6r = transform[0];
		const float f6i = transform[1];
		transform += transform_stride;
		const float f7r = transform[0];
		const float f7i = transform[1];
		transform += transform_stride;
		const float f8r = transform[0];
		const float f8i = transform[1];
		transform += transform_stride;
		const float f9r = transform[0];
		const float f9i = transform[1];
		transform += transform_stride;
		const float f10r = transform[0];
		const float f10i = transform[1];
		transform += transform_stride;
		const float f11r = transform[0];
		const float f11i = transform[1];
		transform += transform_stride;
		const float f12r = transform[0];
		const float f12i = transform[1];
		transform += transform_stride;
		const float f13r = transform[0];
		const float f13i = transform[1];
		transform += transform_stride;
		const float f14r = transform[0];
		const float f14i = transform[1];
		transform += transform_stride;
		const float f15r = transform[0];
		const float f15i = transform[1];
		transform += transform_stride;
		scalar_ifft16_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r, f13r, f14r, f15r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i, f8i, f9i, f10i, f11i, f12i, f13i, f14i, f15i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[ 0][column];
		const float f8  = block[ 1][column];
		const float f1r = block[ 2][column];
		const float f1i = block[ 3][column];
		const float f2r = block[ 4][column];
		const float f2i = block[ 5][column];
		const float f3r = block[ 6][column];
		const float f3i = block[ 7][column];
		const float f4r = block[ 8][column];
		const float f4i = block[ 9][column];
		const float f5r = block[10][column];
		const float f5i = block[11][column];
		const float f6r = block[12][column];
		const float f6i = block[13][column];
		const float f7r = block[14][column];
		const float f7i = block[15][column];
		scalar_ifft16_real(
			f0, f8, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i, f6r, f6i, f7r, f7i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[row][column], 0.0f);
		}
	}
}
