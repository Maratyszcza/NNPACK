#include <stdint.h>
#include <stddef.h>

#include <scalar/fft/real.h>
#include <scalar/fft/soa.h>
#include <scalar/fft/dualreal.h>

#include <nnpack/utils.h>
#include <nnpack/activations.h>


#define BLOCK_SIZE 8


void nnp_fft8x8_with_offset__scalar(
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
	const float *restrict row4 = data + doz(BLOCK_SIZE / 2, row_offset) * data_stride;
	float* restrict output = &block[0][column_offset];
	for (uint32_t column = column_offset; column < column_end; column++) {
		scalar_fft8_real(row0, row4, data_stride,
			row_offset, row_count,
			&block[0][column], BLOCK_SIZE);

		row0 += 1;
		row4 += 1;
		output += 1;
	}

	{
		float x0, y0, x1r, y1r, x2r, y2r, x3r, y3r;
		float x4, y4, x1i, y1i, x2i, y2i, x3i, y3i;
		scalar_fft8_dualreal(
			&block[0][0],
			&x0, &y0, &x1r, &y1r, &x2r, &y2r, &x3r, &y3r,
			&x4, &y4, &x1i, &y1i, &x2i, &y2i, &x3i, &y3i);
		transform[0] = x0;
		transform[1] = x4;
		transform += transform_stride;	
		transform[0] = y0;
		transform[1] = y4;
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
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		float f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r;
		float f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i;
		scalar_fft8_soa(
			&block[row][0],
			&f0r, &f1r, &f2r, &f3r, &f4r, &f5r, &f6r, &f7r,
			&f0i, &f1i, &f2i, &f3i, &f4i, &f5i, &f6i, &f7i);
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
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft8x8_with_offset__scalar(
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
		const float x4 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y4 = transform[1];
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
		scalar_ifft8_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
			x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
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
		scalar_ifft8_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[0][column];
		const float f4  = block[1][column];
		const float f1r = block[2][column];
		const float f1i = block[3][column];
		const float f2r = block[4][column];
		const float f2i = block[5][column];
		const float f3r = block[6][column];
		const float f3i = block[7][column];
		scalar_ifft8_real(
			f0, f4, f1r, f1i, f2r, f2i, f3r, f3i,
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

void nnp_ifft8x8_with_bias__scalar(
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
		const float x0 = transform[0] + bias_value * 64.0f;
		const float x4 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y4 = transform[1];
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
		scalar_ifft8_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
			x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
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
		scalar_ifft8_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[0][column];
		const float f4  = block[1][column];
		const float f1r = block[2][column];
		const float f1i = block[3][column];
		const float f2r = block[4][column];
		const float f2i = block[5][column];
		const float f3r = block[6][column];
		const float f3i = block[7][column];
		scalar_ifft8_real(
			f0, f4, f1r, f1i, f2r, f2i, f3r, f3i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row][column];
		}
	}
}

void nnp_ifft8x8_with_bias_with_relu__scalar(
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
		const float x0 = transform[0] + bias_value * 64.0f;
		const float x4 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y4 = transform[1];
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
		scalar_ifft8_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
			x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
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
		scalar_ifft8_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[0][column];
		const float f4  = block[1][column];
		const float f1r = block[2][column];
		const float f1i = block[3][column];
		const float f2r = block[4][column];
		const float f2i = block[5][column];
		const float f3r = block[6][column];
		const float f3i = block[7][column];
		scalar_ifft8_real(
			f0, f4, f1r, f1i, f2r, f2i, f3r, f3i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[row][column], 0.0f);
		}
	}
}
