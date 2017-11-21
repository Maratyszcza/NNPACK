#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <nnpack/macros.h>
#include <nnpack/activations.h>

#include <scalar/winograd/f6x6k3x3.h>


#define BLOCK_SIZE 8
#define KERNEL_SIZE 3
#define OUTPUT_SIZE (BLOCK_SIZE - KERNEL_SIZE + 1)


void nnp_iwt8x8_3x3_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];
	if (row_offset != 0) {
		memset(&block[0][0], 0, row_offset * BLOCK_SIZE * sizeof(float));
	}
	const uint32_t row_end = row_offset + row_count;
	if (row_end != BLOCK_SIZE) {
		memset(&block[row_end][0], 0, (BLOCK_SIZE - row_end) * BLOCK_SIZE * sizeof(float));
	}

	for (uint32_t row = row_offset; row < row_end; row++) {
		float d0, d1, d2, d3, d4, d5, d6, d7;
		d0 = d1 = d2 = d3 = d4 = d5 = d6 = d7 = 0.0f;

		const float *restrict row_data = data;
		uint32_t remaining_column_count = column_count;
		switch (column_offset) {
			case 0:
				d0 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 1:
				d1 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 2:
				d2 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 3:
				d3 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 4:
				d4 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 5:
				d5 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 6:
				d6 = *row_data++;
				if (--remaining_column_count == 0) {
					break;
				}
			case 7:
				d7 = *row_data;
				break;
			default:
				NNP_UNREACHABLE;
		}
		winograd_f6k3_input_transform(d0, d1, d2, d3, d4, d5, d6, d7,
			&block[row][0], &block[row][1], &block[row][2], &block[row][3],
			&block[row][4], &block[row][5], &block[row][6], &block[row][7]);

		data += data_stride;
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		float wd0, wd1, wd2, wd3, wd4, wd5, wd6, wd7;
		winograd_f6k3_input_transform(
			block[0][column], block[1][column], block[2][column], block[3][column],
			block[4][column], block[5][column], block[6][column], block[7][column],
			&wd0, &wd1, &wd2, &wd3, &wd4, &wd5, &wd6, &wd7);
		*transform = wd0;
		transform += transform_stride;
		*transform = wd1;
		transform += transform_stride;
		*transform = wd2;
		transform += transform_stride;
		*transform = wd3;
		transform += transform_stride;
		*transform = wd4;
		transform += transform_stride;
		*transform = wd5;
		transform += transform_stride;
		*transform = wd6;
		transform += transform_stride;
		*transform = wd7;
		transform += transform_stride;
	}
}

void nnp_kwt8x8_3x3__scalar(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	float block[KERNEL_SIZE][BLOCK_SIZE];

	for (uint32_t row = 0; row < KERNEL_SIZE; row++) {
		float w0, w1, w2, w3, w4, w5, w6, w7;
		winograd_f6k3_kernel_transform(
			g[0],
			g[1],
			g[2],
			&block[row][0], &block[row][1], &block[row][2], &block[row][3],
			&block[row][4], &block[row][5], &block[row][6], &block[row][7],
			true);
		g += KERNEL_SIZE;
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		float w0, w1, w2, w3, w4, w5, w6, w7;
		winograd_f6k3_kernel_transform(
			block[0][column], block[1][column], block[2][column],
			&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
			true);
		*transform = w0;
		transform += transform_stride;
		*transform = w1;
		transform += transform_stride;
		*transform = w2;
		transform += transform_stride;
		*transform = w3;
		transform += transform_stride;
		*transform = w4;
		transform += transform_stride;
		*transform = w5;
		transform += transform_stride;
		*transform = w6;
		transform += transform_stride;
		*transform = w7;
		transform += transform_stride;
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_kwt8x8_3Rx3R__scalar(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);
	g += KERNEL_SIZE * (KERNEL_SIZE - 1);

	float block[KERNEL_SIZE][BLOCK_SIZE];

	for (uint32_t row = 0; row < KERNEL_SIZE; row++) {
		float w0, w1, w2, w3, w4, w5, w6, w7;
		winograd_f6k3_kernel_transform(
			g[2],
			g[1],
			g[0],
			&block[row][0], &block[row][1], &block[row][2], &block[row][3],
			&block[row][4], &block[row][5], &block[row][6], &block[row][7],
			true);
		g -= KERNEL_SIZE;
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		float w0, w1, w2, w3, w4, w5, w6, w7;
		winograd_f6k3_kernel_transform(
			block[0][column], block[1][column], block[2][column],
			&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
			true);
		*transform = w0;
		transform += transform_stride;
		*transform = w1;
		transform += transform_stride;
		*transform = w2;
		transform += transform_stride;
		*transform = w3;
		transform += transform_stride;
		*transform = w4;
		transform += transform_stride;
		*transform = w5;
		transform += transform_stride;
		*transform = w6;
		transform += transform_stride;
		*transform = w7;
		transform += transform_stride;
	}
}

void nnp_owt8x8_3x3__scalar(
	const float transform[restrict static 1],
	float output[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	float block[OUTPUT_SIZE][BLOCK_SIZE];
	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float m0 = *transform;
		transform += transform_stride;
		const float m1 = *transform;
		transform += transform_stride;
		const float m2 = *transform;
		transform += transform_stride;
		const float m3 = *transform;
		transform += transform_stride;
		const float m4 = *transform;
		transform += transform_stride;
		const float m5 = *transform;
		transform += transform_stride;
		const float m6 = *transform;
		transform += transform_stride;
		const float m7 = *transform;
		transform += transform_stride;

		winograd_f6k3_output_transform(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&block[0][column], &block[1][column], &block[2][column],
			&block[3][column], &block[4][column], &block[5][column]);
	}

	const uint32_t row_end = row_offset + row_count;
	for (uint32_t row = row_offset; row < row_end; row++) {
		float s0, s1, s2, s3, s4, s5;
		winograd_f6k3_output_transform(
			block[row][0], block[row][1], block[row][2], block[row][3],
			block[row][4], block[row][5], block[row][6], block[row][7],
			&s0, &s1, &s2, &s3, &s4, &s5);
		float *restrict row_output = output + (row - row_offset) * output_stride;
		uint32_t remaining_column_count = column_count;
		switch (column_offset) {
			case 0:
				*row_output++ = s0;
				if (--remaining_column_count == 0) {
					break;
				}
			case 1:
				*row_output++ = s1;
				if (--remaining_column_count == 0) {
					break;
				}
			case 2:
				*row_output++ = s2;
				if (--remaining_column_count == 0) {
					break;
				}
			case 3:
				*row_output++ = s3;
				if (--remaining_column_count == 0) {
					break;
				}
			case 4:
				*row_output++ = s4;
				if (--remaining_column_count == 0) {
					break;
				}
			case 5:
				*row_output = s5;
				break;
			default:
				NNP_UNREACHABLE;
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_owt8x8_3x3_with_bias__scalar(
	const float transform[restrict static 1],
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);
	const uint32_t row_offset = 0;
	const uint32_t column_offset = 0;

	float block[OUTPUT_SIZE][BLOCK_SIZE];
	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float m0 = *transform;
		transform += transform_stride;
		float m1 = *transform;
		transform += transform_stride;
		const float m2 = *transform;
		transform += transform_stride;
		const float m3 = *transform;
		transform += transform_stride;
		const float m4 = *transform;
		transform += transform_stride;
		const float m5 = *transform;
		transform += transform_stride;
		const float m6 = *transform;
		transform += transform_stride;
		const float m7 = *transform;
		transform += transform_stride;

		if (column == 1) {
			const float bias_value = *bias;
			m1 += bias_value;
		}

		winograd_f6k3_output_transform(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&block[0][column], &block[1][column], &block[2][column],
			&block[3][column], &block[4][column], &block[5][column]);
	}

	const uint32_t row_end = row_offset + row_count;
	for (uint32_t row = row_offset; row < row_end; row++) {
		float s0, s1, s2, s3, s4, s5;
		winograd_f6k3_output_transform(
			block[row][0], block[row][1], block[row][2], block[row][3],
			block[row][4], block[row][5], block[row][6], block[row][7],
			&s0, &s1, &s2, &s3, &s4, &s5);
		float *restrict row_output = output + (row - row_offset) * output_stride;
		uint32_t remaining_column_count = column_count;
		switch (column_offset) {
			case 0:
				*row_output++ = s0;
				if (--remaining_column_count == 0) {
					break;
				}
			case 1:
				*row_output++ = s1;
				if (--remaining_column_count == 0) {
					break;
				}
			case 2:
				*row_output++ = s2;
				if (--remaining_column_count == 0) {
					break;
				}
			case 3:
				*row_output++ = s3;
				if (--remaining_column_count == 0) {
					break;
				}
			case 4:
				*row_output++ = s4;
				if (--remaining_column_count == 0) {
					break;
				}
			case 5:
				*row_output = s5;
				break;
			default:
				NNP_UNREACHABLE;
		}
	}
}

void nnp_owt8x8_3x3_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);
	const uint32_t row_offset = 0;
	const uint32_t column_offset = 0;

	float block[OUTPUT_SIZE][BLOCK_SIZE];
	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float m0 = *transform;
		transform += transform_stride;
		float m1 = *transform;
		transform += transform_stride;
		const float m2 = *transform;
		transform += transform_stride;
		const float m3 = *transform;
		transform += transform_stride;
		const float m4 = *transform;
		transform += transform_stride;
		const float m5 = *transform;
		transform += transform_stride;
		const float m6 = *transform;
		transform += transform_stride;
		const float m7 = *transform;
		transform += transform_stride;

		if (column == 1) {
			const float bias_value = *bias;
			m1 += bias_value;
		}

		winograd_f6k3_output_transform(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&block[0][column], &block[1][column], &block[2][column],
			&block[3][column], &block[4][column], &block[5][column]);
	}

	const uint32_t row_end = row_offset + row_count;
	for (uint32_t row = row_offset; row < row_end; row++) {
		float s0, s1, s2, s3, s4, s5;
		winograd_f6k3_output_transform(
			block[row][0], block[row][1], block[row][2], block[row][3],
			block[row][4], block[row][5], block[row][6], block[row][7],
			&s0, &s1, &s2, &s3, &s4, &s5);
		float *restrict row_output = output + (row - row_offset) * output_stride;
		uint32_t remaining_column_count = column_count;
		switch (column_offset) {
			case 0:
				*row_output++ = relu(s0, 0.0f);
				if (--remaining_column_count == 0) {
					break;
				}
			case 1:
				*row_output++ = relu(s1, 0.0f);
				if (--remaining_column_count == 0) {
					break;
				}
			case 2:
				*row_output++ = relu(s2, 0.0f);
				if (--remaining_column_count == 0) {
					break;
				}
			case 3:
				*row_output++ = relu(s3, 0.0f);
				if (--remaining_column_count == 0) {
					break;
				}
			case 4:
				*row_output++ = relu(s4, 0.0f);
				if (--remaining_column_count == 0) {
					break;
				}
			case 5:
				*row_output = relu(s5, 0.0f);
				break;
			default:
				NNP_UNREACHABLE;
		}
	}
}
