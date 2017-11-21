#include <stdint.h>
#include <stddef.h>

#include <nnpack/arm_neon.h>
#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <neon/winograd/f6x6k3x3.h>
#include <neon/transpose.h>


void nnp_iwt8x8_3x3_fp16_with_offset__neonhp(
	const float data[restrict static 1],
	void *restrict transform,
	size_t data_stride,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	NNP_SIMD_ALIGN float32x4_t wd[8][2];
	if NNP_LIKELY(row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0) {
		// Fast path where we can directly load `data` into `wd`.
		for (size_t i = 0; i < 8; i++) {
			for (size_t j = 0; j < 2; j++) {
				wd[i][j] = vld1q_f32(&data[i * data_stride + j * 4]);
			}
		}
	} else {
		NNP_SIMD_ALIGN float block[8][8] = {{ 0 }};
		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				block[row_offset + i][column_offset + j] = data[i * data_stride + j];
			}
		}

		for (size_t i = 0; i < 8; i++) {
			for (size_t j = 0; j < 2; j++) {
				wd[i][j] = vld1q_f32(&block[i][j * 4]);
			}
		}
	}

	for (size_t col = 0; col < 2; col++) {
		winograd_f6k3_input_transform_inplace(&wd[0][col],
			&wd[1][col],
			&wd[2][col],
			&wd[3][col],
			&wd[4][col],
			&wd[5][col],
			&wd[6][col],
			&wd[7][col]);
		neon_transpose4x4_inplace_f32(&wd[0][col], &wd[1][col], &wd[2][col], &wd[3][col]);
		neon_transpose4x4_inplace_f32(&wd[4][col], &wd[5][col], &wd[6][col], &wd[7][col]);
	}

	vswapq_f32(&wd[4][0], &wd[0][1]);
	vswapq_f32(&wd[5][0], &wd[1][1]);
	vswapq_f32(&wd[6][0], &wd[2][1]);
	vswapq_f32(&wd[7][0], &wd[3][1]);

	for (size_t col = 0; col < 2; col++) {
		winograd_f6k3_input_transform_inplace(&wd[0][col],
			&wd[1][col],
			&wd[2][col],
			&wd[3][col],
			&wd[4][col],
			&wd[5][col],
			&wd[6][col],
			&wd[7][col]);
	}

	for (size_t col = 0; col < 2; col++) {
		for (size_t row = 0; row < 8; row++) {
			vst1q_f16_f32(transform, wd[row][col]);
			transform += transform_stride;
		}
	}
}

void nnp_kwt8x8_3x3_fp16__neonhp(
	const float g[restrict static 9],
	void *restrict transform,
	size_t stride_g,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	const float32x4_t g0 = vld1q_f32(g);
	const float32x4_t g1 = vld1q_f32(g + 3);
	// g2[3] is junk
	const float32x4_t g2 = vextq_f32(vld1q_f32(g + 5), vld1q_f32(g + 5), 1);
	NNP_SIMD_ALIGN float32x4_t w[8];
	winograd_f6k3_kernel_transform(g0, g1, g2,
		&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7],
		true /* rescale coefficients */);
	neon_transpose4x4_inplace_f32(&w[0], &w[1], &w[2], &w[3]);
	neon_transpose4x4_inplace_f32(&w[4], &w[5], &w[6], &w[7]);

	NNP_SIMD_ALIGN float32x4_t wg[8][2];
	winograd_f6k3_kernel_transform(w[0], w[1], w[2],
		&wg[0][0], &wg[1][0], &wg[2][0], &wg[3][0],
		&wg[4][0], &wg[5][0], &wg[6][0], &wg[7][0],
		true /* rescale coefficients */);
	winograd_f6k3_kernel_transform(w[4], w[5], w[6],
		&wg[0][1], &wg[1][1], &wg[2][1], &wg[3][1],
		&wg[4][1], &wg[5][1], &wg[6][1], &wg[7][1],
		true /* rescale coefficients */);

	for (size_t col = 0; col < 2; col++) {
		for (size_t row = 0; row < 8; row++) {
			vst1q_f16_f32(transform, wg[row][col]);
			transform += transform_stride;
		}
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_owt8x8_3x3_fp16__neonhp(
	const void *restrict transform,
	float output[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	NNP_SIMD_ALIGN float32x4_t s[8][2];
	for (size_t col = 0; col < 2; col++) {
		s[0][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[1][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[2][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[3][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[4][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[5][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[6][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[7][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		winograd_f6k3_output_transform_inplace(
			&s[0][col], &s[1][col], &s[2][col], &s[3][col],
			&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
		neon_transpose4x4_inplace_f32(&s[0][col], &s[1][col], &s[2][col], &s[3][col]);
		neon_transpose4x4_inplace_f32(&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
	}

	vswapq_f32(&s[4][0], &s[0][1]);
	vswapq_f32(&s[5][0], &s[1][1]);
	vswapq_f32(&s[6][0], &s[2][1]);
	vswapq_f32(&s[7][0], &s[3][1]);

	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		winograd_f6k3_output_transform_inplace(
			&s[0][0], &s[1][0], &s[2][0], &s[3][0],
			&s[4][0], &s[5][0], &s[6][0], &s[7][0]);
		for (size_t i = 0; i < 6; i++) {
			vst1q_f32(&output[i * output_stride], s[i][0]);
		}
		winograd_f6k3_output_transform_inplace(
			&s[0][1], &s[1][1], &s[2][1], &s[3][1],
			&s[4][1], &s[5][1], &s[6][1], &s[7][1]);
		for (size_t i = 0; i < 6; i++) {
			vst1_f32(&output[i * output_stride + 4], vget_low_f32(s[i][1]));
		}
	} else {
		NNP_SIMD_ALIGN float block[6][8];
		for (size_t col = 0; col < 2; col++) {
			winograd_f6k3_output_transform_inplace(
				&s[0][col], &s[1][col], &s[2][col], &s[3][col],
				&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
			vst1q_f32(&block[0][col * 4], s[0][col]);
			vst1q_f32(&block[1][col * 4], s[1][col]);
			vst1q_f32(&block[2][col * 4], s[2][col]);
			vst1q_f32(&block[3][col * 4], s[3][col]);
			vst1q_f32(&block[4][col * 4], s[4][col]);
			vst1q_f32(&block[5][col * 4], s[5][col]);
		}
		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_owt8x8_3x3_fp16_with_bias__neonhp(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
	NNP_SIMD_ALIGN float32x4_t s[8][2];
	for (size_t col = 0; col < 2; col++) {
		s[0][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		// Only difference in the with_bias vs non with_bias case.
		if (col == 0) {
			s[1][col] = vld1q_f32_f16(transform) + vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
		} else {
			s[1][col] = vld1q_f32_f16(transform);
		}
		transform += transform_stride;
		s[2][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[3][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[4][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[5][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[6][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[7][col] = vld1q_f32_f16(transform);
		transform += transform_stride;

		winograd_f6k3_output_transform_inplace(
			&s[0][col], &s[1][col], &s[2][col], &s[3][col],
			&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
		neon_transpose4x4_inplace_f32(&s[0][col], &s[1][col], &s[2][col], &s[3][col]);
		neon_transpose4x4_inplace_f32(&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
	}

	vswapq_f32(&s[4][0], &s[0][1]);
	vswapq_f32(&s[5][0], &s[1][1]);
	vswapq_f32(&s[6][0], &s[2][1]);
	vswapq_f32(&s[7][0], &s[3][1]);
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		winograd_f6k3_output_transform_inplace(
			&s[0][0], &s[1][0], &s[2][0], &s[3][0],
			&s[4][0], &s[5][0], &s[6][0], &s[7][0]);
		for (size_t i = 0; i < row_count; i++) {
			vst1q_f32(&output[i * output_stride], s[i][0]);
		}
		winograd_f6k3_output_transform_inplace(
			&s[0][1], &s[1][1], &s[2][1], &s[3][1],
			&s[4][1], &s[5][1], &s[6][1], &s[7][1]);
		for (size_t i = 0; i < row_count; i++) {
			vst1_f32(&output[i * output_stride + 4], vget_low_f32(s[i][1]));
		}
	} else {
		NNP_SIMD_ALIGN float block[6][8];
		for (size_t col = 0; col < 2; col++) {
			winograd_f6k3_output_transform_inplace(
				&s[0][col], &s[1][col], &s[2][col], &s[3][col],
				&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
			vst1q_f32(&block[0][col * 4], s[0][col]);
			vst1q_f32(&block[1][col * 4], s[1][col]);
			vst1q_f32(&block[2][col * 4], s[2][col]);
			vst1q_f32(&block[3][col * 4], s[3][col]);
			vst1q_f32(&block[4][col * 4], s[4][col]);
			vst1q_f32(&block[5][col * 4], s[5][col]);
		}
		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}

void nnp_owt8x8_3x3_fp16_with_bias_with_relu__neonhp(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
	NNP_SIMD_ALIGN float32x4_t s[8][2];
	for (size_t col = 0; col < 2; col++) {
		s[0][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		// Only difference in the with_bias vs non with_bias case.
		if (col == 0) {
			s[1][col] = vld1q_f32_f16(transform) + vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
		} else {
			s[1][col] = vld1q_f32_f16(transform);
		}
		transform += transform_stride;
		s[2][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[3][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[4][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[5][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[6][col] = vld1q_f32_f16(transform);
		transform += transform_stride;
		s[7][col] = vld1q_f32_f16(transform);
		transform += transform_stride;

		winograd_f6k3_output_transform_inplace(
			&s[0][col], &s[1][col], &s[2][col], &s[3][col],
			&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
		neon_transpose4x4_inplace_f32(&s[0][col], &s[1][col], &s[2][col], &s[3][col]);
		neon_transpose4x4_inplace_f32(&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
	}

	vswapq_f32(&s[4][0], &s[0][1]);
	vswapq_f32(&s[5][0], &s[1][1]);
	vswapq_f32(&s[6][0], &s[2][1]);
	vswapq_f32(&s[7][0], &s[3][1]);
	const float32x4_t zero = vdupq_n_f32(0.0f);
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		winograd_f6k3_output_transform_inplace(
			&s[0][0], &s[1][0], &s[2][0], &s[3][0],
			&s[4][0], &s[5][0], &s[6][0], &s[7][0]);
		for (size_t i = 0; i < row_count; i++) {
			vst1q_f32(&output[i * output_stride], neon_relu_f32(s[i][0], zero));
		}
		winograd_f6k3_output_transform_inplace(
			&s[0][1], &s[1][1], &s[2][1], &s[3][1],
			&s[4][1], &s[5][1], &s[6][1], &s[7][1]);
		for (size_t i = 0; i < row_count; i++) {
			vst1_f32(&output[i * output_stride + 4], vget_low_f32(neon_relu_f32(s[i][1], zero)));
		}
	} else {
		NNP_SIMD_ALIGN float block[6][8];
		for (size_t col = 0; col < 2; col++) {
			winograd_f6k3_output_transform_inplace(
				&s[0][col], &s[1][col], &s[2][col], &s[3][col],
				&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
			vst1q_f32(&block[0][col * 4], neon_relu_f32(s[0][col], zero));
			vst1q_f32(&block[1][col * 4], neon_relu_f32(s[1][col], zero));
			vst1q_f32(&block[2][col * 4], neon_relu_f32(s[2][col], zero));
			vst1q_f32(&block[3][col * 4], neon_relu_f32(s[3][col], zero));
			vst1q_f32(&block[4][col * 4], neon_relu_f32(s[4][col], zero));
			vst1q_f32(&block[5][col * 4], neon_relu_f32(s[5][col], zero));
		}
		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}
