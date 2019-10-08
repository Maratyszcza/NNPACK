#include <stdint.h>
#include <stddef.h>

#include <nnpack/arm_neon.h>
#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <neon/winograd/f6x6k3x3.h>
#include <neon/transpose.h>


void nnp_iwt8x8_3x3_with_offset__neon(
	const float data[restrict static 1],
	void* transform,
	size_t data_stride,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	NNP_SIMD_ALIGN float32x4_t wd[16];
	if NNP_LIKELY(row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0) {
		// Fast path where we can directly load `data` into `wd`.
		for (size_t col = 0; col < 2; col++) {
			winograd_f6k3_input_transform(
				vld1q_f32(&data[0 * data_stride + col * 4]),
				vld1q_f32(&data[1 * data_stride + col * 4]),
				vld1q_f32(&data[2 * data_stride + col * 4]),
				vld1q_f32(&data[3 * data_stride + col * 4]),
				vld1q_f32(&data[4 * data_stride + col * 4]),
				vld1q_f32(&data[5 * data_stride + col * 4]),
				vld1q_f32(&data[6 * data_stride + col * 4]),
				vld1q_f32(&data[7 * data_stride + col * 4]),
				&wd[0 + col * 4], &wd[1 + col * 4], &wd[ 2 + col * 4], &wd[ 3 + col * 4],
				&wd[8 + col * 4], &wd[9 + col * 4], &wd[10 + col * 4], &wd[11 + col * 4]);
		}
	} else {
		NNP_SIMD_ALIGN float block[8][8];
		{
			const float32x4_t vzero = vmovq_n_f32(0.0f);
			for (float *block_ptr = &block[0][0], *block_end = &block[8][0]; block_ptr != block_end; block_ptr += 4) {
				vst1q_f32(block_ptr, vzero);
			}
		}
		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				block[row_offset + i][column_offset + j] = data[i * data_stride + j];
			}
		}

		for (size_t col = 0; col < 2; col++) {
			winograd_f6k3_input_transform(
				vld1q_f32(&block[0][col * 4]),
				vld1q_f32(&block[1][col * 4]),
				vld1q_f32(&block[2][col * 4]),
				vld1q_f32(&block[3][col * 4]),
				vld1q_f32(&block[4][col * 4]),
				vld1q_f32(&block[5][col * 4]),
				vld1q_f32(&block[6][col * 4]),
				vld1q_f32(&block[7][col * 4]),
				&wd[0 + col * 4], &wd[1 + col * 4], &wd[ 2 + col * 4], &wd[ 3 + col * 4],
				&wd[8 + col * 4], &wd[9 + col * 4], &wd[10 + col * 4], &wd[11 + col * 4]);
		}
	}

	for (size_t col = 0; col < 2; col++) {
		float32x4_t vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;
		float32x4x4_t vin0123 = vld4q_f32((const float*) &wd[0 + col * 8]);
		float32x4x4_t vin4567 = vld4q_f32((const float*) &wd[4 + col * 8]);
		winograd_f6k3_input_transform(
			vin0123.val[0], vin0123.val[1], vin0123.val[2], vin0123.val[3],
			vin4567.val[0], vin4567.val[1], vin4567.val[2], vin4567.val[3],
			&vout0, &vout1, &vout2, &vout3, &vout4, &vout5, &vout6, &vout7);

		vst1q_f32(transform, vout0);
		transform += transform_stride;
		vst1q_f32(transform, vout1);
		transform += transform_stride;
		vst1q_f32(transform, vout2);
		transform += transform_stride;
		vst1q_f32(transform, vout3);
		transform += transform_stride;
		vst1q_f32(transform, vout4);
		transform += transform_stride;
		vst1q_f32(transform, vout5);
		transform += transform_stride;
		vst1q_f32(transform, vout6);
		transform += transform_stride;
		vst1q_f32(transform, vout7);
		transform += transform_stride;
	}
}

void nnp_kwt8x8_3x3__neon(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	transform_stride /= sizeof(float);

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
			vst1q_f32(transform, wg[row][col]);
			transform += transform_stride;
		}
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_kwt8x8_3Rx3R__neon(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	const float32x4_t g5678 = vld1q_f32(g + 5);
	const float32x4_t g2345 = vld1q_f32(g + 2);
	const float32x4_t g0123 = vld1q_f32(g);

	/* g0 = { g[8], g[7], g[6], g[6] }; */
	const float32x4_t g0 = vcombine_f32(vrev64_f32(vld1_f32(&g[7])), vld1_dup_f32(&g[6]));
	/* g1 = { g[5], g[4], g[3], g[3] }; */
	const float32x4_t g1 = vcombine_f32(vrev64_f32(vld1_f32(&g[4])), vld1_dup_f32(&g[3]));
	/* g2 = { g[2], g[1], g[0], g[0] }; */
	const float32x4_t g2 = vcombine_f32(vrev64_f32(vld1_f32(&g[1])), vld1_dup_f32(&g[0]));

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
			vst1q_f32(transform, wg[row][col]);
			transform += transform_stride;
		}
	}
}

void nnp_owt8x8_3x3__neon(
	const void *restrict transform,
	float output[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		float* output_col0123 = output;
		vst1q_f32(output_col0123, qout0); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout1); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout2); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout3); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout4); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout5);

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		float* output_col45 = output + 4;
		vst1_f32(output_col45, dout0); output_col45 += output_stride;
		vst1_f32(output_col45, dout1); output_col45 += output_stride;
		vst1_f32(output_col45, dout2); output_col45 += output_stride;
		vst1_f32(output_col45, dout3); output_col45 += output_stride;
		vst1_f32(output_col45, dout4); output_col45 += output_stride;
		vst1_f32(output_col45, dout5);
	} else {
		NNP_SIMD_ALIGN float block[6][8];

		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		vst1q_f32(&block[0][0], qout0);
		vst1q_f32(&block[1][0], qout1);
		vst1q_f32(&block[2][0], qout2);
		vst1q_f32(&block[3][0], qout3);
		vst1q_f32(&block[4][0], qout4);
		vst1q_f32(&block[5][0], qout5);

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		vst1_f32(&block[0][4], dout0);
		vst1_f32(&block[1][4], dout1);
		vst1_f32(&block[2][4], dout2);
		vst1_f32(&block[3][4], dout3);
		vst1_f32(&block[4][4], dout4);
		vst1_f32(&block[5][4], dout5);

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_owt8x8_3x3_with_bias__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
		m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
		vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		float* output_col0123 = output;
		vst1q_f32(output_col0123, qout0); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout1); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout2); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout3); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout4); output_col0123 += output_stride;
		vst1q_f32(output_col0123, qout5);

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		float* output_col45 = output + 4;
		vst1_f32(output_col45, dout0); output_col45 += output_stride;
		vst1_f32(output_col45, dout1); output_col45 += output_stride;
		vst1_f32(output_col45, dout2); output_col45 += output_stride;
		vst1_f32(output_col45, dout3); output_col45 += output_stride;
		vst1_f32(output_col45, dout4); output_col45 += output_stride;
		vst1_f32(output_col45, dout5);
	} else {
		NNP_SIMD_ALIGN float block[6][8];

		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		vst1q_f32(&block[0][0], qout0);
		vst1q_f32(&block[1][0], qout1);
		vst1q_f32(&block[2][0], qout2);
		vst1q_f32(&block[3][0], qout3);
		vst1q_f32(&block[4][0], qout4);
		vst1q_f32(&block[5][0], qout5);

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		vst1_f32(&block[0][4], dout0);
		vst1_f32(&block[1][4], dout1);
		vst1_f32(&block[2][4], dout2);
		vst1_f32(&block[3][4], dout3);
		vst1_f32(&block[4][4], dout4);
		vst1_f32(&block[5][4], dout5);

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}

void nnp_owt8x8_3x3s2_with_bias__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride,
	size_t output_stride,
	uint32_t row_count,
	uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
		m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
		vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	NNP_SIMD_ALIGN float block[3][8];

	float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
	winograd_f6k3_output_transformq(
		qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
		qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
		&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
	vst1q_f32(&block[0][0], qout0);
	vst1q_f32(&block[1][0], qout2);
	vst1q_f32(&block[2][0], qout4);

	float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din67 = vld2_f32(read_ptr);
	float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
	winograd_f6k3_output_transform(
		din01.val[0], din01.val[1], din23.val[0], din23.val[1],
		din45.val[0], din45.val[1], din67.val[0], din67.val[1],
		&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
	vst1_f32(&block[0][4], dout0);
	vst1_f32(&block[1][4], dout2);
	vst1_f32(&block[2][4], dout4);

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j * 2];
		}
	}
}

void nnp_owt8x8_3x3_with_bias_with_relu__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
		m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
		vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	if NNP_LIKELY(row_count == 6 && column_count == 6 && output_stride >= 6) {
		// Fast path to reuse `s` array and write directly into `output`.
		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		float* output_col0123 = output;
		const float32x4_t qzero = vmovq_n_f32(0.0f);
		vst1q_f32(output_col0123, neon_reluq_f32(qout0, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout1, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout2, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout3, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout4, qzero)); output_col0123 += output_stride;
		vst1q_f32(output_col0123, neon_reluq_f32(qout5, qzero));

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		float* output_col45 = output + 4;
		const float32x2_t dzero = vmov_n_f32(0.0f);
		vst1_f32(output_col45, neon_relu_f32(dout0, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout1, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout2, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout3, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout4, dzero)); output_col45 += output_stride;
		vst1_f32(output_col45, neon_relu_f32(dout5, dzero));
	} else {
		NNP_SIMD_ALIGN float block[6][8];

		float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
		float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
		winograd_f6k3_output_transformq(
			qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
			qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
			&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
		const float32x4_t qzero = vmovq_n_f32(0.0f);
		vst1q_f32(&block[0][0], neon_reluq_f32(qout0, qzero));
		vst1q_f32(&block[1][0], neon_reluq_f32(qout1, qzero));
		vst1q_f32(&block[2][0], neon_reluq_f32(qout2, qzero));
		vst1q_f32(&block[3][0], neon_reluq_f32(qout3, qzero));
		vst1q_f32(&block[4][0], neon_reluq_f32(qout4, qzero));
		vst1q_f32(&block[5][0], neon_reluq_f32(qout5, qzero));

		float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
		float32x2x2_t din67 = vld2_f32(read_ptr);
		float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
		winograd_f6k3_output_transform(
			din01.val[0], din01.val[1], din23.val[0], din23.val[1],
			din45.val[0], din45.val[1], din67.val[0], din67.val[1],
			&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
		const float32x2_t dzero = vmov_n_f32(0.0f);
		vst1_f32(&block[0][4], neon_relu_f32(dout0, dzero));
		vst1_f32(&block[1][4], neon_relu_f32(dout1, dzero));
		vst1_f32(&block[2][4], neon_relu_f32(dout2, dzero));
		vst1_f32(&block[3][4], neon_relu_f32(dout3, dzero));
		vst1_f32(&block[4][4], neon_relu_f32(dout4, dzero));
		vst1_f32(&block[5][4], neon_relu_f32(dout5, dzero));

		for (size_t i = 0; i < row_count; i++) {
			for (size_t j = 0; j < column_count; j++) {
				output[i * output_stride + j] = block[i][j];
			}
		}
	}
}

void nnp_owt8x8_3x3s2_with_bias_with_relu__neon(
	const void *restrict transform,
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	NNP_SIMD_ALIGN float buffer[8 * 6];
	float*restrict qbuffer = buffer;
	float*restrict dbuffer = buffer + 32;
	float32x2_t vbias = vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vld1_dup_f32(bias)), 32));
	for (uint32_t col = 0; col < 2; col++) {
		const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
		/* The only difference in the with_bias vs non with_bias case. */
		m1 = vcombine_f32(vadd_f32(vget_low_f32(m1), vbias), vget_high_f32(m1));
		vbias = vmov_n_f32(0.0f);
		const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m6 = vld1q_f32(transform); transform += transform_stride;
		const float32x4_t m7 = vld1q_f32(transform); transform += transform_stride;
		float32x4_t o0, o1, o2, o3, o4, o5;
		winograd_f6k3_output_transformq(
			m0, m1, m2, m3, m4, m5, m6, m7,
			&o0, &o1, &o2, &o3, &o4, &o5);
		vst1q_f32(qbuffer, o0); qbuffer += 4;
		vst1q_f32(qbuffer, o1); qbuffer += 4;
		vst1q_f32(qbuffer, o2); qbuffer += 4;
		vst1q_f32(qbuffer, o3); qbuffer += 4;
		vst1_f32(dbuffer, vget_low_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_low_f32(o5)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o4)); dbuffer += 2;
		vst1_f32(dbuffer, vget_high_f32(o5)); dbuffer += 2;
	}

	const float*restrict read_ptr = buffer;
	NNP_SIMD_ALIGN float block[3][8];

	float32x4x4_t qin0123 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4x4_t qin4567 = vld4q_f32(read_ptr); read_ptr += 16;
	float32x4_t qout0, qout1, qout2, qout3, qout4, qout5;
	winograd_f6k3_output_transformq(
		qin0123.val[0], qin0123.val[1], qin0123.val[2], qin0123.val[3],
		qin4567.val[0], qin4567.val[1], qin4567.val[2], qin4567.val[3],
		&qout0, &qout1, &qout2, &qout3, &qout4, &qout5);
	const float32x4_t qzero = vmovq_n_f32(0.0f);
	vst1q_f32(&block[0][0], neon_reluq_f32(qout0, qzero));
	vst1q_f32(&block[1][0], neon_reluq_f32(qout2, qzero));
	vst1q_f32(&block[2][0], neon_reluq_f32(qout4, qzero));

	float32x2x2_t din01 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din23 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din45 = vld2_f32(read_ptr); read_ptr += 4;
	float32x2x2_t din67 = vld2_f32(read_ptr);
	float32x2_t dout0, dout1, dout2, dout3, dout4, dout5;
	winograd_f6k3_output_transform(
		din01.val[0], din01.val[1], din23.val[0], din23.val[1],
		din45.val[0], din45.val[1], din67.val[0], din67.val[1],
		&dout0, &dout1, &dout2, &dout3, &dout4, &dout5);
	const float32x2_t dzero = vmov_n_f32(0.0f);
	vst1_f32(&block[0][4], neon_relu_f32(dout0, dzero));
	vst1_f32(&block[1][4], neon_relu_f32(dout2, dzero));
	vst1_f32(&block[2][4], neon_relu_f32(dout4, dzero));

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j * 2];
		}
	}
}
