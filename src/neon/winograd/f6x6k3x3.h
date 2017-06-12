#pragma once

#include <stdbool.h>

#include <nnpack/arm_neon.h>
#include <nnpack/macros.h>


static NNP_INLINE void winograd_f6k3_input_transform_inplace(
	float32x4_t d0[restrict static 1],
	float32x4_t d1[restrict static 1],
	float32x4_t d2[restrict static 1],
	float32x4_t d3[restrict static 1],
	float32x4_t d4[restrict static 1],
	float32x4_t d5[restrict static 1],
	float32x4_t d6[restrict static 1],
	float32x4_t d7[restrict static 1])
{
	const float32x4_t const_0_25 = vdupq_n_f32(0.25f);

	// Compute wd0 := d0 - d6
	// float32x4_t wd0 = *d0 - *d6;
	const float32x4_t d4_sub_d2 = *d4 - *d2;
	// Compute wd7 := d7 - d1
	float32x4_t wd7 = *d7 - *d1;
	const float32x4_t d3_sub_d5 = *d3 - *d5;
	// Compute wd1 := d2 + d6
	float32x4_t wd1 = *d2 + *d6;
	// Compute wd2 := d1 + d5
	float32x4_t wd2 = *d1 + *d5;
	// Compute wd4 := d5 + 0.25 * d1
	float32x4_t wd4 = vmuladdq_f32(*d5, const_0_25, *d1);
	// Compute wd5 := d6 - 5.0 * d4
	float32x4_t wd5 = vmulsubq_f32(*d6, vdupq_n_f32(5.0f), *d4);
	// Compute wd3 := d6 + 0.25 * d2
	float32x4_t wd3 = vmuladdq_f32(*d6, const_0_25, *d2);
	// Compute wd6 := d1 + 0.25 * d5
	float32x4_t wd6 = vmuladdq_f32(*d1, const_0_25, *d5);

	const float32x4_t const_5_25 = vdupq_n_f32(5.25f);
	// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
	*d0 = vmuladdq_f32(*d0 - *d6, const_5_25, d4_sub_d2);
	// Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
	*d7 = vmuladdq_f32(*d7 - *d1, const_5_25, d3_sub_d5);

	const float32x4_t const_4_25 = vdupq_n_f32(4.25f);
	// Compute
	//   wd1 := (d6 + d2) - 4.25 * d4
	//   wd2 := (d1 + d5) - 4.25 * d3
	wd1 = vmulsubq_f32(wd1, const_4_25, *d4);
	wd2 = vmulsubq_f32(wd2, const_4_25, *d3);

	const float32x4_t const_1_25 = vdupq_n_f32(1.25f);
	// Compute
	//   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
	//   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
	//   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
	//   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
	wd3 = vmulsubq_f32(wd3, const_1_25, *d4);
	const float32x4_t d3_times_1_25 = *d3 * const_1_25;
	wd5 = vmuladdq_f32(wd5, vdupq_n_f32(4.0), *d2);
	wd4 -= d3_times_1_25;
	wd6 -= d3_times_1_25;

	const float32x4_t const_2 = vdupq_n_f32(2.0f);
	wd4 *= const_2;
	wd6 *= const_2;

	// *d0 = wd0;
	*d1 = wd1 + wd2;
	*d2 = wd1 - wd2;
	*d3 = wd3 + wd4;
	*d4 = wd3 - wd4;
	*d5 = wd5 + wd6;
	*d6 = wd5 - wd6;
	//*d7 = wd7;
}

static NNP_INLINE void winograd_f6k3_kernel_transform(
	const float32x4_t g0, const float32x4_t g1, const float32x4_t g2,
	float32x4_t transform0[restrict static 1],
	float32x4_t transform1[restrict static 1],
	float32x4_t transform2[restrict static 1],
	float32x4_t transform3[restrict static 1],
	float32x4_t transform4[restrict static 1],
	float32x4_t transform5[restrict static 1],
	float32x4_t transform6[restrict static 1],
	float32x4_t transform7[restrict static 1],
	bool rescale_coefficients)
{
	/*
	 * w0 = g0
	 * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
	 * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
	 * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
	 * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
	 * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
	 * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
	 * w7 = g2
	 */

	/*
	 * Compute
	 *   w2 := g0 + g2
	 *   w4 := g0 + 4 * g2
	 *   w6 := g2 + 4 * g0
	 */
	const float32x4_t const_4 = vdupq_n_f32(4.0f);
	float32x4_t w2 = g0 + g2;
	float32x4_t w4 = vmuladdq_f32(g0, const_4, g2);
	float32x4_t w6 = vmuladdq_f32(g2, const_4, g0);

	/*
	 * Compute
	 *   w1 = (g0 + g2) + g1
	 *   w2 = (g0 + g2) - g1
	 *   w3 = (g0 + 4 * g2) + 2 * g1
	 *   w4 = (g0 + 4 * g2) - 2 * g1
	 *   w5 = (g2 + 4 * g0) + 2 * g1
	 *   w6 = (g2 + 4 * g0) - 2 * g1
	 */
	const float32x4_t two_g1 = g1 * vdupq_n_f32(2.0f);
	float32x4_t w1 = w2 + g1;
	w2 = w2 - g1;
	float32x4_t w3 = w4 + two_g1;
	w4 = w4 - two_g1;
	float32x4_t w5 = w6 + two_g1;
	w6 = w6 - two_g1;

	if (rescale_coefficients) {
		const float32x4_t minus_2_over_9 = vdupq_n_f32(-0x1.C71C72p-3f);
		w1 *= minus_2_over_9;
		w2 *= minus_2_over_9;

		const float32x4_t rcp_90 = vdupq_n_f32(0x1.6C16C2p-7f);
		w3 *= rcp_90;
		w4 *= rcp_90;

		const float32x4_t rcp_180 = vdupq_n_f32(0x1.6C16C2p-8f);
		w5 *= rcp_180;
		w6 *= rcp_180;
	}

	*transform0 = g0;
	*transform1 = w1;
	*transform2 = w2;
	*transform3 = w3;
	*transform4 = w4;
	*transform5 = w5;
	*transform6 = w6;
	*transform7 = g2;
}

static NNP_INLINE void winograd_f6k3_output_transform_inplace(
	float32x4_t m0[restrict static 1],
	float32x4_t m1[restrict static 1],
	float32x4_t m2[restrict static 1],
	float32x4_t m3[restrict static 1],
	float32x4_t m4[restrict static 1],
	float32x4_t m5[restrict static 1],
	float32x4_t m6[restrict static 1],
	float32x4_t m7[restrict static 1])
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	const float32x4_t m1_add_m2 = *m1 + *m2;
	const float32x4_t m1_sub_m2 = *m1 - *m2;
	const float32x4_t m3_add_m4 = *m3 + *m4;
	const float32x4_t m3_sub_m4 = *m3 - *m4;
	const float32x4_t m5_add_m6 = *m5 + *m6;
	const float32x4_t m5_sub_m6 = *m5 - *m6;

	// Finised with M[0-6] as **inputs** here.
	*m0 = *m0 + m1_add_m2;
	*m5 = *m7 + m1_sub_m2;
	// Finised with M[0-7] as **inputs** here.

	const float32x4_t const_16 = vdupq_n_f32(16.0f);
	*m1 = vmuladdq_f32(m1_sub_m2, const_16, m5_sub_m6);
	*m4 = vmuladdq_f32(m1_add_m2, const_16, m3_add_m4);

	const float32x4_t const_8 = vdupq_n_f32(8.0f);
	*m2 = vmuladdq_f32(m1_add_m2, const_8, m5_add_m6);
	*m3 = vmuladdq_f32(m1_sub_m2, const_8, m3_sub_m4);

	const float32x4_t const_32 = vdupq_n_f32(32.0f);
	*m0 = vmuladdq_f32(*m0, const_32, m5_add_m6);
	*m0 += m3_add_m4;

	*m5 = vmuladdq_f32(*m5, const_32, m3_sub_m4);
	*m5 += m5_sub_m6;

	const float32x4_t const_2 = vdupq_n_f32(2.0f);
	*m1 = vmuladdq_f32(*m1, m3_sub_m4, const_2);
	*m4 = vmuladdq_f32(*m4, m5_add_m6, const_2);

	const float32x4_t const_4 = vdupq_n_f32(4.0f);
	*m2 = vmuladdq_f32(*m2, m3_add_m4, const_4);
	*m3 = vmuladdq_f32(*m3, m5_sub_m6, const_4);
}
