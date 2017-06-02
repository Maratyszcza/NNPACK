#pragma once

#include <stdbool.h>

#include <nnpack/macros.h>


static NNP_INLINE void winograd_f6k3_input_transform(
	const float d0, const float d1, const float d2, const float d3, 
	const float d4, const float d5, const float d6, const float d7,
	float transform0[restrict static 1],
	float transform1[restrict static 1],
	float transform2[restrict static 1],
	float transform3[restrict static 1],
	float transform4[restrict static 1],
	float transform5[restrict static 1],
	float transform6[restrict static 1],
	float transform7[restrict static 1])
{
	const float const_0_25 = 0.25f;

	// Compute wd0 := d0 - d6
	float wd0 = d0 - d6;
	const float d4_sub_d2 = d4 - d2;
	// Compute wd7 := d7 - d1
	float wd7 = d7 - d1;
	const float d3_sub_d5 = d3 - d5;
	// Compute wd1 := d2 + d6
	float wd1 = d2 + d6;
	// Compute wd2 := d1 + d5
	float wd2 = d1 + d5;
	// Compute wd4 := d5 + 0.25 * d1
	float wd4 = d5 + const_0_25 * d1;
	// Compute wd5 := d6 - 5.0 * d4
	float wd5 = d6 - 5.0f * d4;
	// Compute wd3 := d6 + 0.25 * d2
	float wd3 = d6 + const_0_25 * d2;
	// Compute wd6 := d1 + 0.25 * d5
	float wd6 = d1 + const_0_25 * d5;

	const float const_5_25 = 5.25f;
	// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
	wd0 += const_5_25 * d4_sub_d2;
	// Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
	wd7 += const_5_25 * d3_sub_d5;

	const float const_4_25 = 4.25f;
	// Compute
	//   wd1 := (d6 + d2) - 4.25 * d4
	//   wd2 := (d1 + d5) - 4.25 * d3
	wd1 -= const_4_25 * d4;
	wd2 -= const_4_25 * d3;

	const float const_1_25 = 1.25f;
	// Compute
	//   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
	//   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
	//   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
	//   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
	wd3 -= const_1_25 * d4;
	const float d3_times_1_25 = d3 * const_1_25;
	wd5 += 4.0f * d2;
	wd4 -= d3_times_1_25;
	wd6 -= d3_times_1_25;

	const float const_2 = 2.0f;
	wd4 *= const_2;
	wd6 *= const_2;

	*transform0 = wd0;
	*transform1 = wd1 + wd2;
	*transform2 = wd1 - wd2;
	*transform3 = wd3 + wd4;
	*transform4 = wd3 - wd4;
	*transform5 = wd5 + wd6;
	*transform6 = wd5 - wd6;
	*transform7 = wd7;
}

static NNP_INLINE void winograd_f6k3_kernel_transform(
	const float g0, const float g1, const float g2,
	float transform0[restrict static 1],
	float transform1[restrict static 1],
	float transform2[restrict static 1],
	float transform3[restrict static 1],
	float transform4[restrict static 1],
	float transform5[restrict static 1],
	float transform6[restrict static 1],
	float transform7[restrict static 1],
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
	const float const_4 = 4.0f;
	float w2 = g0 + g2;
	float w4 = g0 + const_4 * g2;
	float w6 = g2 + const_4 * g0;

	/*
	 * Compute
	 *   w1 = (g0 + g2) + g1
	 *   w2 = (g0 + g2) - g1
	 *   w3 = (g0 + 4 * g2) + 2 * g1
	 *   w4 = (g0 + 4 * g2) - 2 * g1
	 *   w5 = (g2 + 4 * g0) + 2 * g1
	 *   w6 = (g2 + 4 * g0) - 2 * g1
	 */
	const float two_g1 = g1 * 2.0f;
	float w1 = w2 + g1;
	w2 = w2 - g1;
	float w3 = w4 + two_g1;
	w4 = w4 - two_g1;
	float w5 = w6 + two_g1;
	w6 = w6 - two_g1;

	if (rescale_coefficients) {
		const float minus_2_over_9 = -0x1.C71C72p-3f;
		w1 *= minus_2_over_9;
		w2 *= minus_2_over_9;

		const float rcp_90 = 0x1.6C16C2p-7f;
		w3 *= rcp_90;
		w4 *= rcp_90;

		const float rcp_180 = 0x1.6C16C2p-8f;
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

static NNP_INLINE void winograd_f6k3_output_transform(
	const float m0, const float m1, const float m2, const float m3, const float m4, const float m5, const float m6, const float m7,
	float output0[restrict static 1],
	float output1[restrict static 1],
	float output2[restrict static 1],
	float output3[restrict static 1],
	float output4[restrict static 1],
	float output5[restrict static 1])
{
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */

	const float m1_add_m2 = m1 + m2;
	const float m1_sub_m2 = m1 - m2;
	const float m3_add_m4 = m3 + m4;
	const float m3_sub_m4 = m3 - m4;
	const float m5_add_m6 = m5 + m6;
	const float m5_sub_m6 = m5 - m6;

	float s0 = m0 + m1_add_m2;
	float s5 = m7 + m1_sub_m2;

	const float const_16 = 16.0f;
	float s1 = m1_sub_m2 + const_16 * m5_sub_m6;
	float s4 = m1_add_m2 + const_16 * m3_add_m4;

	const float const_8 = 8.0f;
	float s2 = m1_add_m2 + const_8 * m5_add_m6;
	float s3 = m1_sub_m2 + const_8 * m3_sub_m4;

	const float const_32 = 32.0f;
	s0 += const_32 * m5_add_m6;
	s5 += const_32 * m3_sub_m4;

	s0 += m3_add_m4;
	s5 += m5_sub_m6;

	const float const_2 = 2.0f;
	s1 += m3_sub_m4 * const_2;
	s4 += m5_add_m6 * const_2;

	const float const_4 = 4.0f;
	s2 += m3_add_m4 * const_4;
	s3 += m5_sub_m6 * const_4;

	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
}
