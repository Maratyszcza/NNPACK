#pragma once

#include <stdbool.h>


#include <nnpack/simd.h>


static inline void winograd_f6k3_input_transform(
	const v4f input0[restrict static 1],
	const v4f input1[restrict static 1],
	const v4f input2[restrict static 1],
	const v4f input3[restrict static 1],
	const v4f input4[restrict static 1],
	const v4f input5[restrict static 1],
	const v4f input6[restrict static 1],
	const v4f input7[restrict static 1],
	v4f transform0[restrict static 1],
	v4f transform1[restrict static 1],
	v4f transform2[restrict static 1],
	v4f transform3[restrict static 1],
	v4f transform4[restrict static 1],
	v4f transform5[restrict static 1],
	v4f transform6[restrict static 1],
	v4f transform7[restrict static 1])
{
	const v4f d0 = *input0;
	const v4f d1 = *input1;
	const v4f d2 = *input2;
	const v4f d3 = *input3;
	const v4f d4 = *input4;
	const v4f d5 = *input5;
	const v4f d6 = *input6;
	const v4f d7 = *input7;

	const v4f const_0_25 = v4f_splat(0.25f);

	// Compute wd0 := d0 - d6
	v4f wd0 = d0 - d6;
	const v4f d4_sub_d2 = d4 - d2;
	// Compute wd7 := d7 - d1
	v4f wd7 = d7 - d1;
	const v4f d3_sub_d5 = d3 - d5;
	// Compute wd1 := d2 + d6
	v4f wd1 = d2 + d6;
	// Compute wd2 := d1 + d5
	v4f wd2 = d1 + d5;
	// Compute wd4 := d5 + 0.25 * d1
	v4f wd4 = d5 + const_0_25 * d1;
	// Compute wd5 := d6 - 5.0 * d4
	v4f wd5 = d6 - v4f_splat(5.0f) * d4;
	// Compute wd3 := d6 + 0.25 * d2
	v4f wd3 = d6 + const_0_25 * d2;
	// Compute wd6 := d1 + 0.25 * d5
	v4f wd6 = d1 + const_0_25 * d5;

	const v4f const_5_25 = v4f_splat(5.25f);
	// Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
	wd0 += const_5_25 * d4_sub_d2;
	// Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
	wd7 += const_5_25 * d3_sub_d5;

	const v4f const_4_25 = v4f_splat(4.25f);
	// Compute
	//   wd1 := (d6 + d2) - 4.25 * d4
	//   wd2 := (d1 + d5) - 4.25 * d3
	wd1 -= const_4_25 * d4;
	wd2 -= const_4_25 * d3;

	const v4f const_1_25 = v4f_splat(1.25f);
	// Compute
	//   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
	//   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
	//   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
	//   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
	wd3 -= const_1_25 * d4;
	const v4f d3_times_1_25 = d3 * const_1_25;
	wd5 += v4f_splat(4.0) * d2;
	wd4 -= d3_times_1_25;
	wd6 -= d3_times_1_25;

	const v4f const_2 = v4f_splat(2.0f);
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

static inline void winograd_f6k3_kernel_transform(
	const v4f kernel0[restrict static 1],
	const v4f kernel1[restrict static 1],
	const v4f kernel2[restrict static 1],
	v4f transform0[restrict static 1],
	v4f transform1[restrict static 1],
	v4f transform2[restrict static 1],
	v4f transform3[restrict static 1],
	v4f transform4[restrict static 1],
	v4f transform5[restrict static 1],
	v4f transform6[restrict static 1],
	v4f transform7[restrict static 1],
	bool rescale_coefficients)
{
	const v4f g0 = *kernel0;
	const v4f g1 = *kernel1;
	const v4f g2 = *kernel2;

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
	const v4f const_4 = v4f_splat(4.0f);
	v4f w2 = g0 + g2;
	v4f w4 = g0 + const_4 * g2;
	v4f w6 = g2 + const_4 * g0;

	/*
	 * Compute
	 *   w1 = (g0 + g2) + g1
	 *   w2 = (g0 + g2) - g1
	 *   w3 = (g0 + 4 * g2) + 2 * g1
	 *   w4 = (g0 + 4 * g2) - 2 * g1
	 *   w5 = (g2 + 4 * g0) + 2 * g1
	 *   w6 = (g2 + 4 * g0) - 2 * g1
	 */
	const v4f two_g1 = g1 * v4f_splat(2.0f);
	v4f w1 = w2 + g1;
	w2 = w2 - g1;
	v4f w3 = w4 + two_g1;
	w4 = w4 - two_g1;
	v4f w5 = w6 + two_g1;
	w6 = w6 - two_g1;

	if (rescale_coefficients) {
		const v4f minus_2_over_9 = v4f_splat(-0x1.C71C72p-3f);
		w1 *= minus_2_over_9;
		w2 *= minus_2_over_9;

		const v4f rcp_90 = v4f_splat( 0x1.6C16C2p-7f);
		w3 *= rcp_90;
		w4 *= rcp_90;

		const v4f rcp_180 = v4f_splat( 0x1.6C16C2p-8f);
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

static inline void winograd_f6k3_output_transform(
	const v4f transform0[restrict static 1],
	const v4f transform1[restrict static 1],
	const v4f transform2[restrict static 1],
	const v4f transform3[restrict static 1],
	const v4f transform4[restrict static 1],
	const v4f transform5[restrict static 1],
	const v4f transform6[restrict static 1],
	const v4f transform7[restrict static 1],
	v4f output0[restrict static 1],
	v4f output1[restrict static 1],
	v4f output2[restrict static 1],
	v4f output3[restrict static 1],
	v4f output4[restrict static 1],
	v4f output5[restrict static 1])
{
	const v4f m0 = *transform0;
	const v4f m1 = *transform1;
	const v4f m2 = *transform2;
	const v4f m3 = *transform3;
	const v4f m4 = *transform4;
	const v4f m5 = *transform5;
	const v4f m6 = *transform6;
	const v4f m7 = *transform7;

	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 *
	 */

	const v4f m1_add_m2 = m1 + m2;
	const v4f m1_sub_m2 = m1 - m2;
	const v4f m3_add_m4 = m3 + m4;
	const v4f m3_sub_m4 = m3 - m4;
	const v4f m5_add_m6 = m5 + m6;
	const v4f m5_sub_m6 = m5 - m6;

	v4f s0 = m0 + m1_add_m2;
	v4f s5 = m7 + m1_sub_m2;

	const v4f const_16 = v4f_splat(16.0f);
	v4f s1 = m1_sub_m2 + const_16 * m5_sub_m6;
	v4f s4 = m1_add_m2 + const_16 * m3_add_m4;

	const v4f const_8 = v4f_splat(8.0f);
	v4f s2 = m1_add_m2 + const_8 * m5_add_m6;
	v4f s3 = m1_sub_m2 + const_8 * m3_sub_m4;

	const v4f const_32 = v4f_splat(32.0f);
	s0 += const_32 * m5_add_m6;
	s5 += const_32 * m3_sub_m4;

	s0 += m3_add_m4;
	s5 += m5_sub_m6;

	const v4f const_2 = v4f_splat(2.0f);
	s1 += m3_sub_m4 * const_2;
	s4 += m5_add_m6 * const_2;

	const v4f const_4 = v4f_splat(4.0f);
	s2 += m3_add_m4 * const_4;
	s3 += m5_sub_m6 * const_4;

	*output0 = s0;
	*output1 = s1;
	*output2 = s2;
	*output3 = s3;
	*output4 = s4;
	*output5 = s5;
}
