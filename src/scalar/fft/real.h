#pragma once

#include <nnpack/fft-constants.h>
#include <scalar/fft/aos.h>


static inline void scalar_fft8_real(
	const float t0[restrict static 4],
	const float t4[restrict static 4],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	float w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	scalar_fft4_aos(t0, t4, stride_t, row_offset, row_count,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i);

	const float half = 0.5f;
	const float g1r = half * (w1r + w3r);
	const float g1i = half * (w1i - w3i);
	const float two_h1r = w1i + w3i;
	const float two_h1i = w3r - w1r;

	const float sqrt2_over_4 = SQRT2_OVER_4;
	const float h1_plus  = sqrt2_over_4 * (two_h1i + two_h1r);
	const float h1_minus = sqrt2_over_4 * (two_h1i - two_h1r);

	const float f0 = w0r + w0i;
	const float f4 = w0r - w0i;
	const float f1r = g1r + h1_plus;
	const float f1i = h1_minus + g1i;
	const float f2r =  w2r;
	const float f2i = -w2i;
	const float f3r = g1r - h1_plus;
	const float f3i = h1_minus - g1i;

	/* Store outputs */
	f[0 * stride_f] = f0;
	f[1 * stride_f] = f4;
	f[2 * stride_f] = f1r;
	f[3 * stride_f] = f1i;
	f[4 * stride_f] = f2r;
	f[5 * stride_f] = f2i;
	f[6 * stride_f] = f3r;
	f[7 * stride_f] = f3i;
}

static inline void scalar_fft16_real(
	const float t0[restrict static 8],
	const float t8[restrict static 8],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	float w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	scalar_fft8_aos(t0, t8, stride_t, row_offset, row_count,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i, &w4r, &w4i, &w5r, &w5i, &w6r, &w6i, &w7r, &w7i);

	const float half = 0.5f;
	const float g1r = half * (w1r + w7r);
	const float g1i = half * (w1i - w7i);
	const float g2r = half * (w2r + w6r);
	const float g2i = half * (w2i - w6i);
	const float g3r = half * (w3r + w5r);
	const float g3i = half * (w3i - w5i);

	const float two_h1r = w1i + w7i;
	const float two_h1i = w7r - w1r;
	const float two_h2r = w2i + w6i;
	const float two_h2i = w6r - w2r;
	const float two_h3r = w3i + w5i;
	const float two_h3i = w5r - w3r;

	const float sqrt2_over_4 = SQRT2_OVER_4;
	const float h2_plus  = sqrt2_over_4 * (two_h2i + two_h2r);
	const float h2_minus = sqrt2_over_4 * (two_h2i - two_h2r);

	const float half_cos_1pi_over_8 = COS_1PI_OVER_8 * 0.5f;
	const float half_cos_3pi_over_8 = COS_3PI_OVER_8 * 0.5f;

	const float f0  =  w0r + w0i;
	const float f8  =  w0r - w0i;
	const float f1r =  g1r + two_h1r * half_cos_1pi_over_8 + two_h1i * half_cos_3pi_over_8;
	const float f1i =  g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;
	const float f2r =  g2r + h2_plus;
	const float f2i =  h2_minus + g2i;
	const float f3r =  g3r + two_h3r * half_cos_3pi_over_8 + two_h3i * half_cos_1pi_over_8;
	const float f3i =  g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const float f4r =  w4r;
	const float f4i = -w4i;
	const float f5r =  g3r - two_h3r * half_cos_3pi_over_8 - two_h3i * half_cos_1pi_over_8;
	const float f5i = -g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const float f6r =  g2r - h2_plus;
	const float f6i =  h2_minus - g2i;
	const float f7r =  g1r - two_h1r * half_cos_1pi_over_8 - two_h1i * half_cos_3pi_over_8;
	const float f7i = -g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;

	/* Store outputs */
	f[ 0 * stride_f] = f0;
	f[ 1 * stride_f] = f8;
	f[ 2 * stride_f] = f1r;
	f[ 3 * stride_f] = f1i;
	f[ 4 * stride_f] = f2r;
	f[ 5 * stride_f] = f2i;
	f[ 6 * stride_f] = f3r;
	f[ 7 * stride_f] = f3i;
	f[ 8 * stride_f] = f4r;
	f[ 9 * stride_f] = f4i;
	f[10 * stride_f] = f5r;
	f[11 * stride_f] = f5i;
	f[12 * stride_f] = f6r;
	f[13 * stride_f] = f6i;
	f[14 * stride_f] = f7r;
	f[15 * stride_f] = f7i;
}

static inline void scalar_ifft8_real(
	float f0, float f4, float f1r, float f1i, float f2r, float f2i, float f3r, float f3i,
	float t0[restrict static 4],
	float t4[restrict static 4],
	size_t stride_t)
{
	/* Load inputs and scale */
	const float scale = 0.5f;
	f0  *= scale;
	f4  *= scale;
	f1r *= scale;
	f1i *= scale;
	f3r *= scale;
	f3i *= scale;

	const float w0r =  f0 + f4;
	const float w0i =  f0 - f4;
	const float w2r =  f2r;
	const float w2i = -f2i;

	const float g1r = f1r + f3r;
	const float g1i = f1i - f3i;

	const float h1r = f1r - f3r;
	const float h1i = f1i + f3i;

	const float h1_plus  = h1r + h1i;
	const float h1_minus = h1r - h1i;

	const float sqrt2_over2 = SQRT2_OVER_2;
	const float w1r =  g1r - sqrt2_over2 * h1_plus;
	const float w1i =  g1i + sqrt2_over2 * h1_minus;
	const float w3r =  g1r + sqrt2_over2 * h1_plus;
	const float w3i = -g1i + sqrt2_over2 * h1_minus;

	scalar_ifft4_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i,
		t0, t4, stride_t);
}

static inline void scalar_ifft16_real(
	float f0,  float f8,  float f1r, float f1i, float f2r, float f2i, float f3r, float f3i,
	float f4r, float f4i, float f5r, float f5i, float f6r, float f6i, float f7r, float f7i,
	float t0[restrict static 8],
	float t8[restrict static 8],
	size_t stride_t)
{
	/* Load inputs and scale */
	const float scale = 0.5f;
	f0  *= scale;
	f8  *= scale;
	f1r *= scale;
	f1i *= scale;
	f2r *= scale;
	f2i *= scale;
	f3r *= scale;
	f3i *= scale;
	f5r *= scale;
	f5i *= scale;
	f6r *= scale;
	f6i *= scale;
	f7r *= scale;
	f7i *= scale;

	const float w0r =  f0 + f8;
	const float w0i =  f0 - f8;
	const float w4r =  f4r;
	const float w4i = -f4i;

	const float g2r = f2r + f6r;
	const float g2i = f2i - f6i;

	const float h2r = f2r - f6r;
	const float h2i = f2i + f6i;

	const float h2_plus  = h2r + h2i;
	const float h2_minus = h2r - h2i;

	const float sqrt2_over2 = SQRT2_OVER_2;
	const float w2r =  g2r - sqrt2_over2 * h2_plus;
	const float w2i =  g2i + sqrt2_over2 * h2_minus;
	const float w6r =  g2r + sqrt2_over2 * h2_plus;
	const float w6i = -g2i + sqrt2_over2 * h2_minus;

	const float g1r = f1r + f7r;
	const float g1i = f1i - f7i;
	const float g3r = f3r + f5r;
	const float g3i = f3i - f5i;

	const float h1r = f1r - f7r;
	const float h1i = f1i + f7i;
	const float h3r = f3r - f5r;
	const float h3i = f3i + f5i;

	const float cos_1pi_over_8 = COS_1PI_OVER_8;
	const float cos_3pi_over_8 = COS_3PI_OVER_8;
	const float w1r =  g1r - h1i * cos_1pi_over_8 - h1r * cos_3pi_over_8;
	const float w1i =  g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;
	const float w7r =  g1r + h1i * cos_1pi_over_8 + h1r * cos_3pi_over_8;
	const float w7i = -g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;

	const float w3r =  g3r - h3i * cos_3pi_over_8 - h3r * cos_1pi_over_8;
	const float w3i =  g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;
	const float w5r =  g3r + h3i * cos_3pi_over_8 + h3r * cos_1pi_over_8;
	const float w5i = -g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;

	scalar_ifft8_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
		t0, t8, stride_t);
}
