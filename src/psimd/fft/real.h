#pragma once

#include <nnpack/simd.h>
#include <nnpack/fft-constants.h>
#include <psimd/butterfly.h>
#include <psimd/fft/aos.h>


static inline void v4f_fft8_real(
	const float t0[restrict static 16],
	const float t4[restrict static 16],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	v4f w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	v4f_fft4_aos(t0, t4, stride_t, row_offset, row_count,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i);

	const v4f half = v4f_splat(0.5f);
	const v4f g1r = half * (w1r + w3r);
	const v4f g1i = half * (w1i - w3i);
	const v4f two_h1r = w1i + w3i;
	const v4f two_h1i = w3r - w1r;

	const v4f sqrt2_over_4 = v4f_splat(SQRT2_OVER_4);
	const v4f h1_plus  = sqrt2_over_4 * (two_h1i + two_h1r);
	const v4f h1_minus = sqrt2_over_4 * (two_h1i - two_h1r);

	const v4f f0r = w0r + w0i;
	const v4f f4r = w0r - w0i;
	const v4f f1r = g1r + h1_plus;
	const v4f f1i = h1_minus + g1i;
	const v4f f2r =  w2r;
	const v4f f2i = -w2i;
	const v4f f3r = g1r - h1_plus;
	const v4f f3i = h1_minus - g1i;

	/* Store outputs */
	v4f_st(f + 0 * stride_f, f0r);
	v4f_st(f + 1 * stride_f, f4r);
	v4f_st(f + 2 * stride_f, f1r);
	v4f_st(f + 3 * stride_f, f1i);
	v4f_st(f + 4 * stride_f, f2r);
	v4f_st(f + 5 * stride_f, f2i);
	v4f_st(f + 6 * stride_f, f3r);
	v4f_st(f + 7 * stride_f, f3i);
}

static inline void v4f_fft16_real(
	const float t0[restrict static 32],
	const float t8[restrict static 32],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	v4f w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	v4f_fft8_aos(t0, t8, stride_t, row_offset, row_count,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i, &w4r, &w4i, &w5r, &w5i, &w6r, &w6i, &w7r, &w7i);

	const v4f half = v4f_splat(0.5f);
	const v4f g1r = half * (w1r + w7r);
	const v4f g1i = half * (w1i - w7i);
	const v4f g2r = half * (w2r + w6r);
	const v4f g2i = half * (w2i - w6i);
	const v4f g3r = half * (w3r + w5r);
	const v4f g3i = half * (w3i - w5i);

	const v4f two_h1r = w1i + w7i;
	const v4f two_h1i = w7r - w1r;
	const v4f two_h2r = w2i + w6i;
	const v4f two_h2i = w6r - w2r;
	const v4f two_h3r = w3i + w5i;
	const v4f two_h3i = w5r - w3r;

	const v4f sqrt2_over_4 = v4f_splat(SQRT2_OVER_4);
	const v4f h2_plus  = sqrt2_over_4 * (two_h2i + two_h2r);
	const v4f h2_minus = sqrt2_over_4 * (two_h2i - two_h2r);

	const v4f half_cos_1pi_over_8 = v4f_splat(COS_1PI_OVER_8 * 0.5f);
	const v4f half_cos_3pi_over_8 = v4f_splat(COS_3PI_OVER_8 * 0.5f);

	const v4f f0r =  w0r + w0i;
	const v4f f8r =  w0r - w0i;
	const v4f f1r =  g1r + two_h1r * half_cos_1pi_over_8 + two_h1i * half_cos_3pi_over_8;
	const v4f f1i =  g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;
	const v4f f2r =  g2r + h2_plus;
	const v4f f2i =  h2_minus + g2i;
	const v4f f3r =  g3r + two_h3r * half_cos_3pi_over_8 + two_h3i * half_cos_1pi_over_8;
	const v4f f3i =  g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const v4f f4r =  w4r;
	const v4f f4i = -w4i;
	const v4f f5r =  g3r - two_h3r * half_cos_3pi_over_8 - two_h3i * half_cos_1pi_over_8;
	const v4f f5i = -g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const v4f f6r =  g2r - h2_plus;
	const v4f f6i =  h2_minus - g2i;
	const v4f f7r =  g1r - two_h1r * half_cos_1pi_over_8 - two_h1i * half_cos_3pi_over_8;
	const v4f f7i = -g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;

	/* Store outputs */
	v4f_st(f +  0 * stride_f, f0r);
	v4f_st(f +  1 * stride_f, f8r);
	v4f_st(f +  2 * stride_f, f1r);
	v4f_st(f +  3 * stride_f, f1i);
	v4f_st(f +  4 * stride_f, f2r);
	v4f_st(f +  5 * stride_f, f2i);
	v4f_st(f +  6 * stride_f, f3r);
	v4f_st(f +  7 * stride_f, f3i);
	v4f_st(f +  8 * stride_f, f4r);
	v4f_st(f +  9 * stride_f, f4i);
	v4f_st(f + 10 * stride_f, f5r);
	v4f_st(f + 11 * stride_f, f5i);
	v4f_st(f + 12 * stride_f, f6r);
	v4f_st(f + 13 * stride_f, f6i);
	v4f_st(f + 14 * stride_f, f7r);
	v4f_st(f + 15 * stride_f, f7i);
}

static inline void v4f_ifft8_real(
	v4f f0r, v4f f4r, v4f f1r, v4f f1i, v4f f2r, v4f f2i, v4f f3r, v4f f3i,
	float t0[restrict static 16],
	float t4[restrict static 16],
	size_t stride_t)
{
	/* Load inputs and scale */
	const v4f scale = v4f_splat(0.5f);
	f0r *= scale;
	f4r *= scale;
	f1r *= scale;
	f1i *= scale;
	f3r *= scale;
	f3i *= scale;

	const v4f w0r =  f0r + f4r;
	const v4f w0i =  f0r - f4r;
	const v4f w2r =  f2r;
	const v4f w2i = -f2i;

	const v4f g1r = f1r + f3r;
	const v4f g1i = f1i - f3i;

	const v4f h1r = f1r - f3r;
	const v4f h1i = f1i + f3i;

	const v4f h1_plus  = h1r + h1i;
	const v4f h1_minus = h1r - h1i;

	const v4f sqrt2_over2 = v4f_splat(SQRT2_OVER_2);
	const v4f w1r =  g1r - sqrt2_over2 * h1_plus;
	const v4f w1i =  g1i + sqrt2_over2 * h1_minus;
	const v4f w3r =  g1r + sqrt2_over2 * h1_plus;
	const v4f w3i = -g1i + sqrt2_over2 * h1_minus;

	v4f_ifft4_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i,
		t0, t4, stride_t);
}

static inline void v4f_ifft16_real(
	v4f f0r, v4f f8r, v4f f1r, v4f f1i, v4f f2r, v4f f2i, v4f f3r, v4f f3i,
	v4f f4r, v4f f4i, v4f f5r, v4f f5i, v4f f6r, v4f f6i, v4f f7r, v4f f7i,
	float t0[restrict static 16],
	float t8[restrict static 16],
	size_t stride_t)
{
	/* Load inputs and scale */
	const v4f scale = v4f_splat(0.5f);
	f0r *= scale;
	f8r *= scale;
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

	const v4f w0r =  f0r + f8r;
	const v4f w0i =  f0r - f8r;
	const v4f w4r =  f4r;
	const v4f w4i = -f4i;

	const v4f g2r = f2r + f6r;
	const v4f g2i = f2i - f6i;

	const v4f h2r = f2r - f6r;
	const v4f h2i = f2i + f6i;

	const v4f h2_plus  = h2r + h2i;
	const v4f h2_minus = h2r - h2i;

	const v4f sqrt2_over2 = v4f_splat(SQRT2_OVER_2);
	const v4f w2r =  g2r - sqrt2_over2 * h2_plus;
	const v4f w2i =  g2i + sqrt2_over2 * h2_minus;
	const v4f w6r =  g2r + sqrt2_over2 * h2_plus;
	const v4f w6i = -g2i + sqrt2_over2 * h2_minus;

	const v4f g1r = f1r + f7r;
	const v4f g1i = f1i - f7i;
	const v4f g3r = f3r + f5r;
	const v4f g3i = f3i - f5i;

	const v4f h1r = f1r - f7r;
	const v4f h1i = f1i + f7i;
	const v4f h3r = f3r - f5r;
	const v4f h3i = f3i + f5i;

	const v4f cos_1pi_over_8 = v4f_splat(COS_1PI_OVER_8);
	const v4f cos_3pi_over_8 = v4f_splat(COS_3PI_OVER_8);
	const v4f w1r =  g1r - h1i * cos_1pi_over_8 - h1r * cos_3pi_over_8;
	const v4f w1i =  g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;
	const v4f w7r =  g1r + h1i * cos_1pi_over_8 + h1r * cos_3pi_over_8;
	const v4f w7i = -g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;

	const v4f w3r =  g3r - h3i * cos_3pi_over_8 - h3r * cos_1pi_over_8;
	const v4f w3i =  g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;
	const v4f w5r =  g3r + h3i * cos_3pi_over_8 + h3r * cos_1pi_over_8;
	const v4f w5i = -g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;

	v4f_ifft8_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
		t0, t8, stride_t);
}
