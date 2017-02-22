#pragma once

#include <nnpack/fft-constants.h>
#include <psimd.h>
#include <psimd/butterfly.h>
#include <psimd/fft/aos.h>


static inline void psimd_fft8_real_f32(
	const float t0[restrict static 16],
	const float t4[restrict static 16],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	psimd_f32 w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	psimd_fft4_aos_f32(t0, t4, stride_t, row_offset, row_count,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i);

	const psimd_f32 half = psimd_splat_f32(0.5f);
	const psimd_f32 g1r = half * (w1r + w3r);
	const psimd_f32 g1i = half * (w1i - w3i);
	const psimd_f32 two_h1r = w1i + w3i;
	const psimd_f32 two_h1i = w3r - w1r;

	const psimd_f32 sqrt2_over_4 = psimd_splat_f32(SQRT2_OVER_4);
	const psimd_f32 h1_plus  = sqrt2_over_4 * (two_h1i + two_h1r);
	const psimd_f32 h1_minus = sqrt2_over_4 * (two_h1i - two_h1r);

	const psimd_f32 f0r = w0r + w0i;
	const psimd_f32 f4r = w0r - w0i;
	const psimd_f32 f1r = g1r + h1_plus;
	const psimd_f32 f1i = h1_minus + g1i;
	const psimd_f32 f2r =  w2r;
	const psimd_f32 f2i = -w2i;
	const psimd_f32 f3r = g1r - h1_plus;
	const psimd_f32 f3i = h1_minus - g1i;

	/* Store outputs */
	psimd_store_f32(f + 0 * stride_f, f0r);
	psimd_store_f32(f + 1 * stride_f, f4r);
	psimd_store_f32(f + 2 * stride_f, f1r);
	psimd_store_f32(f + 3 * stride_f, f1i);
	psimd_store_f32(f + 4 * stride_f, f2r);
	psimd_store_f32(f + 5 * stride_f, f2i);
	psimd_store_f32(f + 6 * stride_f, f3r);
	psimd_store_f32(f + 7 * stride_f, f3i);
}

static inline void psimd_fft16_real_f32(
	const float t0[restrict static 32],
	const float t8[restrict static 32],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	psimd_f32 w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	psimd_fft8_aos_f32(t0, t8, stride_t, row_offset, row_count,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i, &w4r, &w4i, &w5r, &w5i, &w6r, &w6i, &w7r, &w7i);

	const psimd_f32 half = psimd_splat_f32(0.5f);
	const psimd_f32 g1r = half * (w1r + w7r);
	const psimd_f32 g1i = half * (w1i - w7i);
	const psimd_f32 g2r = half * (w2r + w6r);
	const psimd_f32 g2i = half * (w2i - w6i);
	const psimd_f32 g3r = half * (w3r + w5r);
	const psimd_f32 g3i = half * (w3i - w5i);

	const psimd_f32 two_h1r = w1i + w7i;
	const psimd_f32 two_h1i = w7r - w1r;
	const psimd_f32 two_h2r = w2i + w6i;
	const psimd_f32 two_h2i = w6r - w2r;
	const psimd_f32 two_h3r = w3i + w5i;
	const psimd_f32 two_h3i = w5r - w3r;

	const psimd_f32 sqrt2_over_4 = psimd_splat_f32(SQRT2_OVER_4);
	const psimd_f32 h2_plus  = sqrt2_over_4 * (two_h2i + two_h2r);
	const psimd_f32 h2_minus = sqrt2_over_4 * (two_h2i - two_h2r);

	const psimd_f32 half_cos_1pi_over_8 = psimd_splat_f32(COS_1PI_OVER_8 * 0.5f);
	const psimd_f32 half_cos_3pi_over_8 = psimd_splat_f32(COS_3PI_OVER_8 * 0.5f);

	const psimd_f32 f0r =  w0r + w0i;
	const psimd_f32 f8r =  w0r - w0i;
	const psimd_f32 f1r =  g1r + two_h1r * half_cos_1pi_over_8 + two_h1i * half_cos_3pi_over_8;
	const psimd_f32 f1i =  g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;
	const psimd_f32 f2r =  g2r + h2_plus;
	const psimd_f32 f2i =  h2_minus + g2i;
	const psimd_f32 f3r =  g3r + two_h3r * half_cos_3pi_over_8 + two_h3i * half_cos_1pi_over_8;
	const psimd_f32 f3i =  g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const psimd_f32 f4r =  w4r;
	const psimd_f32 f4i = -w4i;
	const psimd_f32 f5r =  g3r - two_h3r * half_cos_3pi_over_8 - two_h3i * half_cos_1pi_over_8;
	const psimd_f32 f5i = -g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const psimd_f32 f6r =  g2r - h2_plus;
	const psimd_f32 f6i =  h2_minus - g2i;
	const psimd_f32 f7r =  g1r - two_h1r * half_cos_1pi_over_8 - two_h1i * half_cos_3pi_over_8;
	const psimd_f32 f7i = -g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;

	/* Store outputs */
	psimd_store_f32(f +  0 * stride_f, f0r);
	psimd_store_f32(f +  1 * stride_f, f8r);
	psimd_store_f32(f +  2 * stride_f, f1r);
	psimd_store_f32(f +  3 * stride_f, f1i);
	psimd_store_f32(f +  4 * stride_f, f2r);
	psimd_store_f32(f +  5 * stride_f, f2i);
	psimd_store_f32(f +  6 * stride_f, f3r);
	psimd_store_f32(f +  7 * stride_f, f3i);
	psimd_store_f32(f +  8 * stride_f, f4r);
	psimd_store_f32(f +  9 * stride_f, f4i);
	psimd_store_f32(f + 10 * stride_f, f5r);
	psimd_store_f32(f + 11 * stride_f, f5i);
	psimd_store_f32(f + 12 * stride_f, f6r);
	psimd_store_f32(f + 13 * stride_f, f6i);
	psimd_store_f32(f + 14 * stride_f, f7r);
	psimd_store_f32(f + 15 * stride_f, f7i);
}

static inline void psimd_ifft8_real_f32(
	psimd_f32 f0r, psimd_f32 f4r, psimd_f32 f1r, psimd_f32 f1i, psimd_f32 f2r, psimd_f32 f2i, psimd_f32 f3r, psimd_f32 f3i,
	float t0[restrict static 16],
	float t4[restrict static 16],
	size_t stride_t)
{
	/* Load inputs and scale */
	const psimd_f32 scale = psimd_splat_f32(0.5f);
	f0r *= scale;
	f4r *= scale;
	f1r *= scale;
	f1i *= scale;
	f3r *= scale;
	f3i *= scale;

	const psimd_f32 w0r =  f0r + f4r;
	const psimd_f32 w0i =  f0r - f4r;
	const psimd_f32 w2r =  f2r;
	const psimd_f32 w2i = -f2i;

	const psimd_f32 g1r = f1r + f3r;
	const psimd_f32 g1i = f1i - f3i;

	const psimd_f32 h1r = f1r - f3r;
	const psimd_f32 h1i = f1i + f3i;

	const psimd_f32 h1_plus  = h1r + h1i;
	const psimd_f32 h1_minus = h1r - h1i;

	const psimd_f32 sqrt2_over2 = psimd_splat_f32(SQRT2_OVER_2);
	const psimd_f32 w1r =  g1r - sqrt2_over2 * h1_plus;
	const psimd_f32 w1i =  g1i + sqrt2_over2 * h1_minus;
	const psimd_f32 w3r =  g1r + sqrt2_over2 * h1_plus;
	const psimd_f32 w3i = -g1i + sqrt2_over2 * h1_minus;

	psimd_ifft4_aos_f32(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i,
		t0, t4, stride_t);
}

static inline void psimd_ifft16_real_f32(
	psimd_f32 f0r, psimd_f32 f8r, psimd_f32 f1r, psimd_f32 f1i, psimd_f32 f2r, psimd_f32 f2i, psimd_f32 f3r, psimd_f32 f3i,
	psimd_f32 f4r, psimd_f32 f4i, psimd_f32 f5r, psimd_f32 f5i, psimd_f32 f6r, psimd_f32 f6i, psimd_f32 f7r, psimd_f32 f7i,
	float t0[restrict static 16],
	float t8[restrict static 16],
	size_t stride_t)
{
	/* Load inputs and scale */
	const psimd_f32 scale = psimd_splat_f32(0.5f);
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

	const psimd_f32 w0r =  f0r + f8r;
	const psimd_f32 w0i =  f0r - f8r;
	const psimd_f32 w4r =  f4r;
	const psimd_f32 w4i = -f4i;

	const psimd_f32 g2r = f2r + f6r;
	const psimd_f32 g2i = f2i - f6i;

	const psimd_f32 h2r = f2r - f6r;
	const psimd_f32 h2i = f2i + f6i;

	const psimd_f32 h2_plus  = h2r + h2i;
	const psimd_f32 h2_minus = h2r - h2i;

	const psimd_f32 sqrt2_over2 = psimd_splat_f32(SQRT2_OVER_2);
	const psimd_f32 w2r =  g2r - sqrt2_over2 * h2_plus;
	const psimd_f32 w2i =  g2i + sqrt2_over2 * h2_minus;
	const psimd_f32 w6r =  g2r + sqrt2_over2 * h2_plus;
	const psimd_f32 w6i = -g2i + sqrt2_over2 * h2_minus;

	const psimd_f32 g1r = f1r + f7r;
	const psimd_f32 g1i = f1i - f7i;
	const psimd_f32 g3r = f3r + f5r;
	const psimd_f32 g3i = f3i - f5i;

	const psimd_f32 h1r = f1r - f7r;
	const psimd_f32 h1i = f1i + f7i;
	const psimd_f32 h3r = f3r - f5r;
	const psimd_f32 h3i = f3i + f5i;

	const psimd_f32 cos_1pi_over_8 = psimd_splat_f32(COS_1PI_OVER_8);
	const psimd_f32 cos_3pi_over_8 = psimd_splat_f32(COS_3PI_OVER_8);
	const psimd_f32 w1r =  g1r - h1i * cos_1pi_over_8 - h1r * cos_3pi_over_8;
	const psimd_f32 w1i =  g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;
	const psimd_f32 w7r =  g1r + h1i * cos_1pi_over_8 + h1r * cos_3pi_over_8;
	const psimd_f32 w7i = -g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;

	const psimd_f32 w3r =  g3r - h3i * cos_3pi_over_8 - h3r * cos_1pi_over_8;
	const psimd_f32 w3i =  g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;
	const psimd_f32 w5r =  g3r + h3i * cos_3pi_over_8 + h3r * cos_1pi_over_8;
	const psimd_f32 w5i = -g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;

	psimd_ifft8_aos_f32(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
		t0, t8, stride_t);
}
