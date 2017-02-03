#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/simd.h>
#include <nnpack/fft-constants.h>
#include <psimd/butterfly.h>


static inline void v4f_fft4_aos(
	const float t_lo[restrict static 16],
	const float t_hi[restrict static 16],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	v4f f0r[restrict static 1],
	v4f f0i[restrict static 1],
	v4f f1r[restrict static 1],
	v4f f1i[restrict static 1],
	v4f f2r[restrict static 1],
	v4f f2i[restrict static 1],
	v4f f3r[restrict static 1],
	v4f f3i[restrict static 1])
{
	/* Load inputs and FFT4: butterfly */
	v4f w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	w0r = w0i = w1r = w1i = w2r = w2i = w3r = w3i = v4f_zero();

	const uint32_t row_end = row_start + row_count;
	if (row_start <= 0) {
		w0r = w2r = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 4 && row_end > 4) {
		w2r = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w0r, &w2r);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 1) {
		w0i = w2i = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 5 && row_end > 5) {
		w2i = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w0i, &w2i);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 2) {
		w1r = w3r = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 6 && row_end > 6) {
		w3r = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w1r, &w3r);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 3) {
		w1i = w3i = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 7 && row_end > 7) {
		w3i = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w1i, &w3i);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

fft4_twiddle:
	/*
	 * FFT4: multiplication by twiddle factors:
	 *   w3r, w3i = w3i, -w3r
	 * (negation of w3i is merged into the next butterfly)
	 */
	v4f_swap(&w3r, &w3i);

	/*
	 * 2x FFT2: butterfly
	 */
	v4f_butterfly(&w0r, &w1r);
	v4f_butterfly(&w0i, &w1i);
	v4f_butterfly(&w2r, &w3r);
	v4f_butterfly_with_negated_b(&w2i, &w3i);

	/* Bit reversal */
	v4f_swap(&w1r, &w2r);
	v4f_swap(&w1i, &w2i);

	*f0r = w0r;
	*f0i = w0i;
	*f1r = w1r;
	*f1i = w1i;
	*f2r = w2r;
	*f2i = w2i;
	*f3r = w3r;
	*f3i = w3i;
}

static inline void v4f_ifft4_aos(
	v4f w0r, v4f w0i, v4f w1r, v4f w1i, v4f w2r, v4f w2i, v4f w3r, v4f w3i,
	float t0[restrict static 16],
	float t2[restrict static 16],
	size_t stride_t)
{
	/* Bit reversal */
	v4f_swap(&w1r, &w2r);
	v4f_swap(&w1i, &w2i);

	/*
	 * 2x IFFT2: butterfly
	 */
	v4f_butterfly(&w0r, &w1r);
	v4f_butterfly(&w0i, &w1i);
	v4f_butterfly(&w2r, &w3r);
	v4f_butterfly(&w2i, &w3i);

	/*
	 * IFFT4: multiplication by twiddle factors:
	 *   w3r, w3i = -w3i, w3r
	 * (negation of w3r is merged into the next butterfly)
	 */
	v4f_swap(&w3r, &w3i);

	/* IFFT4: scaling by 1/4 */
	const v4f scale = v4f_splat(0.25f);
	w0r *= scale;
	w0i *= scale;
	w1r *= scale;
	w1i *= scale;
	w2r *= scale;
	w2i *= scale;
	w3r *= scale;
	w3i *= scale;

	/* IFFT4: butterfly and store outputs */
	v4f_butterfly(&w0r, &w2r);
	v4f_st(t0, w0r);
	t0 += stride_t;
	v4f_st(t2, w2r);
	t2 += stride_t;

	v4f_butterfly(&w0i, &w2i);
	v4f_st(t0, w0i);
	t0 += stride_t;
	v4f_st(t2, w2i);
	t2 += stride_t;

	v4f_butterfly_with_negated_b(&w1r, &w3r);
	v4f_st(t0, w1r);
	t0 += stride_t;
	v4f_st(t2, w3r);
	t2 += stride_t;

	v4f_butterfly(&w1i, &w3i);
	v4f_st(t0, w1i);
	v4f_st(t2, w3i);
}

static inline void v4f_fft8_aos(
	const float t_lo[restrict static 32],
	const float t_hi[restrict static 32],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	v4f f0r[restrict static 1],
	v4f f0i[restrict static 1],
	v4f f1r[restrict static 1],
	v4f f1i[restrict static 1],
	v4f f2r[restrict static 1],
	v4f f2i[restrict static 1],
	v4f f3r[restrict static 1],
	v4f f3i[restrict static 1],
	v4f f4r[restrict static 1],
	v4f f4i[restrict static 1],
	v4f f5r[restrict static 1],
	v4f f5i[restrict static 1],
	v4f f6r[restrict static 1],
	v4f f6i[restrict static 1],
	v4f f7r[restrict static 1],
	v4f f7i[restrict static 1])
{
	/* Load inputs and FFT8: butterfly */
	v4f w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	w0r = w0i = w1r = w1i = w2r = w2i = w3r = w3i = w4r = w4i = w5r = w5i = w6r = w6i = w7r = w7i = v4f_zero();

	const uint32_t row_end = row_start + row_count;
	if (row_start <= 0) {
		w0r = w4r = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 8 && row_end > 8) {
		w4r = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w0r, &w4r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 1) {
		w0i = w4i = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 9 && row_end > 9) {
		w4i = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w0i, &w4i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 2) {
		w1r = w5r = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 10 && row_end > 10) {
		w5r = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w1r, &w5r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 3) {
		w1i = w5i = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 11 && row_end > 11) {
		w5i = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w1i, &w5i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 4) {
		w2r = w6r = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 12 && row_end > 12) {
		w6r = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w2r, &w6r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 5) {
		w2i = w6i = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 13 && row_end > 13) {
		w6i = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w2i, &w6i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 6) {
		w3r = w7r = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 14 && row_end > 14) {
		w7r = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w3r, &w7r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 7) {
		w3i = w7i = v4f_ld(t_lo);
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 15 && row_end > 15) {
		w7i = v4f_ld(t_hi);
		t_hi += stride_t;
		v4f_butterfly(&w3i, &w7i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

fft8_twiddle:;
	/*
	 * FFT8: multiplication by twiddle factors:
	 *   
	 *   w5r, w5i = sqrt(2)/2 (w5i + w5r),  sqrt(2)/2 (w5i - w5r)
	 *   w6r, w6i = w6i, -w6r
	 *   w7r, w7i = sqrt(2)/2 (w7i - w7r), -sqrt(2)/2 (w7i + w7r)
	 *
	 * (negation of w6i and w7i is merged into the next butterfly)
	 */
	const v4f sqrt2_over_2 = v4f_splat(SQRT2_OVER_2);
	const v4f new_w5r = sqrt2_over_2 * (w5i + w5r);
	const v4f new_w5i = sqrt2_over_2 * (w5i - w5r);
	const v4f new_w7r = sqrt2_over_2 * (w7i - w7r);
	const v4f minus_new_w7i = sqrt2_over_2 * (w7i + w7r);
	w5r = new_w5r;
	w5i = new_w5i;
	v4f_swap(&w6r, &w6i);
	w7r = new_w7r;
	w7i = minus_new_w7i;

	/*
	 * 2x FFT4: butterfly
	 */
	v4f_butterfly(&w0r, &w2r);
	v4f_butterfly(&w0i, &w2i);
	v4f_butterfly(&w1r, &w3r);
	v4f_butterfly(&w1i, &w3i);
	v4f_butterfly(&w4r, &w6r);
	v4f_butterfly_with_negated_b(&w4i, &w6i);
	v4f_butterfly(&w5r, &w7r);
	v4f_butterfly_with_negated_b(&w5i, &w7i);

	/*
	 * 2x FFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = w3i, -w3r
	 *   w7r, w7i = w7i, -w7r
	 *
	 * (negation of w3i and w7i is merged into the next butterfly)
	 */
	v4f_swap(&w3r, &w3i);
	v4f_swap(&w7r, &w7i);

	/*
	 * 4x FFT2: butterfly
	 */
	v4f_butterfly(&w0r, &w1r);
	v4f_butterfly(&w0i, &w1i);
	v4f_butterfly(&w2r, &w3r);
	v4f_butterfly_with_negated_b(&w2i, &w3i);
	v4f_butterfly(&w4r, &w5r);
	v4f_butterfly(&w4i, &w5i);
	v4f_butterfly(&w6r, &w7r);
	v4f_butterfly_with_negated_b(&w6i, &w7i);

	/* Bit reversal */
	v4f_swap(&w1r, &w4r);
	v4f_swap(&w1i, &w4i);
	v4f_swap(&w3r, &w6r);
	v4f_swap(&w3i, &w6i);

	*f0r = w0r;
	*f0i = w0i;
	*f1r = w1r;
	*f1i = w1i;
	*f2r = w2r;
	*f2i = w2i;
	*f3r = w3r;
	*f3i = w3i;
	*f4r = w4r;
	*f4i = w4i;
	*f5r = w5r;
	*f5i = w5i;
	*f6r = w6r;
	*f6i = w6i;
	*f7r = w7r;
	*f7i = w7i;
}

static inline void v4f_ifft8_aos(
	v4f w0r, v4f w0i, v4f w1r, v4f w1i, v4f w2r, v4f w2i, v4f w3r, v4f w3i,
	v4f w4r, v4f w4i, v4f w5r, v4f w5i, v4f w6r, v4f w6i, v4f w7r, v4f w7i,
	float t_lo[restrict static 32],
	float t_hi[restrict static 32],
	size_t stride_t)
{
	/* Bit reversal */
	v4f_swap(&w1r, &w4r);
	v4f_swap(&w1i, &w4i);
	v4f_swap(&w3r, &w6r);
	v4f_swap(&w3i, &w6i);

	/*
	 * 4x IFFT2: butterfly
	 */
	v4f_butterfly(&w0r, &w1r);
	v4f_butterfly(&w0i, &w1i);
	v4f_butterfly(&w2r, &w3r);
	v4f_butterfly(&w2i, &w3i);
	v4f_butterfly(&w4r, &w5r);
	v4f_butterfly(&w4i, &w5i);
	v4f_butterfly(&w6r, &w7r);
	v4f_butterfly(&w6i, &w7i);

	/*
	 * 2x IFFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = -w3i, w3r
	 *   w7r, w7i = -w7i, w7r
	 *
	 * (negation of w3r and w7r is merged into the next butterfly)
	 */
	v4f_swap(&w3r, &w3i);
	v4f_swap(&w7r, &w7i);

	/*
	 * 2x IFFT4: butterfly
	 */
	v4f_butterfly(&w0r, &w2r);
	v4f_butterfly(&w0i, &w2i);
	v4f_butterfly_with_negated_b(&w1r, &w3r);
	v4f_butterfly(&w1i, &w3i);
	v4f_butterfly(&w4r, &w6r);
	v4f_butterfly(&w4i, &w6i);
	v4f_butterfly_with_negated_b(&w5r, &w7r);
	v4f_butterfly(&w5i, &w7i);

	/*
	 * IFFT8: multiplication by twiddle factors and scaling by 1/8:
	 *
	 *   w5r, w5i =  sqrt(2)/2 (w5r - w5i), sqrt(2)/2 (w5r + w5i)
	 *   w6r, w6i = -w6i, w6r
	 *   w7r, w7i = -sqrt(2)/2 (w7r + w7i), sqrt(2)/2 (w7r - w7i)
	 *
	 * (negation of w6r and w7r is merged into the next butterfly)
	 */
	const v4f sqrt2_over_2 = v4f_splat(SQRT2_OVER_2 * 0.125f);
	const v4f new_w5r = sqrt2_over_2 * (w5r - w5i);
	const v4f new_w5i = sqrt2_over_2 * (w5r + w5i);
	const v4f minus_new_w7r = sqrt2_over_2 * (w7r + w7i);
	const v4f new_w7i = sqrt2_over_2 * (w7r - w7i);
	w5r = new_w5r;
	w5i = new_w5i;
	v4f_swap(&w6r, &w6i);
	w7r = minus_new_w7r;
	w7i = new_w7i;

	/* IFFT8: scaling of remaining coefficients by 1/8 */
	const v4f scale = v4f_splat(0.125f);
	w0r *= scale;
	w0i *= scale;
	w1r *= scale;
	w1i *= scale;
	w2r *= scale;
	w2i *= scale;
	w3r *= scale;
	w3i *= scale;
	w4r *= scale;
	w4i *= scale;
	w6r *= scale;
	w6i *= scale;

	/* IFFT8: butterfly and store outputs */
	v4f_butterfly(&w0r, &w4r);
	v4f_st(t_lo, w0r);
	t_lo += stride_t;
	v4f_st(t_hi, w4r);
	t_hi += stride_t;

	v4f_butterfly(&w0i, &w4i);
	v4f_st(t_lo, w0i);
	t_lo += stride_t;
	v4f_st(t_hi, w4i);
	t_hi += stride_t;

	v4f_butterfly(&w1r, &w5r);
	v4f_st(t_lo, w1r);
	t_lo += stride_t;
	v4f_st(t_hi, w5r);
	t_hi += stride_t;

	v4f_butterfly(&w1i, &w5i);
	v4f_st(t_lo, w1i);
	t_lo += stride_t;
	v4f_st(t_hi, w5i);
	t_hi += stride_t;

	v4f_butterfly_with_negated_b(&w2r, &w6r);
	v4f_st(t_lo, w2r);
	t_lo += stride_t;
	v4f_st(t_hi, w6r);
	t_hi += stride_t;

	v4f_butterfly(&w2i, &w6i);
	v4f_st(t_lo, w2i);
	t_lo += stride_t;
	v4f_st(t_hi, w6i);
	t_hi += stride_t;

	v4f_butterfly_with_negated_b(&w3r, &w7r);
	v4f_st(t_lo, w3r);
	t_lo += stride_t;
	v4f_st(t_hi, w7r);
	t_hi += stride_t;

	v4f_butterfly(&w3i, &w7i);
	v4f_st(t_lo, w3i);
	v4f_st(t_hi, w7i);
}
