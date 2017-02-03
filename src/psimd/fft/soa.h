#pragma once

#include <nnpack/simd.h>
#include <nnpack/fft-constants.h>
#include <psimd/butterfly.h>


static inline void v4f_mul_soa(
	v4f real[restrict static 1],
	v4f imag[restrict static 1],
	v4f cr, v4f ci)
{
	const v4f xr = *real;
	const v4f xi = *imag;

	const v4f new_xr = xr * cr - xi * ci;
	const v4f new_xi = xi * cr + xr * ci;

	*real = new_xr;
	*imag = new_xi;
}

static inline void v4f_mulc_soa(
	v4f real[restrict static 1],
	v4f imag[restrict static 1],
	v4f cr, v4f ci)
{
	const v4f xr = *real;
	const v4f xi = *imag;

	const v4f new_xr = xr * cr + xi * ci;
	const v4f new_xi = xi * cr - xr * ci;

	*real = new_xr;
	*imag = new_xi;
}

static inline void v4f_fft8_soa(
	v4f real0123[restrict static 1],
	v4f real4567[restrict static 1],
	v4f imag0123[restrict static 1],
	v4f imag4567[restrict static 1])
{
	v4f w0123r = *real0123;
	v4f w4567r = *real4567;
	v4f w0123i = *imag0123;
	v4f w4567i = *imag4567;

	/* FFT8: butterfly */
	v4f_butterfly(&w0123r, &w4567r);
	v4f_butterfly(&w0123i, &w4567i);

	/* FFT8: multiplication by twiddle factors */
	v4f_mulc_soa(&w4567r, &w4567i,
		(v4f) { COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4 },
		(v4f) { SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4 });

	/* 2x FFT4: butterfly */
	v4f w0145r = __builtin_shufflevector(w0123r, w4567r, 0, 1, 4, 5);
	v4f w2367r = __builtin_shufflevector(w0123r, w4567r, 2, 3, 6, 7);
	v4f_butterfly(&w0145r, &w2367r);

	v4f w0145i = __builtin_shufflevector(w0123i, w4567i, 0, 1, 4, 5);
	v4f w2367i = __builtin_shufflevector(w0123i, w4567i, 2, 3, 6, 7);
	v4f_butterfly(&w0145i, &w2367i);

	/* 2x FFT4: multiplication by twiddle factors */
	v4f_mulc_soa(&w2367r, &w2367i,
		(v4f) { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 },
		(v4f) { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 });

	/* 4x FFT2: butterfly */
	v4f w0426r = __builtin_shufflevector(w0145r, w2367r, 0, 2, 4, 6);
	v4f w1537r = __builtin_shufflevector(w0145r, w2367r, 1, 3, 5, 7);
	v4f_butterfly(&w0426r, &w1537r);

	v4f w0426i = __builtin_shufflevector(w0145i, w2367i, 0, 2, 4, 6);
	v4f w1537i = __builtin_shufflevector(w0145i, w2367i, 1, 3, 5, 7);
	v4f_butterfly(&w0426i, &w1537i);

	/*
	 * Bit reversal:
	 *   0  4  2  6  1  5  3  7
	 *   ^  ^  ^  ^  ^  ^  ^  ^
	 *   |  |  |  |  |  |  |  |
	 *   0  1  2  3  4  5  6  7
	 */
	*real0123 = w0426r;
	*real4567 = w1537r;
	*imag0123 = w0426i;
	*imag4567 = w1537i;
}

static inline void v4f_ifft8_soa(
	v4f real0123[restrict static 1],
	v4f real4567[restrict static 1],
	v4f imag0123[restrict static 1],
	v4f imag4567[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0  4  2  6  1  5  3  7
	 *   ^  ^  ^  ^  ^  ^  ^  ^
	 *   |  |  |  |  |  |  |  |
	 *   0  1  2  3  4  5  6  7
	 */
	v4f w0426r = *real0123;
	v4f w1537r = *real4567;
	v4f w0426i = *imag0123;
	v4f w1537i = *imag4567;

	/* 4x IFFT2: butterfly */
	v4f_butterfly(&w0426r, &w1537r);
	v4f w0145r = __builtin_shufflevector(w0426r, w1537r, 0, 4, 1, 5);
	v4f w2367r = __builtin_shufflevector(w0426r, w1537r, 2, 6, 3, 7);

	v4f_butterfly(&w0426i, &w1537i);
	v4f w0145i = __builtin_shufflevector(w0426i, w1537i, 0, 4, 1, 5);
	v4f w2367i = __builtin_shufflevector(w0426i, w1537i, 2, 6, 3, 7);

	/* 2x IFFT4: multiplication by twiddle factors */
	v4f_mul_soa(&w2367r, &w2367i,
		(v4f) { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 },
		(v4f) { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 });

	/* 2x IFFT4: butterfly */
	v4f_butterfly(&w0145r, &w2367r);
	v4f w0123r = __builtin_shufflevector(w0145r, w2367r, 0, 1, 4, 5);
	v4f w4567r = __builtin_shufflevector(w0145r, w2367r, 2, 3, 6, 7);

	v4f_butterfly(&w0145i, &w2367i);
	v4f w0123i = __builtin_shufflevector(w0145i, w2367i, 0, 1, 4, 5);
	v4f w4567i = __builtin_shufflevector(w0145i, w2367i, 2, 3, 6, 7);

	/* IFFT8: multiplication by twiddle factors and scaling by 1/8 */
	v4f_mul_soa(&w4567r, &w4567i,
		(v4f) { COS_0PI_OVER_4 * 0.125f, COS_1PI_OVER_4 * 0.125f, COS_2PI_OVER_4 * 0.125f, COS_3PI_OVER_4 * 0.125f },
		(v4f) { SIN_0PI_OVER_4 * 0.125f, SIN_1PI_OVER_4 * 0.125f, SIN_2PI_OVER_4 * 0.125f, SIN_3PI_OVER_4 * 0.125f });

	/* IFFT8: scaling of remaining coefficients by 1/8 */
	const v4f scale = v4f_splat(0.125f);
	w0123r *= scale;
	w0123i *= scale;

	/* IFFT8: butterfly */
	v4f_butterfly(&w0123r, &w4567r);
	v4f_butterfly(&w0123i, &w4567i);

	*real0123 = w0123r;
	*real4567 = w4567r;
	*imag0123 = w0123i;
	*imag4567 = w4567i;
}

static inline void v4f_fft16_soa(
	v4f real0123[restrict static 1],
	v4f real4567[restrict static 1],
	v4f real89AB[restrict static 1],
	v4f realCDEF[restrict static 1],
	v4f imag0123[restrict static 1],
	v4f imag4567[restrict static 1],
	v4f imag89AB[restrict static 1],
	v4f imagCDEF[restrict static 1])
{
	v4f w0123r = *real0123;
	v4f w4567r = *real4567;
	v4f w89ABr = *real89AB;
	v4f wCDEFr = *realCDEF;
	v4f w0123i = *imag0123;
	v4f w4567i = *imag4567;
	v4f w89ABi = *imag89AB;
	v4f wCDEFi = *imagCDEF;

	/* FFT16: butterfly */
	v4f_butterfly(&w0123r, &w89ABr);
	v4f_butterfly(&w4567r, &wCDEFr);
	v4f_butterfly(&w0123i, &w89ABi);
	v4f_butterfly(&w4567i, &wCDEFi);

	/* FFT16: multiplication by twiddle factors */
	v4f_mulc_soa(&w89ABr, &w89ABi,
		(v4f) { COS_0PI_OVER_8, COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8 },
		(v4f) { SIN_0PI_OVER_8, SIN_1PI_OVER_8, SIN_2PI_OVER_8, SIN_3PI_OVER_8 });
	v4f_mulc_soa(&wCDEFr, &wCDEFi,
		(v4f) { COS_4PI_OVER_8, COS_5PI_OVER_8, COS_6PI_OVER_8, COS_7PI_OVER_8 },
		(v4f) { SIN_4PI_OVER_8, SIN_5PI_OVER_8, SIN_6PI_OVER_8, SIN_7PI_OVER_8 });

	/* 2x FFT8: butterfly */
	v4f_butterfly(&w0123r, &w4567r);
	v4f_butterfly(&w89ABr, &wCDEFr);
	v4f_butterfly(&w0123i, &w4567i);
	v4f_butterfly(&w89ABi, &wCDEFi);

	/* 2x FFT8: multiplication by twiddle factors */
	const v4f fft8_cos_twiddle_factor = { COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4 };
	const v4f fft8_sin_twiddle_factor = { SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4 };
	v4f_mulc_soa(&w4567r, &w4567i, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);
	v4f_mulc_soa(&wCDEFr, &wCDEFi, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);

	/* 4x FFT4: butterfly */
	v4f w0189r = __builtin_shufflevector(w0123r, w89ABr, 0, 1, 4, 5);
	v4f w23ABr = __builtin_shufflevector(w0123r, w89ABr, 2, 3, 6, 7);
	v4f_butterfly(&w0189r, &w23ABr);

	v4f w0189i = __builtin_shufflevector(w0123i, w89ABi, 0, 1, 4, 5);
	v4f w23ABi = __builtin_shufflevector(w0123i, w89ABi, 2, 3, 6, 7);
	v4f_butterfly(&w0189i, &w23ABi);

	v4f w45CDr = __builtin_shufflevector(w4567r, wCDEFr, 0, 1, 4, 5);
	v4f w67EFr = __builtin_shufflevector(w4567r, wCDEFr, 2, 3, 6, 7);
	v4f_butterfly(&w45CDr, &w67EFr);

	v4f w45CDi = __builtin_shufflevector(w4567i, wCDEFi, 0, 1, 4, 5);
	v4f w67EFi = __builtin_shufflevector(w4567i, wCDEFi, 2, 3, 6, 7);
	v4f_butterfly(&w45CDi, &w67EFi);

	/* 4x FFT4: multiplication by twiddle factors */
	const v4f fft4_cos_twiddle_factor = { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 };
	const v4f fft4_sin_twiddle_factor = { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 };
	v4f_mulc_soa(&w23ABr, &w23ABi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);
	v4f_mulc_soa(&w67EFr, &w67EFi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);

	/* 8x FFT2: butterfly */
	v4f w084Cr = __builtin_shufflevector(w0189r, w45CDr, 0, 2, 4, 6);
	v4f w195Dr = __builtin_shufflevector(w0189r, w45CDr, 1, 3, 5, 7);
	v4f_butterfly(&w084Cr, &w195Dr);

	v4f w084Ci = __builtin_shufflevector(w0189i, w45CDi, 0, 2, 4, 6);
	v4f w195Di = __builtin_shufflevector(w0189i, w45CDi, 1, 3, 5, 7);
	v4f_butterfly(&w084Ci, &w195Di);

	v4f w2A6Er = __builtin_shufflevector(w23ABr, w67EFr, 0, 2, 4, 6);
	v4f w3B7Fr = __builtin_shufflevector(w23ABr, w67EFr, 1, 3, 5, 7);
	v4f_butterfly(&w2A6Er, &w3B7Fr);

	v4f w2A6Ei = __builtin_shufflevector(w23ABi, w67EFi, 0, 2, 4, 6);
	v4f w3B7Fi = __builtin_shufflevector(w23ABi, w67EFi, 1, 3, 5, 7);
	v4f_butterfly(&w2A6Ei, &w3B7Fi);

	/*
	 * Bit reversal:
	 *   0   8   4  12   2  10   6  14   1   9   5  13   3  11   7  15
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
	 */
	*real0123 = w084Cr;
	*real4567 = w2A6Er;
	*real89AB = w195Dr;
	*realCDEF = w3B7Fr;
	*imag0123 = w084Ci;
	*imag4567 = w2A6Ei;
	*imag89AB = w195Di;
	*imagCDEF = w3B7Fi;
}

static inline void v4f_ifft16_soa(
	v4f real0123[restrict static 1],
	v4f real4567[restrict static 1],
	v4f real89AB[restrict static 1],
	v4f realCDEF[restrict static 1],
	v4f imag0123[restrict static 1],
	v4f imag4567[restrict static 1],
	v4f imag89AB[restrict static 1],
	v4f imagCDEF[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0   8   4  12   2  10   6  14   1   9   5  13   3  11   7  15
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
	 */
	v4f w084Cr = *real0123;
	v4f w2A6Er = *real4567;
	v4f w195Dr = *real89AB;
	v4f w3B7Fr = *realCDEF;
	v4f w084Ci = *imag0123;
	v4f w2A6Ei = *imag4567;
	v4f w195Di = *imag89AB;
	v4f w3B7Fi = *imagCDEF;

	/* 8x IFFT2: butterfly */
	v4f_butterfly(&w084Cr, &w195Dr);
	v4f w0189r = __builtin_shufflevector(w084Cr, w195Dr, 0, 4, 1, 5);
	v4f w45CDr = __builtin_shufflevector(w084Cr, w195Dr, 2, 6, 3, 7);

	v4f_butterfly(&w084Ci, &w195Di);
	v4f w0189i = __builtin_shufflevector(w084Ci, w195Di, 0, 4, 1, 5);
	v4f w45CDi = __builtin_shufflevector(w084Ci, w195Di, 2, 6, 3, 7);

	v4f_butterfly(&w2A6Er, &w3B7Fr);
	v4f w23ABr = __builtin_shufflevector(w2A6Er, w3B7Fr, 0, 4, 1, 5);
	v4f w67EFr = __builtin_shufflevector(w2A6Er, w3B7Fr, 2, 6, 3, 7);

	v4f_butterfly(&w2A6Ei, &w3B7Fi);
	v4f w23ABi = __builtin_shufflevector(w2A6Ei, w3B7Fi, 0, 4, 1, 5);
	v4f w67EFi = __builtin_shufflevector(w2A6Ei, w3B7Fi, 2, 6, 3, 7);

	/* 4x IFFT4: multiplication by twiddle factors */
	const v4f fft4_cos_twiddle_factor = { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 };
	const v4f fft4_sin_twiddle_factor = { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 };
	v4f_mul_soa(&w23ABr, &w23ABi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);
	v4f_mul_soa(&w67EFr, &w67EFi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);

	/* 4x IFFT4: butterfly */
	v4f_butterfly(&w0189r, &w23ABr);
	v4f w0123r = __builtin_shufflevector(w0189r, w23ABr, 0, 1, 4, 5);
	v4f w89ABr = __builtin_shufflevector(w0189r, w23ABr, 2, 3, 6, 7);

	v4f_butterfly(&w0189i, &w23ABi);
	v4f w0123i = __builtin_shufflevector(w0189i, w23ABi, 0, 1, 4, 5);
	v4f w89ABi = __builtin_shufflevector(w0189i, w23ABi, 2, 3, 6, 7);

	v4f_butterfly(&w45CDr, &w67EFr);
	v4f w4567r = __builtin_shufflevector(w45CDr, w67EFr, 0, 1, 4, 5);
	v4f wCDEFr = __builtin_shufflevector(w45CDr, w67EFr, 2, 3, 6, 7);

	v4f_butterfly(&w45CDi, &w67EFi);
	v4f w4567i = __builtin_shufflevector(w45CDi, w67EFi, 0, 1, 4, 5);
	v4f wCDEFi = __builtin_shufflevector(w45CDi, w67EFi, 2, 3, 6, 7);

	/* 2x IFFT8: multiplication by twiddle factors */
	const v4f fft8_cos_twiddle_factor = { COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4 };
	const v4f fft8_sin_twiddle_factor = { SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4 };
	v4f_mul_soa(&w4567r, &w4567i, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);
	v4f_mul_soa(&wCDEFr, &wCDEFi, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);

	/* 2x IFFT8: butterfly */
	v4f_butterfly(&w0123r, &w4567r);
	v4f_butterfly(&w89ABr, &wCDEFr);
	v4f_butterfly(&w0123i, &w4567i);
	v4f_butterfly(&w89ABi, &wCDEFi);

	/* IFFT16: multiplication by twiddle factors and scaling by 1/16 */
	v4f_mul_soa(&w89ABr, &w89ABi,
		(v4f) { COS_0PI_OVER_8 * 0.0625f, COS_1PI_OVER_8 * 0.0625f, COS_2PI_OVER_8 * 0.0625f, COS_3PI_OVER_8 * 0.0625f },
		(v4f) { SIN_0PI_OVER_8 * 0.0625f, SIN_1PI_OVER_8 * 0.0625f, SIN_2PI_OVER_8 * 0.0625f, SIN_3PI_OVER_8 * 0.0625f });
	v4f_mul_soa(&wCDEFr, &wCDEFi,
		(v4f) { COS_4PI_OVER_8 * 0.0625f, COS_5PI_OVER_8 * 0.0625f, COS_6PI_OVER_8 * 0.0625f, COS_7PI_OVER_8 * 0.0625f },
		(v4f) { SIN_4PI_OVER_8 * 0.0625f, SIN_5PI_OVER_8 * 0.0625f, SIN_6PI_OVER_8 * 0.0625f, SIN_7PI_OVER_8 * 0.0625f });

	/* IFFT16: scaling of remaining coefficients by 1/16 */
	const v4f scale = v4f_splat(0.0625f);
	w0123r *= scale;
	w0123i *= scale;
	w4567r *= scale;
	w4567i *= scale;

	/* IFFT16: butterfly */
	v4f_butterfly(&w0123r, &w89ABr);
	v4f_butterfly(&w4567r, &wCDEFr);
	v4f_butterfly(&w0123i, &w89ABi);
	v4f_butterfly(&w4567i, &wCDEFi);

	*real0123 = w0123r;
	*real4567 = w4567r;
	*real89AB = w89ABr;
	*realCDEF = wCDEFr;
	*imag0123 = w0123i;
	*imag4567 = w4567i;
	*imag89AB = w89ABi;
	*imagCDEF = wCDEFi;
}
