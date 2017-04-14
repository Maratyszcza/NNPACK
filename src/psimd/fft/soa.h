#pragma once

#include <nnpack/fft-constants.h>
#include <psimd.h>
#include <psimd/butterfly.h>


static inline void psimd_cmul_soa_f32(
	psimd_f32 real[restrict static 1],
	psimd_f32 imag[restrict static 1],
	psimd_f32 cr, psimd_f32 ci)
{
	const psimd_f32 xr = *real;
	const psimd_f32 xi = *imag;

	const psimd_f32 new_xr = xr * cr - xi * ci;
	const psimd_f32 new_xi = xi * cr + xr * ci;

	*real = new_xr;
	*imag = new_xi;
}

static inline void psimd_cmulc_soa_f32(
	psimd_f32 real[restrict static 1],
	psimd_f32 imag[restrict static 1],
	psimd_f32 cr, psimd_f32 ci)
{
	const psimd_f32 xr = *real;
	const psimd_f32 xi = *imag;

	const psimd_f32 new_xr = xr * cr + xi * ci;
	const psimd_f32 new_xi = xi * cr - xr * ci;

	*real = new_xr;
	*imag = new_xi;
}

static inline void psimd_fft8_soa_f32(
	psimd_f32 real0123[restrict static 1],
	psimd_f32 real4567[restrict static 1],
	psimd_f32 imag0123[restrict static 1],
	psimd_f32 imag4567[restrict static 1])
{
	psimd_f32 w0123r = *real0123;
	psimd_f32 w4567r = *real4567;
	psimd_f32 w0123i = *imag0123;
	psimd_f32 w4567i = *imag4567;

	/* FFT8: butterfly */
	psimd_butterfly_f32(&w0123r, &w4567r);
	psimd_butterfly_f32(&w0123i, &w4567i);

	/* FFT8: multiplication by twiddle factors */
	psimd_cmulc_soa_f32(&w4567r, &w4567i,
		(psimd_f32) { COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4 },
		(psimd_f32) { SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4 });

	/* 2x FFT4: butterfly */
	psimd_f32 w0145r = psimd_concat_lo_f32(w0123r, w4567r);
	psimd_f32 w2367r = psimd_concat_hi_f32(w0123r, w4567r);
	psimd_butterfly_f32(&w0145r, &w2367r);

	psimd_f32 w0145i = psimd_concat_lo_f32(w0123i, w4567i);
	psimd_f32 w2367i = psimd_concat_hi_f32(w0123i, w4567i);
	psimd_butterfly_f32(&w0145i, &w2367i);

	/* 2x FFT4: multiplication by twiddle factors */
	psimd_cmulc_soa_f32(&w2367r, &w2367i,
		(psimd_f32) { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 },
		(psimd_f32) { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 });

	/* 4x FFT2: butterfly */
	psimd_f32 w0426r = psimd_concat_even_f32(w0145r, w2367r);
	psimd_f32 w1537r = psimd_concat_odd_f32(w0145r, w2367r);
	psimd_butterfly_f32(&w0426r, &w1537r);

	psimd_f32 w0426i = psimd_concat_even_f32(w0145i, w2367i);
	psimd_f32 w1537i = psimd_concat_odd_f32(w0145i, w2367i);
	psimd_butterfly_f32(&w0426i, &w1537i);

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

static inline void psimd_ifft8_soa_f32(
	psimd_f32 real0123[restrict static 1],
	psimd_f32 real4567[restrict static 1],
	psimd_f32 imag0123[restrict static 1],
	psimd_f32 imag4567[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0  4  2  6  1  5  3  7
	 *   ^  ^  ^  ^  ^  ^  ^  ^
	 *   |  |  |  |  |  |  |  |
	 *   0  1  2  3  4  5  6  7
	 */
	psimd_f32 w0426r = *real0123;
	psimd_f32 w1537r = *real4567;
	psimd_f32 w0426i = *imag0123;
	psimd_f32 w1537i = *imag4567;

	/* 4x IFFT2: butterfly */
	psimd_butterfly_f32(&w0426r, &w1537r);
	psimd_f32 w0145r = psimd_interleave_lo_f32(w0426r, w1537r);
	psimd_f32 w2367r = psimd_interleave_hi_f32(w0426r, w1537r);

	psimd_butterfly_f32(&w0426i, &w1537i);
	psimd_f32 w0145i = psimd_interleave_lo_f32(w0426i, w1537i);
	psimd_f32 w2367i = psimd_interleave_hi_f32(w0426i, w1537i);

	/* 2x IFFT4: multiplication by twiddle factors */
	psimd_cmul_soa_f32(&w2367r, &w2367i,
		(psimd_f32) { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 },
		(psimd_f32) { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 });

	/* 2x IFFT4: butterfly */
	psimd_butterfly_f32(&w0145r, &w2367r);
	psimd_f32 w0123r = psimd_concat_lo_f32(w0145r, w2367r);
	psimd_f32 w4567r = psimd_concat_hi_f32(w0145r, w2367r);

	psimd_butterfly_f32(&w0145i, &w2367i);
	psimd_f32 w0123i = psimd_concat_lo_f32(w0145i, w2367i);
	psimd_f32 w4567i = psimd_concat_hi_f32(w0145i, w2367i);

	/* IFFT8: multiplication by twiddle factors and scaling by 1/8 */
	psimd_cmul_soa_f32(&w4567r, &w4567i,
		(psimd_f32) { COS_0PI_OVER_4 * 0.125f, COS_1PI_OVER_4 * 0.125f, COS_2PI_OVER_4 * 0.125f, COS_3PI_OVER_4 * 0.125f },
		(psimd_f32) { SIN_0PI_OVER_4 * 0.125f, SIN_1PI_OVER_4 * 0.125f, SIN_2PI_OVER_4 * 0.125f, SIN_3PI_OVER_4 * 0.125f });

	/* IFFT8: scaling of remaining coefficients by 1/8 */
	const psimd_f32 scale = psimd_splat_f32(0.125f);
	w0123r *= scale;
	w0123i *= scale;

	/* IFFT8: butterfly */
	psimd_butterfly_f32(&w0123r, &w4567r);
	psimd_butterfly_f32(&w0123i, &w4567i);

	*real0123 = w0123r;
	*real4567 = w4567r;
	*imag0123 = w0123i;
	*imag4567 = w4567i;
}

static inline void psimd_fft16_soa_f32(
	psimd_f32 real0123[restrict static 1],
	psimd_f32 real4567[restrict static 1],
	psimd_f32 real89AB[restrict static 1],
	psimd_f32 realCDEF[restrict static 1],
	psimd_f32 imag0123[restrict static 1],
	psimd_f32 imag4567[restrict static 1],
	psimd_f32 imag89AB[restrict static 1],
	psimd_f32 imagCDEF[restrict static 1])
{
	psimd_f32 w0123r = *real0123;
	psimd_f32 w4567r = *real4567;
	psimd_f32 w89ABr = *real89AB;
	psimd_f32 wCDEFr = *realCDEF;
	psimd_f32 w0123i = *imag0123;
	psimd_f32 w4567i = *imag4567;
	psimd_f32 w89ABi = *imag89AB;
	psimd_f32 wCDEFi = *imagCDEF;

	/* FFT16: butterfly */
	psimd_butterfly_f32(&w0123r, &w89ABr);
	psimd_butterfly_f32(&w4567r, &wCDEFr);
	psimd_butterfly_f32(&w0123i, &w89ABi);
	psimd_butterfly_f32(&w4567i, &wCDEFi);

	/* FFT16: multiplication by twiddle factors */
	psimd_cmulc_soa_f32(&w89ABr, &w89ABi,
		(psimd_f32) { COS_0PI_OVER_8, COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8 },
		(psimd_f32) { SIN_0PI_OVER_8, SIN_1PI_OVER_8, SIN_2PI_OVER_8, SIN_3PI_OVER_8 });
	psimd_cmulc_soa_f32(&wCDEFr, &wCDEFi,
		(psimd_f32) { COS_4PI_OVER_8, COS_5PI_OVER_8, COS_6PI_OVER_8, COS_7PI_OVER_8 },
		(psimd_f32) { SIN_4PI_OVER_8, SIN_5PI_OVER_8, SIN_6PI_OVER_8, SIN_7PI_OVER_8 });

	/* 2x FFT8: butterfly */
	psimd_butterfly_f32(&w0123r, &w4567r);
	psimd_butterfly_f32(&w89ABr, &wCDEFr);
	psimd_butterfly_f32(&w0123i, &w4567i);
	psimd_butterfly_f32(&w89ABi, &wCDEFi);

	/* 2x FFT8: multiplication by twiddle factors */
	const psimd_f32 fft8_cos_twiddle_factor = { COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4 };
	const psimd_f32 fft8_sin_twiddle_factor = { SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4 };
	psimd_cmulc_soa_f32(&w4567r, &w4567i, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);
	psimd_cmulc_soa_f32(&wCDEFr, &wCDEFi, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);

	/* 4x FFT4: butterfly */
	psimd_f32 w0189r = psimd_concat_lo_f32(w0123r, w89ABr);
	psimd_f32 w23ABr = psimd_concat_hi_f32(w0123r, w89ABr);
	psimd_butterfly_f32(&w0189r, &w23ABr);

	psimd_f32 w0189i = psimd_concat_lo_f32(w0123i, w89ABi);
	psimd_f32 w23ABi = psimd_concat_hi_f32(w0123i, w89ABi);
	psimd_butterfly_f32(&w0189i, &w23ABi);

	psimd_f32 w45CDr = psimd_concat_lo_f32(w4567r, wCDEFr);
	psimd_f32 w67EFr = psimd_concat_hi_f32(w4567r, wCDEFr);
	psimd_butterfly_f32(&w45CDr, &w67EFr);

	psimd_f32 w45CDi = psimd_concat_lo_f32(w4567i, wCDEFi);
	psimd_f32 w67EFi = psimd_concat_hi_f32(w4567i, wCDEFi);
	psimd_butterfly_f32(&w45CDi, &w67EFi);

	/* 4x FFT4: multiplication by twiddle factors */
	const psimd_f32 fft4_cos_twiddle_factor = { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 };
	const psimd_f32 fft4_sin_twiddle_factor = { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 };
	psimd_cmulc_soa_f32(&w23ABr, &w23ABi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);
	psimd_cmulc_soa_f32(&w67EFr, &w67EFi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);

	/* 8x FFT2: butterfly */
	psimd_f32 w084Cr = psimd_concat_even_f32(w0189r, w45CDr);
	psimd_f32 w195Dr = psimd_concat_odd_f32(w0189r, w45CDr);
	psimd_butterfly_f32(&w084Cr, &w195Dr);

	psimd_f32 w084Ci = psimd_concat_even_f32(w0189i, w45CDi);
	psimd_f32 w195Di = psimd_concat_odd_f32(w0189i, w45CDi);
	psimd_butterfly_f32(&w084Ci, &w195Di);

	psimd_f32 w2A6Er = psimd_concat_even_f32(w23ABr, w67EFr);
	psimd_f32 w3B7Fr = psimd_concat_odd_f32(w23ABr, w67EFr);
	psimd_butterfly_f32(&w2A6Er, &w3B7Fr);

	psimd_f32 w2A6Ei = psimd_concat_even_f32(w23ABi, w67EFi);
	psimd_f32 w3B7Fi = psimd_concat_odd_f32(w23ABi, w67EFi);
	psimd_butterfly_f32(&w2A6Ei, &w3B7Fi);

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

static inline void psimd_ifft16_soa_f32(
	psimd_f32 real0123[restrict static 1],
	psimd_f32 real4567[restrict static 1],
	psimd_f32 real89AB[restrict static 1],
	psimd_f32 realCDEF[restrict static 1],
	psimd_f32 imag0123[restrict static 1],
	psimd_f32 imag4567[restrict static 1],
	psimd_f32 imag89AB[restrict static 1],
	psimd_f32 imagCDEF[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0   8   4  12   2  10   6  14   1   9   5  13   3  11   7  15
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
	 */
	psimd_f32 w084Cr = *real0123;
	psimd_f32 w2A6Er = *real4567;
	psimd_f32 w195Dr = *real89AB;
	psimd_f32 w3B7Fr = *realCDEF;
	psimd_f32 w084Ci = *imag0123;
	psimd_f32 w2A6Ei = *imag4567;
	psimd_f32 w195Di = *imag89AB;
	psimd_f32 w3B7Fi = *imagCDEF;

	/* 8x IFFT2: butterfly */
	psimd_butterfly_f32(&w084Cr, &w195Dr);
	psimd_f32 w0189r = psimd_interleave_lo_f32(w084Cr, w195Dr);
	psimd_f32 w45CDr = psimd_interleave_hi_f32(w084Cr, w195Dr);

	psimd_butterfly_f32(&w084Ci, &w195Di);
	psimd_f32 w0189i = psimd_interleave_lo_f32(w084Ci, w195Di);
	psimd_f32 w45CDi = psimd_interleave_hi_f32(w084Ci, w195Di);

	psimd_butterfly_f32(&w2A6Er, &w3B7Fr);
	psimd_f32 w23ABr = psimd_interleave_lo_f32(w2A6Er, w3B7Fr);
	psimd_f32 w67EFr = psimd_interleave_hi_f32(w2A6Er, w3B7Fr);

	psimd_butterfly_f32(&w2A6Ei, &w3B7Fi);
	psimd_f32 w23ABi = psimd_interleave_lo_f32(w2A6Ei, w3B7Fi);
	psimd_f32 w67EFi = psimd_interleave_hi_f32(w2A6Ei, w3B7Fi);

	/* 4x IFFT4: multiplication by twiddle factors */
	const psimd_f32 fft4_cos_twiddle_factor = { COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2 };
	const psimd_f32 fft4_sin_twiddle_factor = { SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2 };
	psimd_cmul_soa_f32(&w23ABr, &w23ABi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);
	psimd_cmul_soa_f32(&w67EFr, &w67EFi, fft4_cos_twiddle_factor, fft4_sin_twiddle_factor);

	/* 4x IFFT4: butterfly */
	psimd_butterfly_f32(&w0189r, &w23ABr);
	psimd_f32 w0123r = psimd_concat_lo_f32(w0189r, w23ABr);
	psimd_f32 w89ABr = psimd_concat_hi_f32(w0189r, w23ABr);

	psimd_butterfly_f32(&w0189i, &w23ABi);
	psimd_f32 w0123i = psimd_concat_lo_f32(w0189i, w23ABi);
	psimd_f32 w89ABi = psimd_concat_hi_f32(w0189i, w23ABi);

	psimd_butterfly_f32(&w45CDr, &w67EFr);
	psimd_f32 w4567r = psimd_concat_lo_f32(w45CDr, w67EFr);
	psimd_f32 wCDEFr = psimd_concat_hi_f32(w45CDr, w67EFr);

	psimd_butterfly_f32(&w45CDi, &w67EFi);
	psimd_f32 w4567i = psimd_concat_lo_f32(w45CDi, w67EFi);
	psimd_f32 wCDEFi = psimd_concat_hi_f32(w45CDi, w67EFi);

	/* 2x IFFT8: multiplication by twiddle factors */
	const psimd_f32 fft8_cos_twiddle_factor = { COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4 };
	const psimd_f32 fft8_sin_twiddle_factor = { SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4 };
	psimd_cmul_soa_f32(&w4567r, &w4567i, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);
	psimd_cmul_soa_f32(&wCDEFr, &wCDEFi, fft8_cos_twiddle_factor, fft8_sin_twiddle_factor);

	/* 2x IFFT8: butterfly */
	psimd_butterfly_f32(&w0123r, &w4567r);
	psimd_butterfly_f32(&w89ABr, &wCDEFr);
	psimd_butterfly_f32(&w0123i, &w4567i);
	psimd_butterfly_f32(&w89ABi, &wCDEFi);

	/* IFFT16: multiplication by twiddle factors and scaling by 1/16 */
	psimd_cmul_soa_f32(&w89ABr, &w89ABi,
		(psimd_f32) { COS_0PI_OVER_8 * 0.0625f, COS_1PI_OVER_8 * 0.0625f, COS_2PI_OVER_8 * 0.0625f, COS_3PI_OVER_8 * 0.0625f },
		(psimd_f32) { SIN_0PI_OVER_8 * 0.0625f, SIN_1PI_OVER_8 * 0.0625f, SIN_2PI_OVER_8 * 0.0625f, SIN_3PI_OVER_8 * 0.0625f });
	psimd_cmul_soa_f32(&wCDEFr, &wCDEFi,
		(psimd_f32) { COS_4PI_OVER_8 * 0.0625f, COS_5PI_OVER_8 * 0.0625f, COS_6PI_OVER_8 * 0.0625f, COS_7PI_OVER_8 * 0.0625f },
		(psimd_f32) { SIN_4PI_OVER_8 * 0.0625f, SIN_5PI_OVER_8 * 0.0625f, SIN_6PI_OVER_8 * 0.0625f, SIN_7PI_OVER_8 * 0.0625f });

	/* IFFT16: scaling of remaining coefficients by 1/16 */
	const psimd_f32 scale = psimd_splat_f32(0.0625f);
	w0123r *= scale;
	w0123i *= scale;
	w4567r *= scale;
	w4567i *= scale;

	/* IFFT16: butterfly */
	psimd_butterfly_f32(&w0123r, &w89ABr);
	psimd_butterfly_f32(&w4567r, &wCDEFr);
	psimd_butterfly_f32(&w0123i, &w89ABi);
	psimd_butterfly_f32(&w4567i, &wCDEFi);

	*real0123 = w0123r;
	*real4567 = w4567r;
	*real89AB = w89ABr;
	*realCDEF = wCDEFr;
	*imag0123 = w0123i;
	*imag4567 = w4567i;
	*imag89AB = w89ABi;
	*imagCDEF = wCDEFi;
}
