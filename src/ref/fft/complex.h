#pragma once

#include <nnpack/fft-constants.h>


/* Android's libm doesn't even have conjf (!) */
#ifdef __ANDROID__
	inline static float _Complex nnp_conjf(float _Complex x) {
		float _Complex y = x;
		__imag y = - __imag y;
		return y;
	}

	#define conjf(x) nnp_conjf(x)
#endif


inline static void butterflyfc(float _Complex a[restrict static 1], float _Complex b[restrict static 1]) {
	const float _Complex new_a = *a + *b;
	const float _Complex new_b = *a - *b;
	*a = new_a;
	*b = new_b;
}

inline static void swapfc(float _Complex a[restrict static 1], float _Complex b[restrict static 1]) {
	const float _Complex new_a = *b;
	const float _Complex new_b = *a;
	*a = new_a;
	*b = new_b;
}

inline static void fft2fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1])
{
	/* FFT2: butterfly */
	butterflyfc(w0, w1);
}

inline static void ifft2fc(
	float _Complex W0[restrict static 1],
	float _Complex W1[restrict static 1])
{
	/* IFFT2: butterfly */
	butterflyfc(W0, W1);

	/* Scale */
	*W0 *= 0.5f;
	*W1 *= 0.5f;
}

inline static void fft4fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1])
{
	/* FFT4: butterfly */
	butterflyfc(w0, w2);
	butterflyfc(w1, w3);

	/* FFT4: multiplication by twiddle factors */
	*w3 *= conjf(CEXP_1PI_OVER_2);

	/* 2x FFT2: butterfly */
	butterflyfc(w0, w1);
	butterflyfc(w2, w3);

	/*
	 * Bit reversal:
	 *   0  2  1  3
	 *   ^  ^  ^  ^
	 *   |  |  |  |
	 *   0  1  2  3
	 */
	swapfc(w1, w2);
}

inline static void ifft4fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0  1  2  3
	 *   ^  ^  ^  ^
	 *   |  |  |  |
	 *   0  2  1  3
	 */
	swapfc(w1, w2);

	/* 2x IFFT2: butterfly */
	butterflyfc(w0, w1);
	butterflyfc(w2, w3);

	/* IFFT4: multiplication by twiddle factors */
	*w3 *= CEXP_1PI_OVER_2;

	/* IFFT4: butterfly */
	butterflyfc(w0, w2);
	butterflyfc(w1, w3);

	/* Scale */
	*w0 *= 0.25f;
	*w1 *= 0.25f;
	*w2 *= 0.25f;
	*w3 *= 0.25f;
}

inline static void fft8fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1],
	float _Complex w4[restrict static 1],
	float _Complex w5[restrict static 1],
	float _Complex w6[restrict static 1],
	float _Complex w7[restrict static 1])
{
	/* FFT8: butterfly */
	butterflyfc(w0, w4);
	butterflyfc(w1, w5);
	butterflyfc(w2, w6);
	butterflyfc(w3, w7);

	/* FFT4: multiplication by twiddle factors */
	*w5 *= conjf(CEXP_1PI_OVER_4);
	*w6 *= conjf(CEXP_2PI_OVER_4);
	*w7 *= conjf(CEXP_3PI_OVER_4);

	/* 2x FFT4: butterfly */
	butterflyfc(w0, w2);
	butterflyfc(w1, w3);

	butterflyfc(w4, w6);
	butterflyfc(w5, w7);

	/* 2x FFT4: multiplication by twiddle factors */
	*w3 *= conjf(CEXP_1PI_OVER_2);
	*w7 *= conjf(CEXP_1PI_OVER_2);

	/* 4x FFT2: butterfly */
	butterflyfc(w0, w1);
	butterflyfc(w2, w3);
	butterflyfc(w4, w5);
	butterflyfc(w6, w7);

	/*
	 * Bit reversal:
	 *   0  4  2  6  1  5  3  7
	 *   ^  ^  ^  ^  ^  ^  ^  ^
	 *   |  |  |  |  |  |  |  |
	 *   0  1  2  3  4  5  6  7
	 */
	swapfc(w1, w4);
	swapfc(w3, w6);
}

inline static void ifft8fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1],
	float _Complex w4[restrict static 1],
	float _Complex w5[restrict static 1],
	float _Complex w6[restrict static 1],
	float _Complex w7[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0  1  2  3  4  5  6  7
	 *   ^  ^  ^  ^  ^  ^  ^  ^
	 *   |  |  |  |  |  |  |  |
	 *   0  4  2  6  1  5  3  7
	 */
	swapfc(w1, w4);
	swapfc(w3, w6);

	/* 4x IFFT2: butterfly */
	butterflyfc(w0, w1);
	butterflyfc(w2, w3);
	butterflyfc(w4, w5);
	butterflyfc(w6, w7);

	/* 2x IFFT4: multiplication by twiddle factors */
	*w3 *= CEXP_1PI_OVER_2;
	*w7 *= CEXP_1PI_OVER_2;

	/* 2x IFFT4: butterfly */
	butterflyfc(w0, w2);
	butterflyfc(w1, w3);
	butterflyfc(w4, w6);
	butterflyfc(w5, w7);

	/* IFFT4: multiplication by twiddle factors */
	*w5 *= CEXP_1PI_OVER_4;
	*w6 *= CEXP_2PI_OVER_4;
	*w7 *= CEXP_3PI_OVER_4;

	/* IFFT8: butterfly */
	butterflyfc(w0, w4);
	butterflyfc(w1, w5);
	butterflyfc(w2, w6);
	butterflyfc(w3, w7);


	/* Scale */
	*w0 *= 0.125f;
	*w1 *= 0.125f;
	*w2 *= 0.125f;
	*w3 *= 0.125f;
	*w4 *= 0.125f;
	*w5 *= 0.125f;
	*w6 *= 0.125f;
	*w7 *= 0.125f;
}

inline static void fft16fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1],
	float _Complex w4[restrict static 1],
	float _Complex w5[restrict static 1],
	float _Complex w6[restrict static 1],
	float _Complex w7[restrict static 1],
	float _Complex w8[restrict static 1],
	float _Complex w9[restrict static 1],
	float _Complex w10[restrict static 1],
	float _Complex w11[restrict static 1],
	float _Complex w12[restrict static 1],
	float _Complex w13[restrict static 1],
	float _Complex w14[restrict static 1],
	float _Complex w15[restrict static 1])
{
	/* FFT16: butterfly */
	butterflyfc(w0, w8);
	butterflyfc(w1, w9);
	butterflyfc(w2, w10);
	butterflyfc(w3, w11);
	butterflyfc(w4, w12);
	butterflyfc(w5, w13);
	butterflyfc(w6, w14);
	butterflyfc(w7, w15);

	/* FFT16: multiplication by twiddle factors */
	*w9  *= conjf(CEXP_1PI_OVER_8);
	*w10 *= conjf(CEXP_2PI_OVER_8);
	*w11 *= conjf(CEXP_3PI_OVER_8);
	*w12 *= conjf(CEXP_4PI_OVER_8);
	*w13 *= conjf(CEXP_5PI_OVER_8);
	*w14 *= conjf(CEXP_6PI_OVER_8);
	*w15 *= conjf(CEXP_7PI_OVER_8);

	/* 2x FFT8: butterfly */
	butterflyfc(w0,  w4);
	butterflyfc(w1,  w5);
	butterflyfc(w2,  w6);
	butterflyfc(w3,  w7);

	butterflyfc(w8,  w12);
	butterflyfc(w9,  w13);
	butterflyfc(w10, w14);
	butterflyfc(w11, w15);

	/* 2x FFT8: multiplication by twiddle factors */
	*w5  *= conjf(CEXP_1PI_OVER_4);
	*w6  *= conjf(CEXP_2PI_OVER_4);
	*w7  *= conjf(CEXP_3PI_OVER_4);

	*w13 *= conjf(CEXP_1PI_OVER_4);
	*w14 *= conjf(CEXP_2PI_OVER_4);
	*w15 *= conjf(CEXP_3PI_OVER_4);

	/* 4x FFT4: butterfly */
	butterflyfc(w0,  w2);
	butterflyfc(w1,  w3);

	butterflyfc(w4,  w6);
	butterflyfc(w5,  w7);

	butterflyfc(w8,  w10);
	butterflyfc(w9,  w11);

	butterflyfc(w12, w14);
	butterflyfc(w13, w15);

	/* 4x FFT4: multiplication by twiddle factors */
	*w3  *= conjf(CEXP_1PI_OVER_2);
	*w7  *= conjf(CEXP_1PI_OVER_2);
	*w11 *= conjf(CEXP_1PI_OVER_2);
	*w15 *= conjf(CEXP_1PI_OVER_2);

	/* 8x FFT2: butterfly */
	butterflyfc(w0,  w1);
	butterflyfc(w2,  w3);
	butterflyfc(w4,  w5);
	butterflyfc(w6,  w7);
	butterflyfc(w8,  w9);
	butterflyfc(w10, w11);
	butterflyfc(w12, w13);
	butterflyfc(w14, w15);

	/*
	 * Bit reversal:
	 *   0   8   4  12   2  10   6  14   1   9   5  13   3  11   7  15
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
	 */
	swapfc(w1, w8);
	swapfc(w2, w4);
	swapfc(w3, w12);
	swapfc(w5, w10);
	swapfc(w7, w14);
	swapfc(w11, w13);
}

inline static void ifft16fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1],
	float _Complex w4[restrict static 1],
	float _Complex w5[restrict static 1],
	float _Complex w6[restrict static 1],
	float _Complex w7[restrict static 1],
	float _Complex w8[restrict static 1],
	float _Complex w9[restrict static 1],
	float _Complex w10[restrict static 1],
	float _Complex w11[restrict static 1],
	float _Complex w12[restrict static 1],
	float _Complex w13[restrict static 1],
	float _Complex w14[restrict static 1],
	float _Complex w15[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0   8   4  12   2  10   6  14   1   9   5  13   3  11   7  15
	 */
	swapfc(w1, w8);
	swapfc(w2, w4);
	swapfc(w3, w12);
	swapfc(w5, w10);
	swapfc(w7, w14);
	swapfc(w11, w13);

	/* 8x IFFT2: butterfly */
	butterflyfc(w0,  w1);
	butterflyfc(w2,  w3);
	butterflyfc(w4,  w5);
	butterflyfc(w6,  w7);
	butterflyfc(w8,  w9);
	butterflyfc(w10, w11);
	butterflyfc(w12, w13);
	butterflyfc(w14, w15);

	/* 4x IFFT4: multiplication by twiddle factors */
	*w3  *= CEXP_1PI_OVER_2;
	*w7  *= CEXP_1PI_OVER_2;
	*w11 *= CEXP_1PI_OVER_2;
	*w15 *= CEXP_1PI_OVER_2;

	/* 4x IFFT4: butterfly */
	butterflyfc(w0,  w2);
	butterflyfc(w1,  w3);

	butterflyfc(w4,  w6);
	butterflyfc(w5,  w7);

	butterflyfc(w8,  w10);
	butterflyfc(w9,  w11);

	butterflyfc(w12, w14);
	butterflyfc(w13, w15);

	/* 2x IFFT8: multiplication by twiddle factors */
	*w5  *= CEXP_1PI_OVER_4;
	*w6  *= CEXP_2PI_OVER_4;
	*w7  *= CEXP_3PI_OVER_4;

	*w13 *= CEXP_1PI_OVER_4;
	*w14 *= CEXP_2PI_OVER_4;
	*w15 *= CEXP_3PI_OVER_4;

	/* 2x IFFT8: butterfly */
	butterflyfc(w0,  w4);
	butterflyfc(w1,  w5);
	butterflyfc(w2,  w6);
	butterflyfc(w3,  w7);

	butterflyfc(w8,  w12);
	butterflyfc(w9,  w13);
	butterflyfc(w10, w14);
	butterflyfc(w11, w15);

	/* IFFT16: multiplication by twiddle factors */
	*w9  *= CEXP_1PI_OVER_8;
	*w10 *= CEXP_2PI_OVER_8;
	*w11 *= CEXP_3PI_OVER_8;
	*w12 *= CEXP_4PI_OVER_8;
	*w13 *= CEXP_5PI_OVER_8;
	*w14 *= CEXP_6PI_OVER_8;
	*w15 *= CEXP_7PI_OVER_8;

	/* IFFT16: butterfly */
	butterflyfc(w0, w8);
	butterflyfc(w1, w9);
	butterflyfc(w2, w10);
	butterflyfc(w3, w11);
	butterflyfc(w4, w12);
	butterflyfc(w5, w13);
	butterflyfc(w6, w14);
	butterflyfc(w7, w15);

	/* Scale */
	*w0  *= 0.0625f;
	*w1  *= 0.0625f;
	*w2  *= 0.0625f;
	*w3  *= 0.0625f;
	*w4  *= 0.0625f;
	*w5  *= 0.0625f;
	*w6  *= 0.0625f;
	*w7  *= 0.0625f;
	*w8  *= 0.0625f;
	*w9  *= 0.0625f;
	*w10 *= 0.0625f;
	*w11 *= 0.0625f;
	*w12 *= 0.0625f;
	*w13 *= 0.0625f;
	*w14 *= 0.0625f;
	*w15 *= 0.0625f;
}

inline static void fft32fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1],
	float _Complex w4[restrict static 1],
	float _Complex w5[restrict static 1],
	float _Complex w6[restrict static 1],
	float _Complex w7[restrict static 1],
	float _Complex w8[restrict static 1],
	float _Complex w9[restrict static 1],
	float _Complex w10[restrict static 1],
	float _Complex w11[restrict static 1],
	float _Complex w12[restrict static 1],
	float _Complex w13[restrict static 1],
	float _Complex w14[restrict static 1],
	float _Complex w15[restrict static 1],
	float _Complex w16[restrict static 1],
	float _Complex w17[restrict static 1],
	float _Complex w18[restrict static 1],
	float _Complex w19[restrict static 1],
	float _Complex w20[restrict static 1],
	float _Complex w21[restrict static 1],
	float _Complex w22[restrict static 1],
	float _Complex w23[restrict static 1],
	float _Complex w24[restrict static 1],
	float _Complex w25[restrict static 1],
	float _Complex w26[restrict static 1],
	float _Complex w27[restrict static 1],
	float _Complex w28[restrict static 1],
	float _Complex w29[restrict static 1],
	float _Complex w30[restrict static 1],
	float _Complex w31[restrict static 1])
{
	/* FFT32: butterfly */
	butterflyfc(w0,  w16);
	butterflyfc(w1,  w17);
	butterflyfc(w2,  w18);
	butterflyfc(w3,  w19);
	butterflyfc(w4,  w20);
	butterflyfc(w5,  w21);
	butterflyfc(w6,  w22);
	butterflyfc(w7,  w23);
	butterflyfc(w8,  w24);
	butterflyfc(w9,  w25);
	butterflyfc(w10, w26);
	butterflyfc(w11, w27);
	butterflyfc(w12, w28);
	butterflyfc(w13, w29);
	butterflyfc(w14, w30);
	butterflyfc(w15, w31);

	/* FFT32: multiplication by twiddle factors */
	*w17 *= conjf(CEXP__1PI_OVER_16);
	*w18 *= conjf(CEXP__2PI_OVER_16);
	*w19 *= conjf(CEXP__3PI_OVER_16);
	*w20 *= conjf(CEXP__4PI_OVER_16);
	*w21 *= conjf(CEXP__5PI_OVER_16);
	*w22 *= conjf(CEXP__6PI_OVER_16);
	*w23 *= conjf(CEXP__7PI_OVER_16);
	*w24 *= conjf(CEXP__8PI_OVER_16);
	*w25 *= conjf(CEXP__9PI_OVER_16);
	*w26 *= conjf(CEXP_10PI_OVER_16);
	*w27 *= conjf(CEXP_11PI_OVER_16);
	*w28 *= conjf(CEXP_12PI_OVER_16);
	*w29 *= conjf(CEXP_13PI_OVER_16);
	*w30 *= conjf(CEXP_14PI_OVER_16);
	*w31 *= conjf(CEXP_15PI_OVER_16);

	/* 2x FFT16: butterfly */
	butterflyfc(w0,  w8);
	butterflyfc(w1,  w9);
	butterflyfc(w2,  w10);
	butterflyfc(w3,  w11);
	butterflyfc(w4,  w12);
	butterflyfc(w5,  w13);
	butterflyfc(w6,  w14);
	butterflyfc(w7,  w15);

	butterflyfc(w16, w24);
	butterflyfc(w17, w25);
	butterflyfc(w18, w26);
	butterflyfc(w19, w27);
	butterflyfc(w20, w28);
	butterflyfc(w21, w29);
	butterflyfc(w22, w30);
	butterflyfc(w23, w31);

	/* 2x FFT16: multiplication by twiddle factors */
	*w9  *= conjf(CEXP_1PI_OVER_8);
	*w10 *= conjf(CEXP_2PI_OVER_8);
	*w11 *= conjf(CEXP_3PI_OVER_8);
	*w12 *= conjf(CEXP_4PI_OVER_8);
	*w13 *= conjf(CEXP_5PI_OVER_8);
	*w14 *= conjf(CEXP_6PI_OVER_8);
	*w15 *= conjf(CEXP_7PI_OVER_8);

	*w25 *= conjf(CEXP_1PI_OVER_8);
	*w26 *= conjf(CEXP_2PI_OVER_8);
	*w27 *= conjf(CEXP_3PI_OVER_8);
	*w28 *= conjf(CEXP_4PI_OVER_8);
	*w29 *= conjf(CEXP_5PI_OVER_8);
	*w30 *= conjf(CEXP_6PI_OVER_8);
	*w31 *= conjf(CEXP_7PI_OVER_8);

	/* 4x FFT8: butterfly */
	butterflyfc(w0,  w4);
	butterflyfc(w1,  w5);
	butterflyfc(w2,  w6);
	butterflyfc(w3,  w7);

	butterflyfc(w8,  w12);
	butterflyfc(w9,  w13);
	butterflyfc(w10, w14);
	butterflyfc(w11, w15);

	butterflyfc(w16, w20);
	butterflyfc(w17, w21);
	butterflyfc(w18, w22);
	butterflyfc(w19, w23);

	butterflyfc(w24, w28);
	butterflyfc(w25, w29);
	butterflyfc(w26, w30);
	butterflyfc(w27, w31);

	/* 4x FFT8: multiplication by twiddle factors */
	*w5  *= conjf(CEXP_1PI_OVER_4);
	*w6  *= conjf(CEXP_2PI_OVER_4);
	*w7  *= conjf(CEXP_3PI_OVER_4);

	*w13 *= conjf(CEXP_1PI_OVER_4);
	*w14 *= conjf(CEXP_2PI_OVER_4);
	*w15 *= conjf(CEXP_3PI_OVER_4);

	*w21 *= conjf(CEXP_1PI_OVER_4);
	*w22 *= conjf(CEXP_2PI_OVER_4);
	*w23 *= conjf(CEXP_3PI_OVER_4);

	*w29 *= conjf(CEXP_1PI_OVER_4);
	*w30 *= conjf(CEXP_2PI_OVER_4);
	*w31 *= conjf(CEXP_3PI_OVER_4);

	/* 8x FFT4: butterfly */
	butterflyfc(w0,  w2);
	butterflyfc(w1,  w3);

	butterflyfc(w4,  w6);
	butterflyfc(w5,  w7);

	butterflyfc(w8,  w10);
	butterflyfc(w9,  w11);

	butterflyfc(w12, w14);
	butterflyfc(w13, w15);

	butterflyfc(w16, w18);
	butterflyfc(w17, w19);

	butterflyfc(w20, w22);
	butterflyfc(w21, w23);

	butterflyfc(w24, w26);
	butterflyfc(w25, w27);

	butterflyfc(w28, w30);
	butterflyfc(w29, w31);

	/* 8x FFT4: multiplication by twiddle factors */
	*w3  *= conjf(CEXP_1PI_OVER_2);
	*w7  *= conjf(CEXP_1PI_OVER_2);
	*w11 *= conjf(CEXP_1PI_OVER_2);
	*w15 *= conjf(CEXP_1PI_OVER_2);
	*w19 *= conjf(CEXP_1PI_OVER_2);
	*w23 *= conjf(CEXP_1PI_OVER_2);
	*w27 *= conjf(CEXP_1PI_OVER_2);
	*w31 *= conjf(CEXP_1PI_OVER_2);

	/* 16x FFT2: butterfly */
	butterflyfc(w0,  w1);
	butterflyfc(w2,  w3);
	butterflyfc(w4,  w5);
	butterflyfc(w6,  w7);
	butterflyfc(w8,  w9);
	butterflyfc(w10, w11);
	butterflyfc(w12, w13);
	butterflyfc(w14, w15);
	butterflyfc(w16, w17);
	butterflyfc(w18, w19);
	butterflyfc(w20, w21);
	butterflyfc(w22, w23);
	butterflyfc(w24, w25);
	butterflyfc(w26, w27);
	butterflyfc(w28, w29);
	butterflyfc(w30, w31);

	/*
	 * Bit reversal:
	 *   0  16   8  24   4  20  12  28   2  18  10  26   6  22  14  30   1  17   9  25   5  21  13  29   3  19  11  27   7  23  15  31
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
	 */
	swapfc(w1, w16);
	swapfc(w2, w8);
	swapfc(w3, w24);
	swapfc(w5, w20);
	swapfc(w6, w12);
	swapfc(w7, w28);
	swapfc(w9, w18);
	swapfc(w11, w26);
	swapfc(w13, w22);
	swapfc(w15, w30);
	swapfc(w19, w25);
	swapfc(w23, w29);
}

inline static void ifft32fc(
	float _Complex w0[restrict static 1],
	float _Complex w1[restrict static 1],
	float _Complex w2[restrict static 1],
	float _Complex w3[restrict static 1],
	float _Complex w4[restrict static 1],
	float _Complex w5[restrict static 1],
	float _Complex w6[restrict static 1],
	float _Complex w7[restrict static 1],
	float _Complex w8[restrict static 1],
	float _Complex w9[restrict static 1],
	float _Complex w10[restrict static 1],
	float _Complex w11[restrict static 1],
	float _Complex w12[restrict static 1],
	float _Complex w13[restrict static 1],
	float _Complex w14[restrict static 1],
	float _Complex w15[restrict static 1],
	float _Complex w16[restrict static 1],
	float _Complex w17[restrict static 1],
	float _Complex w18[restrict static 1],
	float _Complex w19[restrict static 1],
	float _Complex w20[restrict static 1],
	float _Complex w21[restrict static 1],
	float _Complex w22[restrict static 1],
	float _Complex w23[restrict static 1],
	float _Complex w24[restrict static 1],
	float _Complex w25[restrict static 1],
	float _Complex w26[restrict static 1],
	float _Complex w27[restrict static 1],
	float _Complex w28[restrict static 1],
	float _Complex w29[restrict static 1],
	float _Complex w30[restrict static 1],
	float _Complex w31[restrict static 1])
{
	/*
	 * Bit reversal:
	 *   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
	 *   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^
	 *   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
	 *   0  16   8  24   4  20  12  28   2  18  10  26   6  22  14  30   1  17   9  25   5  21  13  29   3  19  11  27   7  23  15  31
	 */
	swapfc(w1, w16);
	swapfc(w2, w8);
	swapfc(w3, w24);
	swapfc(w5, w20);
	swapfc(w6, w12);
	swapfc(w7, w28);
	swapfc(w9, w18);
	swapfc(w11, w26);
	swapfc(w13, w22);
	swapfc(w15, w30);
	swapfc(w19, w25);
	swapfc(w23, w29);

	/* 16x IFFT2: butterfly */
	butterflyfc(w0,  w1);
	butterflyfc(w2,  w3);
	butterflyfc(w4,  w5);
	butterflyfc(w6,  w7);
	butterflyfc(w8,  w9);
	butterflyfc(w10, w11);
	butterflyfc(w12, w13);
	butterflyfc(w14, w15);
	butterflyfc(w16, w17);
	butterflyfc(w18, w19);
	butterflyfc(w20, w21);
	butterflyfc(w22, w23);
	butterflyfc(w24, w25);
	butterflyfc(w26, w27);
	butterflyfc(w28, w29);
	butterflyfc(w30, w31);

	/* 8x IFFT4: multiplication by twiddle factors */
	*w3  *= CEXP_1PI_OVER_2;
	*w7  *= CEXP_1PI_OVER_2;
	*w11 *= CEXP_1PI_OVER_2;
	*w15 *= CEXP_1PI_OVER_2;
	*w19 *= CEXP_1PI_OVER_2;
	*w23 *= CEXP_1PI_OVER_2;
	*w27 *= CEXP_1PI_OVER_2;
	*w31 *= CEXP_1PI_OVER_2;

	/* 8x IFFT4: butterfly */
	butterflyfc(w0,  w2);
	butterflyfc(w1,  w3);

	butterflyfc(w4,  w6);
	butterflyfc(w5,  w7);

	butterflyfc(w8,  w10);
	butterflyfc(w9,  w11);

	butterflyfc(w12, w14);
	butterflyfc(w13, w15);

	butterflyfc(w16, w18);
	butterflyfc(w17, w19);

	butterflyfc(w20, w22);
	butterflyfc(w21, w23);

	butterflyfc(w24, w26);
	butterflyfc(w25, w27);

	butterflyfc(w28, w30);
	butterflyfc(w29, w31);

	/* 4x IFFT8: multiplication by twiddle factors */
	*w5  *= CEXP_1PI_OVER_4;
	*w6  *= CEXP_2PI_OVER_4;
	*w7  *= CEXP_3PI_OVER_4;

	*w13 *= CEXP_1PI_OVER_4;
	*w14 *= CEXP_2PI_OVER_4;
	*w15 *= CEXP_3PI_OVER_4;

	*w21 *= CEXP_1PI_OVER_4;
	*w22 *= CEXP_2PI_OVER_4;
	*w23 *= CEXP_3PI_OVER_4;

	*w29 *= CEXP_1PI_OVER_4;
	*w30 *= CEXP_2PI_OVER_4;
	*w31 *= CEXP_3PI_OVER_4;

	/* 4x IFFT8: butterfly */
	butterflyfc(w0,  w4);
	butterflyfc(w1,  w5);
	butterflyfc(w2,  w6);
	butterflyfc(w3,  w7);

	butterflyfc(w8,  w12);
	butterflyfc(w9,  w13);
	butterflyfc(w10, w14);
	butterflyfc(w11, w15);

	butterflyfc(w16, w20);
	butterflyfc(w17, w21);
	butterflyfc(w18, w22);
	butterflyfc(w19, w23);

	butterflyfc(w24, w28);
	butterflyfc(w25, w29);
	butterflyfc(w26, w30);
	butterflyfc(w27, w31);

	/* 2x IFFT16: multiplication by twiddle factors */
	*w9  *= CEXP_1PI_OVER_8;
	*w10 *= CEXP_2PI_OVER_8;
	*w11 *= CEXP_3PI_OVER_8;
	*w12 *= CEXP_4PI_OVER_8;
	*w13 *= CEXP_5PI_OVER_8;
	*w14 *= CEXP_6PI_OVER_8;
	*w15 *= CEXP_7PI_OVER_8;

	*w25 *= CEXP_1PI_OVER_8;
	*w26 *= CEXP_2PI_OVER_8;
	*w27 *= CEXP_3PI_OVER_8;
	*w28 *= CEXP_4PI_OVER_8;
	*w29 *= CEXP_5PI_OVER_8;
	*w30 *= CEXP_6PI_OVER_8;
	*w31 *= CEXP_7PI_OVER_8;

	/* 2x IFFT16: butterfly */
	butterflyfc(w0,  w8);
	butterflyfc(w1,  w9);
	butterflyfc(w2,  w10);
	butterflyfc(w3,  w11);
	butterflyfc(w4,  w12);
	butterflyfc(w5,  w13);
	butterflyfc(w6,  w14);
	butterflyfc(w7,  w15);

	butterflyfc(w16, w24);
	butterflyfc(w17, w25);
	butterflyfc(w18, w26);
	butterflyfc(w19, w27);
	butterflyfc(w20, w28);
	butterflyfc(w21, w29);
	butterflyfc(w22, w30);
	butterflyfc(w23, w31);

	/* IFFT32: multiplication by twiddle factors */
	*w17 *= CEXP__1PI_OVER_16;
	*w18 *= CEXP__2PI_OVER_16;
	*w19 *= CEXP__3PI_OVER_16;
	*w20 *= CEXP__4PI_OVER_16;
	*w21 *= CEXP__5PI_OVER_16;
	*w22 *= CEXP__6PI_OVER_16;
	*w23 *= CEXP__7PI_OVER_16;
	*w24 *= CEXP__8PI_OVER_16;
	*w25 *= CEXP__9PI_OVER_16;
	*w26 *= CEXP_10PI_OVER_16;
	*w27 *= CEXP_11PI_OVER_16;
	*w28 *= CEXP_12PI_OVER_16;
	*w29 *= CEXP_13PI_OVER_16;
	*w30 *= CEXP_14PI_OVER_16;
	*w31 *= CEXP_15PI_OVER_16;

	/* IFFT32: butterfly */
	butterflyfc(w0,  w16);
	butterflyfc(w1,  w17);
	butterflyfc(w2,  w18);
	butterflyfc(w3,  w19);
	butterflyfc(w4,  w20);
	butterflyfc(w5,  w21);
	butterflyfc(w6,  w22);
	butterflyfc(w7,  w23);
	butterflyfc(w8,  w24);
	butterflyfc(w9,  w25);
	butterflyfc(w10, w26);
	butterflyfc(w11, w27);
	butterflyfc(w12, w28);
	butterflyfc(w13, w29);
	butterflyfc(w14, w30);
	butterflyfc(w15, w31);

	/* Scale */
	*w0  *= 0.03125f;
	*w1  *= 0.03125f;
	*w2  *= 0.03125f;
	*w3  *= 0.03125f;
	*w4  *= 0.03125f;
	*w5  *= 0.03125f;
	*w6  *= 0.03125f;
	*w7  *= 0.03125f;
	*w8  *= 0.03125f;
	*w9  *= 0.03125f;
	*w10 *= 0.03125f;
	*w11 *= 0.03125f;
	*w12 *= 0.03125f;
	*w13 *= 0.03125f;
	*w14 *= 0.03125f;
	*w15 *= 0.03125f;
	*w16 *= 0.03125f;
	*w17 *= 0.03125f;
	*w18 *= 0.03125f;
	*w19 *= 0.03125f;
	*w20 *= 0.03125f;
	*w21 *= 0.03125f;
	*w22 *= 0.03125f;
	*w23 *= 0.03125f;
	*w24 *= 0.03125f;
	*w25 *= 0.03125f;
	*w26 *= 0.03125f;
	*w27 *= 0.03125f;
	*w28 *= 0.03125f;
	*w29 *= 0.03125f;
	*w30 *= 0.03125f;
	*w31 *= 0.03125f;
}
