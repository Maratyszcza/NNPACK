#include <stddef.h>

#include <nnpack/complex.h>
#include <fft/complex.h>


void nnp_fft2_aos__ref(
	const float t[restrict static 4], size_t t_stride,
	float f[restrict static 4], size_t f_stride)
{
	/* Load inputs */
	float _Complex w0 = CMPLXF(t[0 * t_stride], t[1 * t_stride]);
	float _Complex w1 = CMPLXF(t[2 * t_stride], t[3 * t_stride]);

	fft2fc(&w0, &w1);

	/* Store outputs */
	f[0 * f_stride] = crealf(w0);
	f[1 * f_stride] = cimagf(w0);
	f[2 * f_stride] = crealf(w1);
	f[3 * f_stride] = cimagf(w1);
}

void nnp_fft4_aos__ref(
	const float t[restrict static 8], size_t t_stride,
	float f[restrict static 8], size_t f_stride)
{
	/* Load inputs */
	float _Complex w0 = CMPLXF(t[0 * t_stride], t[1 * t_stride]);
	float _Complex w1 = CMPLXF(t[2 * t_stride], t[3 * t_stride]);
	float _Complex w2 = CMPLXF(t[4 * t_stride], t[5 * t_stride]);
	float _Complex w3 = CMPLXF(t[6 * t_stride], t[7 * t_stride]);

	fft4fc(&w0, &w1, &w2, &w3);

	/* Store outputs */
	f[0 * f_stride] = crealf(w0);
	f[1 * f_stride] = cimagf(w0);
	f[2 * f_stride] = crealf(w1);
	f[3 * f_stride] = cimagf(w1);
	f[4 * f_stride] = crealf(w2);
	f[5 * f_stride] = cimagf(w2);
	f[6 * f_stride] = crealf(w3);
	f[7 * f_stride] = cimagf(w3);
}

void nnp_fft8_aos__ref(
	const float t[restrict static 16], size_t t_stride,
	float f[restrict static 16], size_t f_stride)
{
	/* Load inputs */
	float _Complex w0 = CMPLXF(t[ 0 * t_stride], t[ 1 * t_stride]);
	float _Complex w1 = CMPLXF(t[ 2 * t_stride], t[ 3 * t_stride]);
	float _Complex w2 = CMPLXF(t[ 4 * t_stride], t[ 5 * t_stride]);
	float _Complex w3 = CMPLXF(t[ 6 * t_stride], t[ 7 * t_stride]);
	float _Complex w4 = CMPLXF(t[ 8 * t_stride], t[ 9 * t_stride]);
	float _Complex w5 = CMPLXF(t[10 * t_stride], t[11 * t_stride]);
	float _Complex w6 = CMPLXF(t[12 * t_stride], t[13 * t_stride]);
	float _Complex w7 = CMPLXF(t[14 * t_stride], t[15 * t_stride]);

	fft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	/* Store outputs */
	f[ 0 * f_stride] = crealf(w0);
	f[ 1 * f_stride] = cimagf(w0);
	f[ 2 * f_stride] = crealf(w1);
	f[ 3 * f_stride] = cimagf(w1);
	f[ 4 * f_stride] = crealf(w2);
	f[ 5 * f_stride] = cimagf(w2);
	f[ 6 * f_stride] = crealf(w3);
	f[ 7 * f_stride] = cimagf(w3);
	f[ 8 * f_stride] = crealf(w4);
	f[ 9 * f_stride] = cimagf(w4);
	f[10 * f_stride] = crealf(w5);
	f[11 * f_stride] = cimagf(w5);
	f[12 * f_stride] = crealf(w6);
	f[13 * f_stride] = cimagf(w6);
	f[14 * f_stride] = crealf(w7);
	f[15 * f_stride] = cimagf(w7);
}

void nnp_fft16_aos__ref(
	const float t[restrict static 32], size_t t_stride,
	float f[restrict static 32], size_t f_stride)
{
	/* Load inputs */
	float _Complex w0  = CMPLXF(t[ 0 * t_stride], t[ 1 * t_stride]);
	float _Complex w1  = CMPLXF(t[ 2 * t_stride], t[ 3 * t_stride]);
	float _Complex w2  = CMPLXF(t[ 4 * t_stride], t[ 5 * t_stride]);
	float _Complex w3  = CMPLXF(t[ 6 * t_stride], t[ 7 * t_stride]);
	float _Complex w4  = CMPLXF(t[ 8 * t_stride], t[ 9 * t_stride]);
	float _Complex w5  = CMPLXF(t[10 * t_stride], t[11 * t_stride]);
	float _Complex w6  = CMPLXF(t[12 * t_stride], t[13 * t_stride]);
	float _Complex w7  = CMPLXF(t[14 * t_stride], t[15 * t_stride]);
	float _Complex w8  = CMPLXF(t[16 * t_stride], t[17 * t_stride]);
	float _Complex w9  = CMPLXF(t[18 * t_stride], t[19 * t_stride]);
	float _Complex w10 = CMPLXF(t[20 * t_stride], t[21 * t_stride]);
	float _Complex w11 = CMPLXF(t[22 * t_stride], t[23 * t_stride]);
	float _Complex w12 = CMPLXF(t[24 * t_stride], t[25 * t_stride]);
	float _Complex w13 = CMPLXF(t[26 * t_stride], t[27 * t_stride]);
	float _Complex w14 = CMPLXF(t[28 * t_stride], t[29 * t_stride]);
	float _Complex w15 = CMPLXF(t[30 * t_stride], t[31 * t_stride]);

	fft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	/* Store outputs */
	f[ 0 * f_stride] = crealf(w0);
	f[ 1 * f_stride] = cimagf(w0);
	f[ 2 * f_stride] = crealf(w1);
	f[ 3 * f_stride] = cimagf(w1);
	f[ 4 * f_stride] = crealf(w2);
	f[ 5 * f_stride] = cimagf(w2);
	f[ 6 * f_stride] = crealf(w3);
	f[ 7 * f_stride] = cimagf(w3);
	f[ 8 * f_stride] = crealf(w4);
	f[ 9 * f_stride] = cimagf(w4);
	f[10 * f_stride] = crealf(w5);
	f[11 * f_stride] = cimagf(w5);
	f[12 * f_stride] = crealf(w6);
	f[13 * f_stride] = cimagf(w6);
	f[14 * f_stride] = crealf(w7);
	f[15 * f_stride] = cimagf(w7);
	f[16 * f_stride] = crealf(w8);
	f[17 * f_stride] = cimagf(w8);
	f[18 * f_stride] = crealf(w9);
	f[19 * f_stride] = cimagf(w9);
	f[20 * f_stride] = crealf(w10);
	f[21 * f_stride] = cimagf(w10);
	f[22 * f_stride] = crealf(w11);
	f[23 * f_stride] = cimagf(w11);
	f[24 * f_stride] = crealf(w12);
	f[25 * f_stride] = cimagf(w12);
	f[26 * f_stride] = crealf(w13);
	f[27 * f_stride] = cimagf(w13);
	f[28 * f_stride] = crealf(w14);
	f[29 * f_stride] = cimagf(w14);
	f[30 * f_stride] = crealf(w15);
	f[31 * f_stride] = cimagf(w15);
}

void nnp_fft32_aos__ref(
	const float t[restrict static 64], size_t t_stride,
	float f[restrict static 64], size_t f_stride)
{
	/* Load inputs */
	float _Complex w0  = CMPLXF(t[ 0 * t_stride], t[ 1 * t_stride]);
	float _Complex w1  = CMPLXF(t[ 2 * t_stride], t[ 3 * t_stride]);
	float _Complex w2  = CMPLXF(t[ 4 * t_stride], t[ 5 * t_stride]);
	float _Complex w3  = CMPLXF(t[ 6 * t_stride], t[ 7 * t_stride]);
	float _Complex w4  = CMPLXF(t[ 8 * t_stride], t[ 9 * t_stride]);
	float _Complex w5  = CMPLXF(t[10 * t_stride], t[11 * t_stride]);
	float _Complex w6  = CMPLXF(t[12 * t_stride], t[13 * t_stride]);
	float _Complex w7  = CMPLXF(t[14 * t_stride], t[15 * t_stride]);
	float _Complex w8  = CMPLXF(t[16 * t_stride], t[17 * t_stride]);
	float _Complex w9  = CMPLXF(t[18 * t_stride], t[19 * t_stride]);
	float _Complex w10 = CMPLXF(t[20 * t_stride], t[21 * t_stride]);
	float _Complex w11 = CMPLXF(t[22 * t_stride], t[23 * t_stride]);
	float _Complex w12 = CMPLXF(t[24 * t_stride], t[25 * t_stride]);
	float _Complex w13 = CMPLXF(t[26 * t_stride], t[27 * t_stride]);
	float _Complex w14 = CMPLXF(t[28 * t_stride], t[29 * t_stride]);
	float _Complex w15 = CMPLXF(t[30 * t_stride], t[31 * t_stride]);
	float _Complex w16 = CMPLXF(t[32 * t_stride], t[33 * t_stride]);
	float _Complex w17 = CMPLXF(t[34 * t_stride], t[35 * t_stride]);
	float _Complex w18 = CMPLXF(t[36 * t_stride], t[37 * t_stride]);
	float _Complex w19 = CMPLXF(t[38 * t_stride], t[39 * t_stride]);
	float _Complex w20 = CMPLXF(t[40 * t_stride], t[41 * t_stride]);
	float _Complex w21 = CMPLXF(t[42 * t_stride], t[43 * t_stride]);
	float _Complex w22 = CMPLXF(t[44 * t_stride], t[45 * t_stride]);
	float _Complex w23 = CMPLXF(t[46 * t_stride], t[47 * t_stride]);
	float _Complex w24 = CMPLXF(t[48 * t_stride], t[49 * t_stride]);
	float _Complex w25 = CMPLXF(t[50 * t_stride], t[51 * t_stride]);
	float _Complex w26 = CMPLXF(t[52 * t_stride], t[53 * t_stride]);
	float _Complex w27 = CMPLXF(t[54 * t_stride], t[55 * t_stride]);
	float _Complex w28 = CMPLXF(t[56 * t_stride], t[57 * t_stride]);
	float _Complex w29 = CMPLXF(t[58 * t_stride], t[59 * t_stride]);
	float _Complex w30 = CMPLXF(t[60 * t_stride], t[61 * t_stride]);
	float _Complex w31 = CMPLXF(t[62 * t_stride], t[63 * t_stride]);

	fft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	/* Store outputs */
	f[ 0 * f_stride] = crealf(w0);
	f[ 1 * f_stride] = cimagf(w0);
	f[ 2 * f_stride] = crealf(w1);
	f[ 3 * f_stride] = cimagf(w1);
	f[ 4 * f_stride] = crealf(w2);
	f[ 5 * f_stride] = cimagf(w2);
	f[ 6 * f_stride] = crealf(w3);
	f[ 7 * f_stride] = cimagf(w3);
	f[ 8 * f_stride] = crealf(w4);
	f[ 9 * f_stride] = cimagf(w4);
	f[10 * f_stride] = crealf(w5);
	f[11 * f_stride] = cimagf(w5);
	f[12 * f_stride] = crealf(w6);
	f[13 * f_stride] = cimagf(w6);
	f[14 * f_stride] = crealf(w7);
	f[15 * f_stride] = cimagf(w7);
	f[16 * f_stride] = crealf(w8);
	f[17 * f_stride] = cimagf(w8);
	f[18 * f_stride] = crealf(w9);
	f[19 * f_stride] = cimagf(w9);
	f[20 * f_stride] = crealf(w10);
	f[21 * f_stride] = cimagf(w10);
	f[22 * f_stride] = crealf(w11);
	f[23 * f_stride] = cimagf(w11);
	f[24 * f_stride] = crealf(w12);
	f[25 * f_stride] = cimagf(w12);
	f[26 * f_stride] = crealf(w13);
	f[27 * f_stride] = cimagf(w13);
	f[28 * f_stride] = crealf(w14);
	f[29 * f_stride] = cimagf(w14);
	f[30 * f_stride] = crealf(w15);
	f[31 * f_stride] = cimagf(w15);
	f[32 * f_stride] = crealf(w16);
	f[33 * f_stride] = cimagf(w16);
	f[34 * f_stride] = crealf(w17);
	f[35 * f_stride] = cimagf(w17);
	f[36 * f_stride] = crealf(w18);
	f[37 * f_stride] = cimagf(w18);
	f[38 * f_stride] = crealf(w19);
	f[39 * f_stride] = cimagf(w19);
	f[40 * f_stride] = crealf(w20);
	f[41 * f_stride] = cimagf(w20);
	f[42 * f_stride] = crealf(w21);
	f[43 * f_stride] = cimagf(w21);
	f[44 * f_stride] = crealf(w22);
	f[45 * f_stride] = cimagf(w22);
	f[46 * f_stride] = crealf(w23);
	f[47 * f_stride] = cimagf(w23);
	f[48 * f_stride] = crealf(w24);
	f[49 * f_stride] = cimagf(w24);
	f[50 * f_stride] = crealf(w25);
	f[51 * f_stride] = cimagf(w25);
	f[52 * f_stride] = crealf(w26);
	f[53 * f_stride] = cimagf(w26);
	f[54 * f_stride] = crealf(w27);
	f[55 * f_stride] = cimagf(w27);
	f[56 * f_stride] = crealf(w28);
	f[57 * f_stride] = cimagf(w28);
	f[58 * f_stride] = crealf(w29);
	f[59 * f_stride] = cimagf(w29);
	f[60 * f_stride] = crealf(w30);
	f[61 * f_stride] = cimagf(w30);
	f[62 * f_stride] = crealf(w31);
	f[63 * f_stride] = cimagf(w31);
}

void nnp_ifft2_aos__ref(
	const float f[restrict static 4], size_t f_stride,
	float t[restrict static 4], size_t t_stride)
{
	/* Load inputs */
	float _Complex w0 = CMPLXF(f[0 * f_stride], f[1 * f_stride]);
	float _Complex w1 = CMPLXF(f[2 * f_stride], f[3 * f_stride]);

	ifft2fc(&w0, &w1);

	/* Store outputs */
	t[0 * t_stride] = crealf(w0);
	t[1 * t_stride] = cimagf(w0);
	t[2 * t_stride] = crealf(w1);
	t[3 * t_stride] = cimagf(w1);
}

void nnp_ifft4_aos__ref(
	const float f[restrict static 8], size_t f_stride,
	float t[restrict static 8], size_t t_stride)
{
	/* Load inputs */
	float _Complex w0 = CMPLXF(f[0 * f_stride], f[1 * f_stride]);
	float _Complex w1 = CMPLXF(f[2 * f_stride], f[3 * f_stride]);
	float _Complex w2 = CMPLXF(f[4 * f_stride], f[5 * f_stride]);
	float _Complex w3 = CMPLXF(f[6 * f_stride], f[7 * f_stride]);

	ifft4fc(&w0, &w1, &w2, &w3);

	/* Store outputs */
	t[0 * t_stride] = crealf(w0);
	t[1 * t_stride] = cimagf(w0);
	t[2 * t_stride] = crealf(w1);
	t[3 * t_stride] = cimagf(w1);
	t[4 * t_stride] = crealf(w2);
	t[5 * t_stride] = cimagf(w2);
	t[6 * t_stride] = crealf(w3);
	t[7 * t_stride] = cimagf(w3);
}

void nnp_ifft8_aos__ref(
	const float f[restrict static 16], size_t f_stride,
	float t[restrict static 16], size_t t_stride)
{
	/* Load inputs */
	float _Complex w0 = CMPLXF(f[ 0 * f_stride], f[ 1 * f_stride]);
	float _Complex w1 = CMPLXF(f[ 2 * f_stride], f[ 3 * f_stride]);
	float _Complex w2 = CMPLXF(f[ 4 * f_stride], f[ 5 * f_stride]);
	float _Complex w3 = CMPLXF(f[ 6 * f_stride], f[ 7 * f_stride]);
	float _Complex w4 = CMPLXF(f[ 8 * f_stride], f[ 9 * f_stride]);
	float _Complex w5 = CMPLXF(f[10 * f_stride], f[11 * f_stride]);
	float _Complex w6 = CMPLXF(f[12 * f_stride], f[13 * f_stride]);
	float _Complex w7 = CMPLXF(f[14 * f_stride], f[15 * f_stride]);

	ifft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	/* Store outputs */
	t[ 0 * t_stride] = crealf(w0);
	t[ 1 * t_stride] = cimagf(w0);
	t[ 2 * t_stride] = crealf(w1);
	t[ 3 * t_stride] = cimagf(w1);
	t[ 4 * t_stride] = crealf(w2);
	t[ 5 * t_stride] = cimagf(w2);
	t[ 6 * t_stride] = crealf(w3);
	t[ 7 * t_stride] = cimagf(w3);
	t[ 8 * t_stride] = crealf(w4);
	t[ 9 * t_stride] = cimagf(w4);
	t[10 * t_stride] = crealf(w5);
	t[11 * t_stride] = cimagf(w5);
	t[12 * t_stride] = crealf(w6);
	t[13 * t_stride] = cimagf(w6);
	t[14 * t_stride] = crealf(w7);
	t[15 * t_stride] = cimagf(w7);
}

void nnp_ifft16_aos__ref(
	const float f[restrict static 32], size_t f_stride,
	float t[restrict static 32], size_t t_stride)
{
	/* Load inputs */
	float _Complex w0  = CMPLXF(f[ 0 * f_stride], f[ 1 * f_stride]);
	float _Complex w1  = CMPLXF(f[ 2 * f_stride], f[ 3 * f_stride]);
	float _Complex w2  = CMPLXF(f[ 4 * f_stride], f[ 5 * f_stride]);
	float _Complex w3  = CMPLXF(f[ 6 * f_stride], f[ 7 * f_stride]);
	float _Complex w4  = CMPLXF(f[ 8 * f_stride], f[ 9 * f_stride]);
	float _Complex w5  = CMPLXF(f[10 * f_stride], f[11 * f_stride]);
	float _Complex w6  = CMPLXF(f[12 * f_stride], f[13 * f_stride]);
	float _Complex w7  = CMPLXF(f[14 * f_stride], f[15 * f_stride]);
	float _Complex w8  = CMPLXF(f[16 * f_stride], f[17 * f_stride]);
	float _Complex w9  = CMPLXF(f[18 * f_stride], f[19 * f_stride]);
	float _Complex w10 = CMPLXF(f[20 * f_stride], f[21 * f_stride]);
	float _Complex w11 = CMPLXF(f[22 * f_stride], f[23 * f_stride]);
	float _Complex w12 = CMPLXF(f[24 * f_stride], f[25 * f_stride]);
	float _Complex w13 = CMPLXF(f[26 * f_stride], f[27 * f_stride]);
	float _Complex w14 = CMPLXF(f[28 * f_stride], f[29 * f_stride]);
	float _Complex w15 = CMPLXF(f[30 * f_stride], f[31 * f_stride]);

	ifft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	/* Store outputs */
	t[ 0 * t_stride] = crealf(w0);
	t[ 1 * t_stride] = cimagf(w0);
	t[ 2 * t_stride] = crealf(w1);
	t[ 3 * t_stride] = cimagf(w1);
	t[ 4 * t_stride] = crealf(w2);
	t[ 5 * t_stride] = cimagf(w2);
	t[ 6 * t_stride] = crealf(w3);
	t[ 7 * t_stride] = cimagf(w3);
	t[ 8 * t_stride] = crealf(w4);
	t[ 9 * t_stride] = cimagf(w4);
	t[10 * t_stride] = crealf(w5);
	t[11 * t_stride] = cimagf(w5);
	t[12 * t_stride] = crealf(w6);
	t[13 * t_stride] = cimagf(w6);
	t[14 * t_stride] = crealf(w7);
	t[15 * t_stride] = cimagf(w7);
	t[16 * t_stride] = crealf(w8);
	t[17 * t_stride] = cimagf(w8);
	t[18 * t_stride] = crealf(w9);
	t[19 * t_stride] = cimagf(w9);
	t[20 * t_stride] = crealf(w10);
	t[21 * t_stride] = cimagf(w10);
	t[22 * t_stride] = crealf(w11);
	t[23 * t_stride] = cimagf(w11);
	t[24 * t_stride] = crealf(w12);
	t[25 * t_stride] = cimagf(w12);
	t[26 * t_stride] = crealf(w13);
	t[27 * t_stride] = cimagf(w13);
	t[28 * t_stride] = crealf(w14);
	t[29 * t_stride] = cimagf(w14);
	t[30 * t_stride] = crealf(w15);
	t[31 * t_stride] = cimagf(w15);
}

void nnp_ifft32_aos__ref(
	const float f[restrict static 64], size_t f_stride,
	float t[restrict static 64], size_t t_stride)
{
	/* Load inputs */
	float _Complex w0  = CMPLXF(f[ 0 * f_stride], f[ 1 * f_stride]);
	float _Complex w1  = CMPLXF(f[ 2 * f_stride], f[ 3 * f_stride]);
	float _Complex w2  = CMPLXF(f[ 4 * f_stride], f[ 5 * f_stride]);
	float _Complex w3  = CMPLXF(f[ 6 * f_stride], f[ 7 * f_stride]);
	float _Complex w4  = CMPLXF(f[ 8 * f_stride], f[ 9 * f_stride]);
	float _Complex w5  = CMPLXF(f[10 * f_stride], f[11 * f_stride]);
	float _Complex w6  = CMPLXF(f[12 * f_stride], f[13 * f_stride]);
	float _Complex w7  = CMPLXF(f[14 * f_stride], f[15 * f_stride]);
	float _Complex w8  = CMPLXF(f[16 * f_stride], f[17 * f_stride]);
	float _Complex w9  = CMPLXF(f[18 * f_stride], f[19 * f_stride]);
	float _Complex w10 = CMPLXF(f[20 * f_stride], f[21 * f_stride]);
	float _Complex w11 = CMPLXF(f[22 * f_stride], f[23 * f_stride]);
	float _Complex w12 = CMPLXF(f[24 * f_stride], f[25 * f_stride]);
	float _Complex w13 = CMPLXF(f[26 * f_stride], f[27 * f_stride]);
	float _Complex w14 = CMPLXF(f[28 * f_stride], f[29 * f_stride]);
	float _Complex w15 = CMPLXF(f[30 * f_stride], f[31 * f_stride]);
	float _Complex w16 = CMPLXF(f[32 * f_stride], f[33 * f_stride]);
	float _Complex w17 = CMPLXF(f[34 * f_stride], f[35 * f_stride]);
	float _Complex w18 = CMPLXF(f[36 * f_stride], f[37 * f_stride]);
	float _Complex w19 = CMPLXF(f[38 * f_stride], f[39 * f_stride]);
	float _Complex w20 = CMPLXF(f[40 * f_stride], f[41 * f_stride]);
	float _Complex w21 = CMPLXF(f[42 * f_stride], f[43 * f_stride]);
	float _Complex w22 = CMPLXF(f[44 * f_stride], f[45 * f_stride]);
	float _Complex w23 = CMPLXF(f[46 * f_stride], f[47 * f_stride]);
	float _Complex w24 = CMPLXF(f[48 * f_stride], f[49 * f_stride]);
	float _Complex w25 = CMPLXF(f[50 * f_stride], f[51 * f_stride]);
	float _Complex w26 = CMPLXF(f[52 * f_stride], f[53 * f_stride]);
	float _Complex w27 = CMPLXF(f[54 * f_stride], f[55 * f_stride]);
	float _Complex w28 = CMPLXF(f[56 * f_stride], f[57 * f_stride]);
	float _Complex w29 = CMPLXF(f[58 * f_stride], f[59 * f_stride]);
	float _Complex w30 = CMPLXF(f[60 * f_stride], f[61 * f_stride]);
	float _Complex w31 = CMPLXF(f[62 * f_stride], f[63 * f_stride]);

	ifft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	/* Store outputs */
	t[ 0 * t_stride] = crealf(w0);
	t[ 1 * t_stride] = cimagf(w0);
	t[ 2 * t_stride] = crealf(w1);
	t[ 3 * t_stride] = cimagf(w1);
	t[ 4 * t_stride] = crealf(w2);
	t[ 5 * t_stride] = cimagf(w2);
	t[ 6 * t_stride] = crealf(w3);
	t[ 7 * t_stride] = cimagf(w3);
	t[ 8 * t_stride] = crealf(w4);
	t[ 9 * t_stride] = cimagf(w4);
	t[10 * t_stride] = crealf(w5);
	t[11 * t_stride] = cimagf(w5);
	t[12 * t_stride] = crealf(w6);
	t[13 * t_stride] = cimagf(w6);
	t[14 * t_stride] = crealf(w7);
	t[15 * t_stride] = cimagf(w7);
	t[16 * t_stride] = crealf(w8);
	t[17 * t_stride] = cimagf(w8);
	t[18 * t_stride] = crealf(w9);
	t[19 * t_stride] = cimagf(w9);
	t[20 * t_stride] = crealf(w10);
	t[21 * t_stride] = cimagf(w10);
	t[22 * t_stride] = crealf(w11);
	t[23 * t_stride] = cimagf(w11);
	t[24 * t_stride] = crealf(w12);
	t[25 * t_stride] = cimagf(w12);
	t[26 * t_stride] = crealf(w13);
	t[27 * t_stride] = cimagf(w13);
	t[28 * t_stride] = crealf(w14);
	t[29 * t_stride] = cimagf(w14);
	t[30 * t_stride] = crealf(w15);
	t[31 * t_stride] = cimagf(w15);
	t[32 * t_stride] = crealf(w16);
	t[33 * t_stride] = cimagf(w16);
	t[34 * t_stride] = crealf(w17);
	t[35 * t_stride] = cimagf(w17);
	t[36 * t_stride] = crealf(w18);
	t[37 * t_stride] = cimagf(w18);
	t[38 * t_stride] = crealf(w19);
	t[39 * t_stride] = cimagf(w19);
	t[40 * t_stride] = crealf(w20);
	t[41 * t_stride] = cimagf(w20);
	t[42 * t_stride] = crealf(w21);
	t[43 * t_stride] = cimagf(w21);
	t[44 * t_stride] = crealf(w22);
	t[45 * t_stride] = cimagf(w22);
	t[46 * t_stride] = crealf(w23);
	t[47 * t_stride] = cimagf(w23);
	t[48 * t_stride] = crealf(w24);
	t[49 * t_stride] = cimagf(w24);
	t[50 * t_stride] = crealf(w25);
	t[51 * t_stride] = cimagf(w25);
	t[52 * t_stride] = crealf(w26);
	t[53 * t_stride] = cimagf(w26);
	t[54 * t_stride] = crealf(w27);
	t[55 * t_stride] = cimagf(w27);
	t[56 * t_stride] = crealf(w28);
	t[57 * t_stride] = cimagf(w28);
	t[58 * t_stride] = crealf(w29);
	t[59 * t_stride] = cimagf(w29);
	t[60 * t_stride] = crealf(w30);
	t[61 * t_stride] = cimagf(w30);
	t[62 * t_stride] = crealf(w31);
	t[63 * t_stride] = cimagf(w31);
}
