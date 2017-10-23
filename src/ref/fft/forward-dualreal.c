#include <stddef.h>

#include <nnpack/fft-constants.h>
#include <nnpack/complex.h>
#include <fft/complex.h>


void nnp_fft8_dualreal__ref(const float t[restrict static 16], float f[restrict static 16]) {
	float _Complex w0 = CMPLXF(t[0], t[ 8]);
	float _Complex w1 = CMPLXF(t[1], t[ 9]);
	float _Complex w2 = CMPLXF(t[2], t[10]);
	float _Complex w3 = CMPLXF(t[3], t[11]);
	float _Complex w4 = CMPLXF(t[4], t[12]);
	float _Complex w5 = CMPLXF(t[5], t[13]);
	float _Complex w6 = CMPLXF(t[6], t[14]);
	float _Complex w7 = CMPLXF(t[7], t[15]);

	fft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	const float x0 = crealf(w0);
	const float h0 = cimagf(w0);

	const float _Complex x1 =  0.5f  * (w1 + conjf(w7));
	const float _Complex h1 = -0.5fi * (w1 - conjf(w7));
	const float _Complex x2 =  0.5f  * (w2 + conjf(w6));
	const float _Complex h2 = -0.5fi * (w2 - conjf(w6));
	const float _Complex x3 =  0.5f  * (w3 + conjf(w5));
	const float _Complex h3 = -0.5fi * (w3 - conjf(w5));

	const float x4 = crealf(w4);
	const float h4 = cimagf(w4);

	f[0] = x0;
	f[1] = h0;
	f[2] = crealf(x1);
	f[3] = crealf(h1);
	f[4] = crealf(x2);
	f[5] = crealf(h2);
	f[6] = crealf(x3);
	f[7] = crealf(h3);

	f[ 8] = x4;
	f[ 9] = h4;
	f[10] = cimagf(x1);
	f[11] = cimagf(h1);
	f[12] = cimagf(x2);
	f[13] = cimagf(h2);
	f[14] = cimagf(x3);
	f[15] = cimagf(h3);
}

void nnp_fft16_dualreal__ref(const float t[restrict static 16], float f[restrict static 16]) {
	float _Complex w0  = CMPLXF(t[ 0], t[16]);
	float _Complex w1  = CMPLXF(t[ 1], t[17]);
	float _Complex w2  = CMPLXF(t[ 2], t[18]);
	float _Complex w3  = CMPLXF(t[ 3], t[19]);
	float _Complex w4  = CMPLXF(t[ 4], t[20]);
	float _Complex w5  = CMPLXF(t[ 5], t[21]);
	float _Complex w6  = CMPLXF(t[ 6], t[22]);
	float _Complex w7  = CMPLXF(t[ 7], t[23]);
	float _Complex w8  = CMPLXF(t[ 8], t[24]);
	float _Complex w9  = CMPLXF(t[ 9], t[25]);
	float _Complex w10 = CMPLXF(t[10], t[26]);
	float _Complex w11 = CMPLXF(t[11], t[27]);
	float _Complex w12 = CMPLXF(t[12], t[28]);
	float _Complex w13 = CMPLXF(t[13], t[29]);
	float _Complex w14 = CMPLXF(t[14], t[30]);
	float _Complex w15 = CMPLXF(t[15], t[31]);

	fft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	const float x0 = crealf(w0);
	const float h0 = cimagf(w0);

	const float _Complex x1 =  0.5f  * (w1 + conjf(w15));
	const float _Complex h1 = -0.5fi * (w1 - conjf(w15));
	const float _Complex x2 =  0.5f  * (w2 + conjf(w14));
	const float _Complex h2 = -0.5fi * (w2 - conjf(w14));
	const float _Complex x3 =  0.5f  * (w3 + conjf(w13));
	const float _Complex h3 = -0.5fi * (w3 - conjf(w13));
	const float _Complex x4 =  0.5f  * (w4 + conjf(w12));
	const float _Complex h4 = -0.5fi * (w4 - conjf(w12));
	const float _Complex x5 =  0.5f  * (w5 + conjf(w11));
	const float _Complex h5 = -0.5fi * (w5 - conjf(w11));
	const float _Complex x6 =  0.5f  * (w6 + conjf(w10));
	const float _Complex h6 = -0.5fi * (w6 - conjf(w10));
	const float _Complex x7 =  0.5f  * (w7 + conjf(w9));
	const float _Complex h7 = -0.5fi * (w7 - conjf(w9));

	const float x8 = crealf(w8);
	const float h8 = cimagf(w8);

	f[ 0] = x0;
	f[ 1] = h0;
	f[ 2] = crealf(x1);
	f[ 3] = crealf(h1);
	f[ 4] = crealf(x2);
	f[ 5] = crealf(h2);
	f[ 6] = crealf(x3);
	f[ 7] = crealf(h3);
	f[ 8] = crealf(x4);
	f[ 9] = crealf(h4);
	f[10] = crealf(x5);
	f[11] = crealf(h5);
	f[12] = crealf(x6);
	f[13] = crealf(h6);
	f[14] = crealf(x7);
	f[15] = crealf(h7);

	f[16] = x8;
	f[17] = h8;
	f[18] = cimagf(x1);
	f[19] = cimagf(h1);
	f[20] = cimagf(x2);
	f[21] = cimagf(h2);
	f[22] = cimagf(x3);
	f[23] = cimagf(h3);
	f[24] = cimagf(x4);
	f[25] = cimagf(h4);
	f[26] = cimagf(x5);
	f[27] = cimagf(h5);
	f[28] = cimagf(x6);
	f[29] = cimagf(h6);
	f[30] = cimagf(x7);
	f[31] = cimagf(h7);
}

void nnp_fft32_dualreal__ref(const float t[restrict static 32], float f[restrict static 32]) {
	float _Complex w0  = CMPLXF(t[ 0], t[32]);
	float _Complex w1  = CMPLXF(t[ 1], t[33]);
	float _Complex w2  = CMPLXF(t[ 2], t[34]);
	float _Complex w3  = CMPLXF(t[ 3], t[35]);
	float _Complex w4  = CMPLXF(t[ 4], t[36]);
	float _Complex w5  = CMPLXF(t[ 5], t[37]);
	float _Complex w6  = CMPLXF(t[ 6], t[38]);
	float _Complex w7  = CMPLXF(t[ 7], t[39]);
	float _Complex w8  = CMPLXF(t[ 8], t[40]);
	float _Complex w9  = CMPLXF(t[ 9], t[41]);
	float _Complex w10 = CMPLXF(t[10], t[42]);
	float _Complex w11 = CMPLXF(t[11], t[43]);
	float _Complex w12 = CMPLXF(t[12], t[44]);
	float _Complex w13 = CMPLXF(t[13], t[45]);
	float _Complex w14 = CMPLXF(t[14], t[46]);
	float _Complex w15 = CMPLXF(t[15], t[47]);
	float _Complex w16 = CMPLXF(t[16], t[48]);
	float _Complex w17 = CMPLXF(t[17], t[49]);
	float _Complex w18 = CMPLXF(t[18], t[50]);
	float _Complex w19 = CMPLXF(t[19], t[51]);
	float _Complex w20 = CMPLXF(t[20], t[52]);
	float _Complex w21 = CMPLXF(t[21], t[53]);
	float _Complex w22 = CMPLXF(t[22], t[54]);
	float _Complex w23 = CMPLXF(t[23], t[55]);
	float _Complex w24 = CMPLXF(t[24], t[56]);
	float _Complex w25 = CMPLXF(t[25], t[57]);
	float _Complex w26 = CMPLXF(t[26], t[58]);
	float _Complex w27 = CMPLXF(t[27], t[59]);
	float _Complex w28 = CMPLXF(t[28], t[60]);
	float _Complex w29 = CMPLXF(t[29], t[61]);
	float _Complex w30 = CMPLXF(t[30], t[62]);
	float _Complex w31 = CMPLXF(t[31], t[63]);

	fft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	const float x0 = crealf(w0);
	const float h0 = cimagf(w0);

	const float _Complex x1  =  0.5f  * (w1  + conjf(w31));
	const float _Complex h1  = -0.5fi * (w1  - conjf(w31));
	const float _Complex x2  =  0.5f  * (w2  + conjf(w30));
	const float _Complex h2  = -0.5fi * (w2  - conjf(w30));
	const float _Complex x3  =  0.5f  * (w3  + conjf(w29));
	const float _Complex h3  = -0.5fi * (w3  - conjf(w29));
	const float _Complex x4  =  0.5f  * (w4  + conjf(w28));
	const float _Complex h4  = -0.5fi * (w4  - conjf(w28));
	const float _Complex x5  =  0.5f  * (w5  + conjf(w27));
	const float _Complex h5  = -0.5fi * (w5  - conjf(w27));
	const float _Complex x6  =  0.5f  * (w6  + conjf(w26));
	const float _Complex h6  = -0.5fi * (w6  - conjf(w26));
	const float _Complex x7  =  0.5f  * (w7  + conjf(w25));
	const float _Complex h7  = -0.5fi * (w7  - conjf(w25));
	const float _Complex x8  =  0.5f  * (w8  + conjf(w24));
	const float _Complex h8  = -0.5fi * (w8  - conjf(w24));
	const float _Complex x9  =  0.5f  * (w9  + conjf(w23));
	const float _Complex h9  = -0.5fi * (w9  - conjf(w23));
	const float _Complex x10 =  0.5f  * (w10 + conjf(w22));
	const float _Complex h10 = -0.5fi * (w10 - conjf(w22));
	const float _Complex x11 =  0.5f  * (w11 + conjf(w21));
	const float _Complex h11 = -0.5fi * (w11 - conjf(w21));
	const float _Complex x12 =  0.5f  * (w12 + conjf(w20));
	const float _Complex h12 = -0.5fi * (w12 - conjf(w20));
	const float _Complex x13 =  0.5f  * (w13 + conjf(w19));
	const float _Complex h13 = -0.5fi * (w13 - conjf(w19));
	const float _Complex x14 =  0.5f  * (w14 + conjf(w18));
	const float _Complex h14 = -0.5fi * (w14 - conjf(w18));
	const float _Complex x15 =  0.5f  * (w15 + conjf(w17));
	const float _Complex h15 = -0.5fi * (w15 - conjf(w17));

	const float x16 = crealf(w16);
	const float h16 = cimagf(w16);

	f[ 0] = x0;
	f[ 1] = h0;
	f[ 2] = crealf(x1);
	f[ 3] = crealf(h1);
	f[ 4] = crealf(x2);
	f[ 5] = crealf(h2);
	f[ 6] = crealf(x3);
	f[ 7] = crealf(h3);
	f[ 8] = crealf(x4);
	f[ 9] = crealf(h4);
	f[10] = crealf(x5);
	f[11] = crealf(h5);
	f[12] = crealf(x6);
	f[13] = crealf(h6);
	f[14] = crealf(x7);
	f[15] = crealf(h7);
	f[16] = crealf(x8);
	f[17] = crealf(h8);
	f[18] = crealf(x9);
	f[19] = crealf(h9);
	f[20] = crealf(x10);
	f[21] = crealf(h10);
	f[22] = crealf(x11);
	f[23] = crealf(h11);
	f[24] = crealf(x12);
	f[25] = crealf(h12);
	f[26] = crealf(x13);
	f[27] = crealf(h13);
	f[28] = crealf(x14);
	f[29] = crealf(h14);
	f[30] = crealf(x15);
	f[31] = crealf(h15);

	f[32] = x16;
	f[33] = h16;
	f[34] = cimagf(x1);
	f[35] = cimagf(h1);
	f[36] = cimagf(x2);
	f[37] = cimagf(h2);
	f[38] = cimagf(x3);
	f[39] = cimagf(h3);
	f[40] = cimagf(x4);
	f[41] = cimagf(h4);
	f[42] = cimagf(x5);
	f[43] = cimagf(h5);
	f[44] = cimagf(x6);
	f[45] = cimagf(h6);
	f[46] = cimagf(x7);
	f[47] = cimagf(h7);
	f[48] = cimagf(x8);
	f[49] = cimagf(h8);
	f[50] = cimagf(x9);
	f[51] = cimagf(h9);
	f[52] = cimagf(x10);
	f[53] = cimagf(h10);
	f[54] = cimagf(x11);
	f[55] = cimagf(h11);
	f[56] = cimagf(x12);
	f[57] = cimagf(h12);
	f[58] = cimagf(x13);
	f[59] = cimagf(h13);
	f[60] = cimagf(x14);
	f[61] = cimagf(h14);
	f[62] = cimagf(x15);
	f[63] = cimagf(h15);
}
