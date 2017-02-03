#include <stddef.h>

#include <nnpack/fft-constants.h>
#include <nnpack/complex.h>
#include <fft/complex.h>


void nnp_ifft8_dualreal__ref(const float f[restrict static 16], float t[restrict static 16]) {
	const float x0 = f[0];
	const float h0 = f[1];
	const float x4 = f[8];
	const float h4 = f[9];

	const float _Complex x1 = CMPLXF(f[2], f[10]);
	const float _Complex h1 = CMPLXF(f[3], f[11]);
	const float _Complex x2 = CMPLXF(f[4], f[12]);
	const float _Complex h2 = CMPLXF(f[5], f[13]);
	const float _Complex x3 = CMPLXF(f[6], f[14]);
	const float _Complex h3 = CMPLXF(f[7], f[15]);

	float _Complex w0 = CMPLXF(x0, h0);
	float _Complex w1 =       x1 + _Complex_I * h1;
	float _Complex w2 =       x2 + _Complex_I * h2;
	float _Complex w3 =       x3 + _Complex_I * h3;
	float _Complex w4 = CMPLXF(x4, h4);
	float _Complex w5 = conjf(x3 - _Complex_I * h3);
	float _Complex w6 = conjf(x2 - _Complex_I * h2);
	float _Complex w7 = conjf(x1 - _Complex_I * h1);
	
	ifft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	t[ 0] = crealf(w0);
	t[ 1] = crealf(w1);
	t[ 2] = crealf(w2);
	t[ 3] = crealf(w3);
	t[ 4] = crealf(w4);
	t[ 5] = crealf(w5);
	t[ 6] = crealf(w6);
	t[ 7] = crealf(w7);
	t[ 8] = cimagf(w0);
	t[ 9] = cimagf(w1);
	t[10] = cimagf(w2);
	t[11] = cimagf(w3);
	t[12] = cimagf(w4);
	t[13] = cimagf(w5);
	t[14] = cimagf(w6);
	t[15] = cimagf(w7);
}

void nnp_ifft16_dualreal__ref(const float f[restrict static 32], float t[restrict static 32]) {
	const float x0 = f[0];
	const float h0 = f[1];
	const float x8 = f[16];
	const float h8 = f[17];

	const float _Complex x1 = CMPLXF(f[ 2], f[18]);
	const float _Complex h1 = CMPLXF(f[ 3], f[19]);
	const float _Complex x2 = CMPLXF(f[ 4], f[20]);
	const float _Complex h2 = CMPLXF(f[ 5], f[21]);
	const float _Complex x3 = CMPLXF(f[ 6], f[22]);
	const float _Complex h3 = CMPLXF(f[ 7], f[23]);
	const float _Complex x4 = CMPLXF(f[ 8], f[24]);
	const float _Complex h4 = CMPLXF(f[ 9], f[25]);
	const float _Complex x5 = CMPLXF(f[10], f[26]);
	const float _Complex h5 = CMPLXF(f[11], f[27]);
	const float _Complex x6 = CMPLXF(f[12], f[28]);
	const float _Complex h6 = CMPLXF(f[13], f[29]);
	const float _Complex x7 = CMPLXF(f[14], f[30]);
	const float _Complex h7 = CMPLXF(f[15], f[31]);

	float _Complex w0  = CMPLXF(x0, h0);
	float _Complex w1  =       x1 + _Complex_I * h1;
	float _Complex w2  =       x2 + _Complex_I * h2;
	float _Complex w3  =       x3 + _Complex_I * h3;
	float _Complex w4  =       x4 + _Complex_I * h4;
	float _Complex w5  =       x5 + _Complex_I * h5;
	float _Complex w6  =       x6 + _Complex_I * h6;
	float _Complex w7  =       x7 + _Complex_I * h7;
	float _Complex w8  = CMPLXF(x8, h8);
	float _Complex w9  = conjf(x7 - _Complex_I * h7);
	float _Complex w10 = conjf(x6 - _Complex_I * h6);
	float _Complex w11 = conjf(x5 - _Complex_I * h5);
	float _Complex w12 = conjf(x4 - _Complex_I * h4);
	float _Complex w13 = conjf(x3 - _Complex_I * h3);
	float _Complex w14 = conjf(x2 - _Complex_I * h2);
	float _Complex w15 = conjf(x1 - _Complex_I * h1);
	
	ifft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	t[ 0] = crealf(w0);
	t[ 1] = crealf(w1);
	t[ 2] = crealf(w2);
	t[ 3] = crealf(w3);
	t[ 4] = crealf(w4);
	t[ 5] = crealf(w5);
	t[ 6] = crealf(w6);
	t[ 7] = crealf(w7);
	t[ 8] = crealf(w8);
	t[ 9] = crealf(w9);
	t[10] = crealf(w10);
	t[11] = crealf(w11);
	t[12] = crealf(w12);
	t[13] = crealf(w13);
	t[14] = crealf(w14);
	t[15] = crealf(w15);
	t[16] = cimagf(w0);
	t[17] = cimagf(w1);
	t[18] = cimagf(w2);
	t[19] = cimagf(w3);
	t[20] = cimagf(w4);
	t[21] = cimagf(w5);
	t[22] = cimagf(w6);
	t[23] = cimagf(w7);
	t[24] = cimagf(w8);
	t[25] = cimagf(w9);
	t[26] = cimagf(w10);
	t[27] = cimagf(w11);
	t[28] = cimagf(w12);
	t[29] = cimagf(w13);
	t[30] = cimagf(w14);
	t[31] = cimagf(w15);
}

void nnp_ifft32_dualreal__ref(const float f[restrict static 64], float t[restrict static 64]) {
	const float x0 = f[0];
	const float h0 = f[1];
	const float x16 = f[32];
	const float h16 = f[33];

	const float _Complex x1  = CMPLXF(f[ 2], f[34]);
	const float _Complex h1  = CMPLXF(f[ 3], f[35]);
	const float _Complex x2  = CMPLXF(f[ 4], f[36]);
	const float _Complex h2  = CMPLXF(f[ 5], f[37]);
	const float _Complex x3  = CMPLXF(f[ 6], f[38]);
	const float _Complex h3  = CMPLXF(f[ 7], f[39]);
	const float _Complex x4  = CMPLXF(f[ 8], f[40]);
	const float _Complex h4  = CMPLXF(f[ 9], f[41]);
	const float _Complex x5  = CMPLXF(f[10], f[42]);
	const float _Complex h5  = CMPLXF(f[11], f[43]);
	const float _Complex x6  = CMPLXF(f[12], f[44]);
	const float _Complex h6  = CMPLXF(f[13], f[45]);
	const float _Complex x7  = CMPLXF(f[14], f[46]);
	const float _Complex h7  = CMPLXF(f[15], f[47]);
	const float _Complex x8  = CMPLXF(f[16], f[48]);
	const float _Complex h8  = CMPLXF(f[17], f[49]);
	const float _Complex x9  = CMPLXF(f[18], f[50]);
	const float _Complex h9  = CMPLXF(f[19], f[51]);
	const float _Complex x10 = CMPLXF(f[20], f[52]);
	const float _Complex h10 = CMPLXF(f[21], f[53]);
	const float _Complex x11 = CMPLXF(f[22], f[54]);
	const float _Complex h11 = CMPLXF(f[23], f[55]);
	const float _Complex x12 = CMPLXF(f[24], f[56]);
	const float _Complex h12 = CMPLXF(f[25], f[57]);
	const float _Complex x13 = CMPLXF(f[26], f[58]);
	const float _Complex h13 = CMPLXF(f[27], f[59]);
	const float _Complex x14 = CMPLXF(f[28], f[60]);
	const float _Complex h14 = CMPLXF(f[29], f[61]);
	const float _Complex x15 = CMPLXF(f[30], f[62]);
	const float _Complex h15 = CMPLXF(f[31], f[63]);

	float _Complex w0  = CMPLXF(x0, h0);
	float _Complex w1  =       x1  + _Complex_I * h1;
	float _Complex w2  =       x2  + _Complex_I * h2;
	float _Complex w3  =       x3  + _Complex_I * h3;
	float _Complex w4  =       x4  + _Complex_I * h4;
	float _Complex w5  =       x5  + _Complex_I * h5;
	float _Complex w6  =       x6  + _Complex_I * h6;
	float _Complex w7  =       x7  + _Complex_I * h7;
	float _Complex w8  =       x8  + _Complex_I * h8;
	float _Complex w9  =       x9  + _Complex_I * h9;
	float _Complex w10 =       x10 + _Complex_I * h10;
	float _Complex w11 =       x11 + _Complex_I * h11;
	float _Complex w12 =       x12 + _Complex_I * h12;
	float _Complex w13 =       x13 + _Complex_I * h13;
	float _Complex w14 =       x14 + _Complex_I * h14;
	float _Complex w15 =       x15 + _Complex_I * h15;
	float _Complex w16 = CMPLXF(x16, h16);
	float _Complex w17 = conjf(x15 - _Complex_I * h15);
	float _Complex w18 = conjf(x14 - _Complex_I * h14);
	float _Complex w19 = conjf(x13 - _Complex_I * h13);
	float _Complex w20 = conjf(x12 - _Complex_I * h12);
	float _Complex w21 = conjf(x11 - _Complex_I * h11);
	float _Complex w22 = conjf(x10 - _Complex_I * h10);
	float _Complex w23 = conjf(x9  - _Complex_I * h9);
	float _Complex w24 = conjf(x8  - _Complex_I * h8);
	float _Complex w25 = conjf(x7  - _Complex_I * h7);
	float _Complex w26 = conjf(x6  - _Complex_I * h6);
	float _Complex w27 = conjf(x5  - _Complex_I * h5);
	float _Complex w28 = conjf(x4  - _Complex_I * h4);
	float _Complex w29 = conjf(x3  - _Complex_I * h3);
	float _Complex w30 = conjf(x2  - _Complex_I * h2);
	float _Complex w31 = conjf(x1  - _Complex_I * h1);

	ifft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	t[ 0] = crealf(w0);
	t[ 1] = crealf(w1);
	t[ 2] = crealf(w2);
	t[ 3] = crealf(w3);
	t[ 4] = crealf(w4);
	t[ 5] = crealf(w5);
	t[ 6] = crealf(w6);
	t[ 7] = crealf(w7);
	t[ 8] = crealf(w8);
	t[ 9] = crealf(w9);
	t[10] = crealf(w10);
	t[11] = crealf(w11);
	t[12] = crealf(w12);
	t[13] = crealf(w13);
	t[14] = crealf(w14);
	t[15] = crealf(w15);
	t[16] = crealf(w16);
	t[17] = crealf(w17);
	t[18] = crealf(w18);
	t[19] = crealf(w19);
	t[20] = crealf(w20);
	t[21] = crealf(w21);
	t[22] = crealf(w22);
	t[23] = crealf(w23);
	t[24] = crealf(w24);
	t[25] = crealf(w25);
	t[26] = crealf(w26);
	t[27] = crealf(w27);
	t[28] = crealf(w28);
	t[29] = crealf(w29);
	t[30] = crealf(w30);
	t[31] = crealf(w31);
	t[32] = cimagf(w0);
	t[33] = cimagf(w1);
	t[34] = cimagf(w2);
	t[35] = cimagf(w3);
	t[36] = cimagf(w4);
	t[37] = cimagf(w5);
	t[38] = cimagf(w6);
	t[39] = cimagf(w7);
	t[40] = cimagf(w8);
	t[41] = cimagf(w9);
	t[42] = cimagf(w10);
	t[43] = cimagf(w11);
	t[44] = cimagf(w12);
	t[45] = cimagf(w13);
	t[46] = cimagf(w14);
	t[47] = cimagf(w15);
	t[48] = cimagf(w16);
	t[49] = cimagf(w17);
	t[50] = cimagf(w18);
	t[51] = cimagf(w19);
	t[52] = cimagf(w20);
	t[53] = cimagf(w21);
	t[54] = cimagf(w22);
	t[55] = cimagf(w23);
	t[56] = cimagf(w24);
	t[57] = cimagf(w25);
	t[58] = cimagf(w26);
	t[59] = cimagf(w27);
	t[60] = cimagf(w28);
	t[61] = cimagf(w29);
	t[62] = cimagf(w30);
	t[63] = cimagf(w31);
}
