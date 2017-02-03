#include <psimd/fft/aos.h>


void nnp_fft4_4aos__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	v4f w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	v4f_fft4_aos(
		t, t + 16, 4, 0, 8,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i);
	v4f_st(f +  0, w0r);
	v4f_st(f +  4, w0i);
	v4f_st(f +  8, w1r);
	v4f_st(f + 12, w1i);
	v4f_st(f + 16, w2r);
	v4f_st(f + 20, w2i);
	v4f_st(f + 24, w3r);
	v4f_st(f + 28, w3i);
}

void nnp_fft8_4aos__psimd(
	const float t[restrict static 64],
	float f[restrict static 64])
{
	v4f w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	v4f_fft8_aos(
		t, t + 32, 4, 0, 16,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i, &w4r, &w4i, &w5r, &w5i, &w6r, &w6i, &w7r, &w7i);
	v4f_st(f +  0, w0r);
	v4f_st(f +  4, w0i);
	v4f_st(f +  8, w1r);
	v4f_st(f + 12, w1i);
	v4f_st(f + 16, w2r);
	v4f_st(f + 20, w2i);
	v4f_st(f + 24, w3r);
	v4f_st(f + 28, w3i);
	v4f_st(f + 32, w4r);
	v4f_st(f + 36, w4i);
	v4f_st(f + 40, w5r);
	v4f_st(f + 44, w5i);
	v4f_st(f + 48, w6r);
	v4f_st(f + 52, w6i);
	v4f_st(f + 56, w7r);
	v4f_st(f + 60, w7i);
}

void nnp_ifft4_4aos__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	const v4f w0r = v4f_ld(f +  0);
	const v4f w0i = v4f_ld(f +  4);
	const v4f w1r = v4f_ld(f +  8);
	const v4f w1i = v4f_ld(f + 12);
	const v4f w2r = v4f_ld(f + 16);
	const v4f w2i = v4f_ld(f + 20);
	const v4f w3r = v4f_ld(f + 24);
	const v4f w3i = v4f_ld(f + 28);

	v4f_ifft4_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i,
		t, t + 16, 4);
}

void nnp_ifft8_4aos__psimd(
	const float f[restrict static 64],
	float t[restrict static 64])
{
	const v4f w0r = v4f_ld(f +  0);
	const v4f w0i = v4f_ld(f +  4);
	const v4f w1r = v4f_ld(f +  8);
	const v4f w1i = v4f_ld(f + 12);
	const v4f w2r = v4f_ld(f + 16);
	const v4f w2i = v4f_ld(f + 20);
	const v4f w3r = v4f_ld(f + 24);
	const v4f w3i = v4f_ld(f + 28);
	const v4f w4r = v4f_ld(f + 32);
	const v4f w4i = v4f_ld(f + 36);
	const v4f w5r = v4f_ld(f + 40);
	const v4f w5i = v4f_ld(f + 44);
	const v4f w6r = v4f_ld(f + 48);
	const v4f w6i = v4f_ld(f + 52);
	const v4f w7r = v4f_ld(f + 56);
	const v4f w7i = v4f_ld(f + 60);

	v4f_ifft8_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
		t, t + 32, 4);
}
