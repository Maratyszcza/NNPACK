#include <psimd/fft/real.h>


void nnp_fft8_4real__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	v4f_fft8_real(
		t, t + 16, 4, 0, 8,
		f, 4);
}

void nnp_fft16_4real__psimd(
	const float t[restrict static 64],
	float f[restrict static 64])
{
	v4f w0r, w8r, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	v4f_fft16_real(
		t, t + 32, 4, 0, 16,
		f, 4);
}

void nnp_ifft8_4real__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	const v4f f0r = v4f_ld(f +  0);
	const v4f f4r = v4f_ld(f +  4);
	const v4f f1r = v4f_ld(f +  8);
	const v4f f1i = v4f_ld(f + 12);
	const v4f f2r = v4f_ld(f + 16);
	const v4f f2i = v4f_ld(f + 20);
	const v4f f3r = v4f_ld(f + 24);
	const v4f f3i = v4f_ld(f + 28);
	v4f_ifft8_real(
		f0r, f4r, f1r, f1i, f2r, f2i, f3r, f3i,
		t, t + 16, 4);
}

void nnp_ifft16_4real__psimd(
	const float f[restrict static 64],
	float t[restrict static 64])
{
	const v4f f0r = v4f_ld(f +  0);
	const v4f f8r = v4f_ld(f +  4);
	const v4f f1r = v4f_ld(f +  8);
	const v4f f1i = v4f_ld(f + 12);
	const v4f f2r = v4f_ld(f + 16);
	const v4f f2i = v4f_ld(f + 20);
	const v4f f3r = v4f_ld(f + 24);
	const v4f f3i = v4f_ld(f + 28);
	const v4f f4r = v4f_ld(f + 32);
	const v4f f4i = v4f_ld(f + 36);
	const v4f f5r = v4f_ld(f + 40);
	const v4f f5i = v4f_ld(f + 44);
	const v4f f6r = v4f_ld(f + 48);
	const v4f f6i = v4f_ld(f + 52);
	const v4f f7r = v4f_ld(f + 56);
	const v4f f7i = v4f_ld(f + 60);
	v4f_ifft16_real(
		f0r, f8r, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i, f6r, f6i, f7r, f7i,
		t, t + 32, 4);
}
