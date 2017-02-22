#include <psimd/fft/real.h>


void nnp_fft8_4real__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	psimd_fft8_real_f32(
		t, t + 16, 4, 0, 8,
		f, 4);
}

void nnp_fft16_4real__psimd(
	const float t[restrict static 64],
	float f[restrict static 64])
{
	psimd_fft16_real_f32(
		t, t + 32, 4, 0, 16,
		f, 4);
}

void nnp_ifft8_4real__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	const psimd_f32 f0r = psimd_load_f32(f +  0);
	const psimd_f32 f4r = psimd_load_f32(f +  4);
	const psimd_f32 f1r = psimd_load_f32(f +  8);
	const psimd_f32 f1i = psimd_load_f32(f + 12);
	const psimd_f32 f2r = psimd_load_f32(f + 16);
	const psimd_f32 f2i = psimd_load_f32(f + 20);
	const psimd_f32 f3r = psimd_load_f32(f + 24);
	const psimd_f32 f3i = psimd_load_f32(f + 28);
	psimd_ifft8_real_f32(
		f0r, f4r, f1r, f1i, f2r, f2i, f3r, f3i,
		t, t + 16, 4);
}

void nnp_ifft16_4real__psimd(
	const float f[restrict static 64],
	float t[restrict static 64])
{
	const psimd_f32 f0r = psimd_load_f32(f +  0);
	const psimd_f32 f8r = psimd_load_f32(f +  4);
	const psimd_f32 f1r = psimd_load_f32(f +  8);
	const psimd_f32 f1i = psimd_load_f32(f + 12);
	const psimd_f32 f2r = psimd_load_f32(f + 16);
	const psimd_f32 f2i = psimd_load_f32(f + 20);
	const psimd_f32 f3r = psimd_load_f32(f + 24);
	const psimd_f32 f3i = psimd_load_f32(f + 28);
	const psimd_f32 f4r = psimd_load_f32(f + 32);
	const psimd_f32 f4i = psimd_load_f32(f + 36);
	const psimd_f32 f5r = psimd_load_f32(f + 40);
	const psimd_f32 f5i = psimd_load_f32(f + 44);
	const psimd_f32 f6r = psimd_load_f32(f + 48);
	const psimd_f32 f6i = psimd_load_f32(f + 52);
	const psimd_f32 f7r = psimd_load_f32(f + 56);
	const psimd_f32 f7i = psimd_load_f32(f + 60);
	psimd_ifft16_real_f32(
		f0r, f8r, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i, f6r, f6i, f7r, f7i,
		t, t + 32, 4);
}
