#include <psimd/fft/aos.h>


void nnp_fft4_4aos__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	psimd_f32 w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	psimd_fft4_aos_f32(
		t, t + 16, 4, 0, 8,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i);
	psimd_store_f32(f +  0, w0r);
	psimd_store_f32(f +  4, w0i);
	psimd_store_f32(f +  8, w1r);
	psimd_store_f32(f + 12, w1i);
	psimd_store_f32(f + 16, w2r);
	psimd_store_f32(f + 20, w2i);
	psimd_store_f32(f + 24, w3r);
	psimd_store_f32(f + 28, w3i);
}

void nnp_fft8_4aos__psimd(
	const float t[restrict static 64],
	float f[restrict static 64])
{
	psimd_f32 w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	psimd_fft8_aos_f32(
		t, t + 32, 4, 0, 16,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i, &w4r, &w4i, &w5r, &w5i, &w6r, &w6i, &w7r, &w7i);
	psimd_store_f32(f +  0, w0r);
	psimd_store_f32(f +  4, w0i);
	psimd_store_f32(f +  8, w1r);
	psimd_store_f32(f + 12, w1i);
	psimd_store_f32(f + 16, w2r);
	psimd_store_f32(f + 20, w2i);
	psimd_store_f32(f + 24, w3r);
	psimd_store_f32(f + 28, w3i);
	psimd_store_f32(f + 32, w4r);
	psimd_store_f32(f + 36, w4i);
	psimd_store_f32(f + 40, w5r);
	psimd_store_f32(f + 44, w5i);
	psimd_store_f32(f + 48, w6r);
	psimd_store_f32(f + 52, w6i);
	psimd_store_f32(f + 56, w7r);
	psimd_store_f32(f + 60, w7i);
}

void nnp_ifft4_4aos__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	const psimd_f32 w0r = psimd_load_f32(f +  0);
	const psimd_f32 w0i = psimd_load_f32(f +  4);
	const psimd_f32 w1r = psimd_load_f32(f +  8);
	const psimd_f32 w1i = psimd_load_f32(f + 12);
	const psimd_f32 w2r = psimd_load_f32(f + 16);
	const psimd_f32 w2i = psimd_load_f32(f + 20);
	const psimd_f32 w3r = psimd_load_f32(f + 24);
	const psimd_f32 w3i = psimd_load_f32(f + 28);

	psimd_ifft4_aos_f32(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i,
		t, t + 16, 4);
}

void nnp_ifft8_4aos__psimd(
	const float f[restrict static 64],
	float t[restrict static 64])
{
	const psimd_f32 w0r = psimd_load_f32(f +  0);
	const psimd_f32 w0i = psimd_load_f32(f +  4);
	const psimd_f32 w1r = psimd_load_f32(f +  8);
	const psimd_f32 w1i = psimd_load_f32(f + 12);
	const psimd_f32 w2r = psimd_load_f32(f + 16);
	const psimd_f32 w2i = psimd_load_f32(f + 20);
	const psimd_f32 w3r = psimd_load_f32(f + 24);
	const psimd_f32 w3i = psimd_load_f32(f + 28);
	const psimd_f32 w4r = psimd_load_f32(f + 32);
	const psimd_f32 w4i = psimd_load_f32(f + 36);
	const psimd_f32 w5r = psimd_load_f32(f + 40);
	const psimd_f32 w5i = psimd_load_f32(f + 44);
	const psimd_f32 w6r = psimd_load_f32(f + 48);
	const psimd_f32 w6i = psimd_load_f32(f + 52);
	const psimd_f32 w7r = psimd_load_f32(f + 56);
	const psimd_f32 w7i = psimd_load_f32(f + 60);

	psimd_ifft8_aos_f32(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
		t, t + 32, 4);
}
