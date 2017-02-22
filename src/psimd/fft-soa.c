#include <psimd/fft/soa.h>


void nnp_fft8_soa__psimd(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	psimd_f32 w0123r = psimd_load_f32(t +  0);
	psimd_f32 w4567r = psimd_load_f32(t +  4);
	psimd_f32 w0123i = psimd_load_f32(t +  8);
	psimd_f32 w4567i = psimd_load_f32(t + 12);

	psimd_fft8_soa_f32(&w0123r, &w4567r, &w0123i, &w4567i);

	psimd_store_f32(f +  0, w0123r);
	psimd_store_f32(f +  4, w4567r);
	psimd_store_f32(f +  8, w0123i);
	psimd_store_f32(f + 12, w4567i);
}

void nnp_fft16_soa__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	psimd_f32 w0123r = psimd_load_f32(t +  0);
	psimd_f32 w4567r = psimd_load_f32(t +  4);
	psimd_f32 w89ABr = psimd_load_f32(t +  8);
	psimd_f32 wCDEFr = psimd_load_f32(t + 12);
	psimd_f32 w0123i = psimd_load_f32(t + 16);
	psimd_f32 w4567i = psimd_load_f32(t + 20);
	psimd_f32 w89ABi = psimd_load_f32(t + 24);
	psimd_f32 wCDEFi = psimd_load_f32(t + 28);

	psimd_fft16_soa_f32(&w0123r, &w4567r, &w89ABr, &wCDEFr, &w0123i, &w4567i, &w89ABi, &wCDEFi);

	psimd_store_f32(f +  0, w0123r);
	psimd_store_f32(f +  4, w4567r);
	psimd_store_f32(f +  8, w89ABr);
	psimd_store_f32(f + 12, wCDEFr);
	psimd_store_f32(f + 16, w0123i);
	psimd_store_f32(f + 20, w4567i);
	psimd_store_f32(f + 24, w89ABi);
	psimd_store_f32(f + 28, wCDEFi);
}

void nnp_ifft8_soa__psimd(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	psimd_f32 w0123r = psimd_load_f32(f +  0);
	psimd_f32 w4567r = psimd_load_f32(f +  4);
	psimd_f32 w0123i = psimd_load_f32(f +  8);
	psimd_f32 w4567i = psimd_load_f32(f + 12);

	psimd_ifft8_soa_f32(&w0123r, &w4567r, &w0123i, &w4567i);

	psimd_store_f32(t +  0, w0123r);
	psimd_store_f32(t +  4, w4567r);
	psimd_store_f32(t +  8, w0123i);
	psimd_store_f32(t + 12, w4567i);
}

void nnp_ifft16_soa__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	psimd_f32 w0123r = psimd_load_f32(f +  0);
	psimd_f32 w4567r = psimd_load_f32(f +  4);
	psimd_f32 w89ABr = psimd_load_f32(f +  8);
	psimd_f32 wCDEFr = psimd_load_f32(f + 12);
	psimd_f32 w0123i = psimd_load_f32(f + 16);
	psimd_f32 w4567i = psimd_load_f32(f + 20);
	psimd_f32 w89ABi = psimd_load_f32(f + 24);
	psimd_f32 wCDEFi = psimd_load_f32(f + 28);

	psimd_ifft16_soa_f32(&w0123r, &w4567r, &w89ABr, &wCDEFr, &w0123i, &w4567i, &w89ABi, &wCDEFi);

	psimd_store_f32(t +  0, w0123r);
	psimd_store_f32(t +  4, w4567r);
	psimd_store_f32(t +  8, w89ABr);
	psimd_store_f32(t + 12, wCDEFr);
	psimd_store_f32(t + 16, w0123i);
	psimd_store_f32(t + 20, w4567i);
	psimd_store_f32(t + 24, w89ABi);
	psimd_store_f32(t + 28, wCDEFi);
}
