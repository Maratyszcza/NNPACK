#include <psimd/fft/dualreal.h>


void nnp_fft8_dualreal__psimd(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	psimd_f32 s0123 = psimd_load_f32(t +  0);
	psimd_f32 s4567 = psimd_load_f32(t +  4);
	psimd_f32 h0123 = psimd_load_f32(t +  8);
	psimd_f32 h4567 = psimd_load_f32(t + 12);

	psimd_fft8_dualreal_f32(&s0123, &s4567, &h0123, &h4567);

	psimd_store_f32(f +  0, s0123);
	psimd_store_f32(f +  4, s4567);
	psimd_store_f32(f +  8, h0123);
	psimd_store_f32(f + 12, h4567);
}

void nnp_fft16_dualreal__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	psimd_f32 s0123 = psimd_load_f32(t +  0);
	psimd_f32 s4567 = psimd_load_f32(t +  4);
	psimd_f32 s89AB = psimd_load_f32(t +  8);
	psimd_f32 sCDEF = psimd_load_f32(t + 12);
	psimd_f32 h0123 = psimd_load_f32(t + 16);
	psimd_f32 h4567 = psimd_load_f32(t + 20);
	psimd_f32 h89AB = psimd_load_f32(t + 24);
	psimd_f32 hCDEF = psimd_load_f32(t + 28);

	psimd_fft16_dualreal_f32(&s0123, &s4567, &s89AB, &sCDEF, &h0123, &h4567, &h89AB, &hCDEF);

	psimd_store_f32(f +  0, s0123);
	psimd_store_f32(f +  4, s4567);
	psimd_store_f32(f +  8, s89AB);
	psimd_store_f32(f + 12, sCDEF);
	psimd_store_f32(f + 16, h0123);
	psimd_store_f32(f + 20, h4567);
	psimd_store_f32(f + 24, h89AB);
	psimd_store_f32(f + 28, hCDEF);
}

void nnp_ifft8_dualreal__psimd(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	psimd_f32 s0123 = psimd_load_f32(f +  0);
	psimd_f32 s4567 = psimd_load_f32(f +  4);
	psimd_f32 h0123 = psimd_load_f32(f +  8);
	psimd_f32 h4567 = psimd_load_f32(f + 12);

	psimd_ifft8_dualreal_f32(&s0123, &s4567, &h0123, &h4567);

	psimd_store_f32(t +  0, s0123);
	psimd_store_f32(t +  4, s4567);
	psimd_store_f32(t +  8, h0123);
	psimd_store_f32(t + 12, h4567);
}

void nnp_ifft16_dualreal__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	psimd_f32 s0123 = psimd_load_f32(f +  0);
	psimd_f32 s4567 = psimd_load_f32(f +  4);
	psimd_f32 s89AB = psimd_load_f32(f +  8);
	psimd_f32 sCDEF = psimd_load_f32(f + 12);
	psimd_f32 h0123 = psimd_load_f32(f + 16);
	psimd_f32 h4567 = psimd_load_f32(f + 20);
	psimd_f32 h89AB = psimd_load_f32(f + 24);
	psimd_f32 hCDEF = psimd_load_f32(f + 28);

	psimd_ifft16_dualreal_f32(&s0123, &s4567, &s89AB, &sCDEF, &h0123, &h4567, &h89AB, &hCDEF);

	psimd_store_f32(t +  0, s0123);
	psimd_store_f32(t +  4, s4567);
	psimd_store_f32(t +  8, s89AB);
	psimd_store_f32(t + 12, sCDEF);
	psimd_store_f32(t + 16, h0123);
	psimd_store_f32(t + 20, h4567);
	psimd_store_f32(t + 24, h89AB);
	psimd_store_f32(t + 28, hCDEF);
}
