#include <psimd/fft/soa.h>


void nnp_fft8_soa__psimd(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	v4f w0123r = v4f_ld(t +  0);
	v4f w4567r = v4f_ld(t +  4);
	v4f w0123i = v4f_ld(t +  8);
	v4f w4567i = v4f_ld(t + 12);

	v4f_fft8_soa(&w0123r, &w4567r, &w0123i, &w4567i);

	v4f_st(f +  0, w0123r);
	v4f_st(f +  4, w4567r);
	v4f_st(f +  8, w0123i);
	v4f_st(f + 12, w4567i);
}

void nnp_fft16_soa__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	v4f w0123r = v4f_ld(t +  0);
	v4f w4567r = v4f_ld(t +  4);
	v4f w89ABr = v4f_ld(t +  8);
	v4f wCDEFr = v4f_ld(t + 12);
	v4f w0123i = v4f_ld(t + 16);
	v4f w4567i = v4f_ld(t + 20);
	v4f w89ABi = v4f_ld(t + 24);
	v4f wCDEFi = v4f_ld(t + 28);

	v4f_fft16_soa(&w0123r, &w4567r, &w89ABr, &wCDEFr, &w0123i, &w4567i, &w89ABi, &wCDEFi);

	v4f_st(f +  0, w0123r);
	v4f_st(f +  4, w4567r);
	v4f_st(f +  8, w89ABr);
	v4f_st(f + 12, wCDEFr);
	v4f_st(f + 16, w0123i);
	v4f_st(f + 20, w4567i);
	v4f_st(f + 24, w89ABi);
	v4f_st(f + 28, wCDEFi);
}

void nnp_ifft8_soa__psimd(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	v4f w0123r = v4f_ld(f +  0);
	v4f w4567r = v4f_ld(f +  4);
	v4f w0123i = v4f_ld(f +  8);
	v4f w4567i = v4f_ld(f + 12);

	v4f_ifft8_soa(&w0123r, &w4567r, &w0123i, &w4567i);

	v4f_st(t +  0, w0123r);
	v4f_st(t +  4, w4567r);
	v4f_st(t +  8, w0123i);
	v4f_st(t + 12, w4567i);
}

void nnp_ifft16_soa__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	v4f w0123r = v4f_ld(f +  0);
	v4f w4567r = v4f_ld(f +  4);
	v4f w89ABr = v4f_ld(f +  8);
	v4f wCDEFr = v4f_ld(f + 12);
	v4f w0123i = v4f_ld(f + 16);
	v4f w4567i = v4f_ld(f + 20);
	v4f w89ABi = v4f_ld(f + 24);
	v4f wCDEFi = v4f_ld(f + 28);

	v4f_ifft16_soa(&w0123r, &w4567r, &w89ABr, &wCDEFr, &w0123i, &w4567i, &w89ABi, &wCDEFi);

	v4f_st(t +  0, w0123r);
	v4f_st(t +  4, w4567r);
	v4f_st(t +  8, w89ABr);
	v4f_st(t + 12, wCDEFr);
	v4f_st(t + 16, w0123i);
	v4f_st(t + 20, w4567i);
	v4f_st(t + 24, w89ABi);
	v4f_st(t + 28, wCDEFi);
}
