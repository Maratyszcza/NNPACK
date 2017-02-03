#include <psimd/fft/dualreal.h>


void nnp_fft8_dualreal__psimd(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	v4f s0123 = v4f_ld(t +  0);
	v4f s4567 = v4f_ld(t +  4);
	v4f h0123 = v4f_ld(t +  8);
	v4f h4567 = v4f_ld(t + 12);

	v4f_fft8_dualreal(&s0123, &s4567, &h0123, &h4567);

	v4f_st(f +  0, s0123);
	v4f_st(f +  4, s4567);
	v4f_st(f +  8, h0123);
	v4f_st(f + 12, h4567);
}

void nnp_fft16_dualreal__psimd(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	v4f s0123 = v4f_ld(t +  0);
	v4f s4567 = v4f_ld(t +  4);
	v4f s89AB = v4f_ld(t +  8);
	v4f sCDEF = v4f_ld(t + 12);
	v4f h0123 = v4f_ld(t + 16);
	v4f h4567 = v4f_ld(t + 20);
	v4f h89AB = v4f_ld(t + 24);
	v4f hCDEF = v4f_ld(t + 28);

	v4f_fft16_dualreal(&s0123, &s4567, &s89AB, &sCDEF, &h0123, &h4567, &h89AB, &hCDEF);

	v4f_st(f +  0, s0123);
	v4f_st(f +  4, s4567);
	v4f_st(f +  8, s89AB);
	v4f_st(f + 12, sCDEF);
	v4f_st(f + 16, h0123);
	v4f_st(f + 20, h4567);
	v4f_st(f + 24, h89AB);
	v4f_st(f + 28, hCDEF);
}

void nnp_ifft8_dualreal__psimd(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	v4f s0123 = v4f_ld(f +  0);
	v4f s4567 = v4f_ld(f +  4);
	v4f h0123 = v4f_ld(f +  8);
	v4f h4567 = v4f_ld(f + 12);

	v4f_ifft8_dualreal(&s0123, &s4567, &h0123, &h4567);

	v4f_st(t +  0, s0123);
	v4f_st(t +  4, s4567);
	v4f_st(t +  8, h0123);
	v4f_st(t + 12, h4567);
}

void nnp_ifft16_dualreal__psimd(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	v4f s0123 = v4f_ld(f +  0);
	v4f s4567 = v4f_ld(f +  4);
	v4f s89AB = v4f_ld(f +  8);
	v4f sCDEF = v4f_ld(f + 12);
	v4f h0123 = v4f_ld(f + 16);
	v4f h4567 = v4f_ld(f + 20);
	v4f h89AB = v4f_ld(f + 24);
	v4f hCDEF = v4f_ld(f + 28);

	v4f_ifft16_dualreal(&s0123, &s4567, &s89AB, &sCDEF, &h0123, &h4567, &h89AB, &hCDEF);

	v4f_st(t +  0, s0123);
	v4f_st(t +  4, s4567);
	v4f_st(t +  8, s89AB);
	v4f_st(t + 12, sCDEF);
	v4f_st(t + 16, h0123);
	v4f_st(t + 20, h4567);
	v4f_st(t + 24, h89AB);
	v4f_st(t + 28, hCDEF);
}
