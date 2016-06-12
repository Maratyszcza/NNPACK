#include <stdint.h>
#include <stddef.h>
#include <complex.h>

#include <psimd/fft/real.h>
#include <psimd/fft/soa.h>
#include <psimd/fft/dualreal.h>


void nnp_s8x8gemm__psimd(
	float acc[restrict static 8 * 8],
	const float x[restrict static 8 * 8],
	const float y[restrict static 8 * 8])
{
	for (size_t row = 0; row < 8; row++) {
		v4f_st(acc + 0, v4f_ld(acc + 0) + v4f_ld(x + 0) * v4f_ld(y + 0));
		v4f_st(acc + 4, v4f_ld(acc + 4) + v4f_ld(x + 4) * v4f_ld(y + 4));

		acc += 8;
		x += 8;
		y += 8;
	}
}

void nnp_ft8x8gemmc__psimd(
	float acc[restrict static 8 * 8],
	const float x[restrict static 8 * 8],
	const float y[restrict static 8 * 8])
{
	for (size_t row = 0; row < 8; row++) {
		const v4f xr = v4f_ld(x + 0);
		const v4f xi = v4f_ld(x + 4);

		const v4f yr = v4f_ld(y + 0);
		v4f accr = v4f_ld(acc + 0) + xr * yr;
		v4f acci = v4f_ld(acc + 4) + xi * yr;

		const v4f yi = v4f_ld(y + 4);
		accr += xi * yi;
		v4f_st(acc + 0, accr);
		acci -= xr * yi;
		v4f_st(acc + 4, acci);

		acc += 8;
		x += 8;
		y += 8;
	}
}


void nnp_ft16x16gemmc__psimd(
	float acc[restrict static 16 * 16],
	const float x[restrict static 16 * 16],
	const float y[restrict static 16 * 16])
{
	for (size_t row = 0; row < 16 * 2; row++) {
		const v4f xr = v4f_ld(x + 0);
		const v4f xi = v4f_ld(x + 4);

		const v4f yr = v4f_ld(y + 0);
		v4f accr = v4f_ld(acc + 0) + xr * yr;
		v4f acci = v4f_ld(acc + 4) + xi * yr;

		const v4f yi = v4f_ld(y + 4);
		accr += xi * yi;
		v4f_st(acc + 0, accr);
		acci -= xr * yi;
		v4f_st(acc + 4, acci);

		acc += 8;
		x += 8;
		y += 8;
	}
}
