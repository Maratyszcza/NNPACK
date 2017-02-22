#include <stddef.h>

#include <psimd.h>


void nnp_s8x8gemm__psimd(
	float acc[restrict static 8 * 8],
	const float x[restrict static 8 * 8],
	const float y[restrict static 8 * 8])
{
	for (size_t row = 0; row < 8; row++) {
		psimd_store_f32(acc + 0, psimd_load_f32(acc + 0) + psimd_load_f32(x + 0) * psimd_load_f32(y + 0));
		psimd_store_f32(acc + 4, psimd_load_f32(acc + 4) + psimd_load_f32(x + 4) * psimd_load_f32(y + 4));

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
		const psimd_f32 xr = psimd_load_f32(x + 0);
		const psimd_f32 xi = psimd_load_f32(x + 4);

		const psimd_f32 yr = psimd_load_f32(y + 0);
		psimd_f32 accr = psimd_load_f32(acc + 0) + xr * yr;
		psimd_f32 acci = psimd_load_f32(acc + 4) + xi * yr;

		const psimd_f32 yi = psimd_load_f32(y + 4);
		accr += xi * yi;
		psimd_store_f32(acc + 0, accr);
		acci -= xr * yi;
		psimd_store_f32(acc + 4, acci);

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
		const psimd_f32 xr = psimd_load_f32(x + 0);
		const psimd_f32 xi = psimd_load_f32(x + 4);

		const psimd_f32 yr = psimd_load_f32(y + 0);
		psimd_f32 accr = psimd_load_f32(acc + 0) + xr * yr;
		psimd_f32 acci = psimd_load_f32(acc + 4) + xi * yr;

		const psimd_f32 yi = psimd_load_f32(y + 4);
		accr += xi * yi;
		psimd_store_f32(acc + 0, accr);
		acci -= xr * yi;
		psimd_store_f32(acc + 4, acci);

		acc += 8;
		x += 8;
		y += 8;
	}
}
