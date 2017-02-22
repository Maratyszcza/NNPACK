#include <stddef.h>

#include <psimd/exp.h>

#include <nnpack/softmax.h>


void nnp_vector_exp__psimd(
	size_t n,
	const float x[restrict static n],
	float y[restrict static n])
{
	do {
		psimd_store_f32(y,
			psimd_exp_f32(psimd_load_f32(x)));

		y += 4;
		x += 4;
		n -= 4;
	} while (n >= 4);
	if (n != 0) {
		psimd_store_f32(y + n - 4,
			psimd_exp_f32(psimd_load_f32(x + n - 4)));
	}
}
