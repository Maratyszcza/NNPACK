#include <stddef.h>

#include <psimd/exp.h>

#include <nnpack/softmax.h>


void nnp_vector_exp__psimd(
	size_t n,
	const float x[restrict static n],
	float y[restrict static n])
{
	do {
		v4f_st(y, v4f_exp(v4f_ld(x)));

		y += 4;
		x += 4;
		n -= 4;
	} while (n >= 4);
	if (n != 0) {
		v4f_st(y + n - 4, v4f_exp(v4f_ld(x + n - 4)));
	}
}
