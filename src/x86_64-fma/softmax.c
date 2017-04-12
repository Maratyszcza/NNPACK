#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <nnpack/utils.h>
#include <nnpack/softmax.h>

float max__avx(size_t n, const float v[restrict static n]);
float sum_exp_minus_c__avx2(size_t n, const float v[restrict static n], float c);
void scaled_exp_minus_c__avx2(size_t n, const float x[restrict static n], float y[restrict static n], float scale, float c);
void inplace_scaled_exp_minus_c__avx2(size_t n, const float v[restrict static n], float scale, float c);

void nnp_softmax__avx2(
	size_t n,
	const float x[restrict static n],
	float y[restrict static n])
{
	const float c = max__avx(n, x);
	const float sum = sum_exp_minus_c__avx2(n, x, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__avx2(n, x, y, scale, c);
}

void nnp_inplace_softmax__avx2(
	size_t n,
	float v[restrict static n])
{
	const float c = max__avx(n, v);
	const float sum = sum_exp_minus_c__avx2(n, v, c);
	const float scale = 1.0f / sum;
	inplace_scaled_exp_minus_c__avx2(n, v, scale, c);
}
