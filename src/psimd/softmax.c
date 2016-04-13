#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <nnpack/simd.h>
#include <nnpack/utils.h>
#include <nnpack/softmax.h>

#include <psimd/exp.h>


static float max__scalar(size_t n, const float v[restrict static n]) {
	float max_v = *v++;
	while (--n) {
		max_v = maxf(max_v, *v++);
	}
	return max_v;
}

static v4f max__psimd(size_t n, const float v[restrict static n]) {
	NNP_ALIGN(16) static const v4i mask[3] = {
		(v4i) { UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0xFFFFFFFF) },
		(v4i) { UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0xFFFFFFFF), UINT32_C(0xFFFFFFFF) },
		(v4i) { UINT32_C(0x00000000), UINT32_C(0xFFFFFFFF), UINT32_C(0xFFFFFFFF), UINT32_C(0xFFFFFFFF) },
	};

	v4f max0, max1, max2, max3;
	max0 = max1 = max2 = max3 = v4f_ld(v);
	v += 4;
	n -= 4;
	while (n >= 16) {
		max0 = v4f_max(max0, v4f_ld(v +  0));
		max1 = v4f_max(max1, v4f_ld(v +  4));
		max2 = v4f_max(max2, v4f_ld(v +  8));
		max3 = v4f_max(max3, v4f_ld(v + 12));

		v += 16;
		n -= 16;
	}
	max0 = v4f_max(v4f_max(max0, max1), v4f_max(max2, max3));
	while (n >= 4) {
		max0 = v4f_max(max0, v4f_ld(v));

		v += 4;
		n -= 4;
	}
	if (n != 0) {
		max0 = v4f_max(max0, v4f_blend(mask[n - 1], v4f_ld(v + n - 4), max0));
	}
	max0 = v4f_max(max0, __builtin_shufflevector(max0, max0, 2, 3, 0, 1));
	max0 = v4f_max(max0, __builtin_shufflevector(max0, max0, 1, 0, 3, 2));
	return max0;
}

static float sum_exp_minus_c__scalar(size_t n, const float v[restrict static n], float c) {
	float sum = 0.0f;
	do {
		sum += expf(*v++ - c);
	} while (--n);
	return sum;
}

static float sum_exp_minus_c__psimd(size_t n, const float v[restrict static n], v4f c) {
	NNP_ALIGN(16) static const v4i mask[3] = {
		(v4i) { UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0xFFFFFFFF) },
		(v4i) { UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0xFFFFFFFF), UINT32_C(0xFFFFFFFF) },
		(v4i) { UINT32_C(0x00000000), UINT32_C(0xFFFFFFFF), UINT32_C(0xFFFFFFFF), UINT32_C(0xFFFFFFFF) },
	};
	v4f sum0, sum1, sum2, sum3;
	sum0 = sum1 = sum2 = sum3 = v4f_zero();
	while (n >= 16) {
		sum0 += v4f_exp(v4f_ld(v +  0) - c);
		sum1 += v4f_exp(v4f_ld(v +  4) - c);
		sum2 += v4f_exp(v4f_ld(v +  8) - c);
		sum3 += v4f_exp(v4f_ld(v + 12) - c);

		v += 16;
		n -= 16;
	}
	sum0 = (sum0 + sum1) + (sum2 + sum3);
	while (n >= 4) {
		sum0 += v4f_exp(v4f_ld(v) - c);

		v += 4;
		n -= 4;
	}
	if (n != 0) {
		sum0 += v4f_exp(v4f_andi(v4f_ld(v + n - 4), mask[n - 1]) - c);
	}
	return v4f_reduce_sum(sum0);
}

static void scaled_exp_minus_c__scalar(size_t n, const float x[restrict static n], float y[restrict static n], float scale, float c) {
	do {
		*y++ = scale * expf(*x++ - c);
	} while (--n);
}

static void inplace_scaled_exp_minus_c__psimd(size_t n, float v[restrict static n], v4f scale, v4f c) {
	const v4f vlast = scale * v4f_exp(v4f_ld(v + n - 4) - c);
	while (n >= 16) {
		const v4f v0 = scale * v4f_exp(v4f_ld(v +  0) - c);
		const v4f v1 = scale * v4f_exp(v4f_ld(v +  4) - c);
		const v4f v2 = scale * v4f_exp(v4f_ld(v +  8) - c);
		const v4f v3 = scale * v4f_exp(v4f_ld(v + 12) - c);

		v4f_st(v +  0, v0);
		v4f_st(v +  4, v1);
		v4f_st(v +  8, v2);
		v4f_st(v + 12, v3);

		v += 16;
		n -= 16;
	}
	while (n >= 4) {
		v4f_st(v, scale * v4f_exp(v4f_ld(v) - c));

		v += 4;
		n -= 4;
	}
	if (n != 0) {
		v4f_st(v + n - 4, vlast);
	}
}

static void outplace_scaled_exp_minus_c__psimd(size_t n, const float x[restrict static n], float y[restrict static n], v4f scale, v4f c) {
	const v4f ylast = scale * v4f_exp(v4f_ld(x + n - 4) - c);
	while (n >= 16) {
		const v4f y0 = scale * v4f_exp(v4f_ld(x +  0) - c);
		const v4f y1 = scale * v4f_exp(v4f_ld(x +  4) - c);
		const v4f y2 = scale * v4f_exp(v4f_ld(x +  8) - c);
		const v4f y3 = scale * v4f_exp(v4f_ld(x + 12) - c);

		v4f_st(y +  0, y0);
		v4f_st(y +  4, y1);
		v4f_st(y +  8, y2);
		v4f_st(y + 12, y3);

		x += 16;
		y += 16;
		n -= 16;
	}
	while (n >= 4) {
		v4f_st(y, scale * v4f_exp(v4f_ld(x) - c));

		x += 4;
		y += 4;
		n -= 4;
	}
	if (n != 0) {
		v4f_st(y + n - 4, ylast);
	}
}

void nnp_inplace_softmax__psimd(
	size_t n,
	float v[restrict static n])
{
	if (n >= 4) {
		const v4f c = max__psimd(n, v);
		const float sum = sum_exp_minus_c__psimd(n, v, c);
		const v4f scale = v4f_splat(1.0f / sum);
		inplace_scaled_exp_minus_c__psimd(n, v, scale, c);
	} else {
		const float c = max__scalar(n, v);
		const float sum = sum_exp_minus_c__scalar(n, v, c);
		const float scale = 1.0f / sum;
		scaled_exp_minus_c__scalar(n, v, v, scale, c);
	}
}

void nnp_outplace_softmax__psimd(
	size_t n,
	const float x[restrict static n],
	float y[restrict static n])
{
	if (n >= 4) {
		const v4f c = max__psimd(n, x);
		const float sum = sum_exp_minus_c__psimd(n, x, c);
		const v4f scale = v4f_splat(1.0f / sum);
		outplace_scaled_exp_minus_c__psimd(n, x, y, scale, c);
	} else {
		const float c = max__scalar(n, x);
		const float sum = sum_exp_minus_c__scalar(n, x, c);
		const float scale = 1.0f / sum;
		scaled_exp_minus_c__scalar(n, x, y, scale, c);
	}
}
