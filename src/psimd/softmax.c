#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <nnpack/macros.h>
#include <nnpack/utils.h>
#include <nnpack/softmax.h>

#include <psimd.h>
#include <psimd/exp.h>


static float max__scalar(size_t n, const float v[restrict static n]) {
	float max_v = *v++;
	while (--n) {
		max_v = maxf(max_v, *v++);
	}
	return max_v;
}

static psimd_f32 max__psimd(size_t n, const float v[restrict static n]) {
	NNP_ALIGN(16) static const int32_t mask[12] = {
		0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF,
		0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF,
		0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
	};

	psimd_f32 max0, max1, max2, max3;
	max0 = max1 = max2 = max3 = psimd_load_f32(v);
	v += 4;
	n -= 4;
	while (n >= 16) {
		max0 = psimd_max_f32(max0, psimd_load_f32(v +  0));
		max1 = psimd_max_f32(max1, psimd_load_f32(v +  4));
		max2 = psimd_max_f32(max2, psimd_load_f32(v +  8));
		max3 = psimd_max_f32(max3, psimd_load_f32(v + 12));

		v += 16;
		n -= 16;
	}
	max0 = psimd_max_f32(psimd_max_f32(max0, max1), psimd_max_f32(max2, max3));
	while (n >= 4) {
		max0 = psimd_max_f32(max0, psimd_load_f32(v));

		v += 4;
		n -= 4;
	}
	if (n != 0) {
		max0 = psimd_max_f32(max0, psimd_blend_f32(psimd_load_s32(&mask[4 * (n - 1)]), psimd_load_f32(v + n - 4), max0));
	}
	return psimd_allreduce_max_f32(max0);
}

static float sum_exp_minus_c__scalar(size_t n, const float v[restrict static n], float c) {
	float sum = 0.0f;
	do {
		sum += expf(*v++ - c);
	} while (--n);
	return sum;
}

static float sum_exp_minus_c__psimd(size_t n, const float v[restrict static n], psimd_f32 c) {
	NNP_ALIGN(16) static const int32_t mask[12] = {
		0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF,
		0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF,
		0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
	};
	psimd_f32 sum0, sum1, sum2, sum3;
	sum0 = sum1 = sum2 = sum3 = psimd_zero_f32();
	while (n >= 16) {
		sum0 += psimd_exp_f32(psimd_load_f32(v +  0) - c);
		sum1 += psimd_exp_f32(psimd_load_f32(v +  4) - c);
		sum2 += psimd_exp_f32(psimd_load_f32(v +  8) - c);
		sum3 += psimd_exp_f32(psimd_load_f32(v + 12) - c);

		v += 16;
		n -= 16;
	}
	sum0 = (sum0 + sum1) + (sum2 + sum3);
	while (n >= 4) {
		sum0 += psimd_exp_f32(psimd_load_f32(v) - c);

		v += 4;
		n -= 4;
	}
	if (n != 0) {
		sum0 += psimd_exp_f32(psimd_andmask_f32(psimd_load_s32(&mask[4 * (n - 1)]), psimd_load_f32(v + n - 4)) - c);
	}
	return psimd_reduce_sum_f32(sum0);
}

static void scaled_exp_minus_c__scalar(size_t n, const float x[restrict static n], float y[restrict static n], float scale, float c) {
	do {
		*y++ = scale * expf(*x++ - c);
	} while (--n);
}

static void inplace_scaled_exp_minus_c__psimd(size_t n, float v[restrict static n], psimd_f32 scale, psimd_f32 c) {
	const psimd_f32 vlast = scale * psimd_exp_f32(psimd_load_f32(v + n - 4) - c);
	while (n >= 16) {
		const psimd_f32 v0 = scale * psimd_exp_f32(psimd_load_f32(v +  0) - c);
		const psimd_f32 v1 = scale * psimd_exp_f32(psimd_load_f32(v +  4) - c);
		const psimd_f32 v2 = scale * psimd_exp_f32(psimd_load_f32(v +  8) - c);
		const psimd_f32 v3 = scale * psimd_exp_f32(psimd_load_f32(v + 12) - c);

		psimd_store_f32(v +  0, v0);
		psimd_store_f32(v +  4, v1);
		psimd_store_f32(v +  8, v2);
		psimd_store_f32(v + 12, v3);

		v += 16;
		n -= 16;
	}
	while (n >= 4) {
		psimd_store_f32(v, scale * psimd_exp_f32(psimd_load_f32(v) - c));

		v += 4;
		n -= 4;
	}
	if (n != 0) {
		psimd_store_f32(v + n - 4, vlast);
	}
}

static void outplace_scaled_exp_minus_c__psimd(size_t n, const float x[restrict static n], float y[restrict static n], psimd_f32 scale, psimd_f32 c) {
	const psimd_f32 ylast = scale * psimd_exp_f32(psimd_load_f32(x + n - 4) - c);
	while (n >= 16) {
		const psimd_f32 y0 = scale * psimd_exp_f32(psimd_load_f32(x +  0) - c);
		const psimd_f32 y1 = scale * psimd_exp_f32(psimd_load_f32(x +  4) - c);
		const psimd_f32 y2 = scale * psimd_exp_f32(psimd_load_f32(x +  8) - c);
		const psimd_f32 y3 = scale * psimd_exp_f32(psimd_load_f32(x + 12) - c);

		psimd_store_f32(y +  0, y0);
		psimd_store_f32(y +  4, y1);
		psimd_store_f32(y +  8, y2);
		psimd_store_f32(y + 12, y3);

		x += 16;
		y += 16;
		n -= 16;
	}
	while (n >= 4) {
		psimd_store_f32(y, scale * psimd_exp_f32(psimd_load_f32(x) - c));

		x += 4;
		y += 4;
		n -= 4;
	}
	if (n != 0) {
		psimd_store_f32(y + n - 4, ylast);
	}
}

void nnp_softmax__psimd(
	size_t n,
	const float x[restrict static n],
	float y[restrict static n])
{
	if (n >= 4) {
		const psimd_f32 c = max__psimd(n, x);
		const float sum = sum_exp_minus_c__psimd(n, x, c);
		const psimd_f32 scale = psimd_splat_f32(1.0f / sum);
		outplace_scaled_exp_minus_c__psimd(n, x, y, scale, c);
	} else {
		const float c = max__scalar(n, x);
		const float sum = sum_exp_minus_c__scalar(n, x, c);
		const float scale = 1.0f / sum;
		scaled_exp_minus_c__scalar(n, x, y, scale, c);
	}
}
void nnp_inplace_softmax__psimd(
	size_t n,
	float v[restrict static n])
{
	if (n >= 4) {
		const psimd_f32 c = max__psimd(n, v);
		const float sum = sum_exp_minus_c__psimd(n, v, c);
		const psimd_f32 scale = psimd_splat_f32(1.0f / sum);
		inplace_scaled_exp_minus_c__psimd(n, v, scale, c);
	} else {
		const float c = max__scalar(n, v);
		const float sum = sum_exp_minus_c__scalar(n, v, c);
		const float scale = 1.0f / sum;
		scaled_exp_minus_c__scalar(n, v, v, scale, c);
	}
}
