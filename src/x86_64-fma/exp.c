#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

__m256 _mm256_exp_ps(__m256 x) {
	const __m256 magic_bias = _mm256_set1_ps(0x1.800000p+23f);
	const __m256 zero_cutoff = _mm256_set1_ps(-0x1.9FE368p+6f); /* The smallest x for which expf(x) is non-zero */
	const __m256 inf_cutoff = _mm256_set1_ps(0x1.62E42Ep+6f); /* The largest x for which expf(x) is finite */
	const __m256 log2e = _mm256_set1_ps(0x1.715476p+3f);
	const __m256 minus_ln2_hi = _mm256_set1_ps(-0x1.62E430p-4f);
	const __m256 minus_ln2_lo = _mm256_set1_ps( 0x1.05C610p-32f);
	const __m256 plus_inf = _mm256_set1_ps(__builtin_inff());

	const __m256 c2 = _mm256_set1_ps(0x1.00088Ap-1f);
	const __m256 c3 = _mm256_set1_ps(0x1.555A86p-3f);
	const __m256 table = _mm256_set_ps(0x1.D5818Ep+0f, 0x1.AE89FAp+0f, 0x1.8ACE54p+0f, 0x1.6A09E6p+0f, 0x1.4BFDAEp+0f, 0x1.306FE0p+0f, 0x1.172B84p+0f, 0x1.000000p+0f);

	const __m256i min_exponent = _mm256_set1_epi32(-126 << 23);
	const __m256i max_exponent = _mm256_set1_epi32(127 << 23);
	const __m256i default_exponent = _mm256_set1_epi32(0x3F800000u);
	const __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFF8);

	__m256 t = _mm256_fmadd_ps(x, log2e, magic_bias);
	__m256i e1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_castps_si256(t), mantissa_mask), 20);
	__m256i e2 = e1;
	e1 = _mm256_min_epi32(_mm256_max_epi32(e1, min_exponent), max_exponent);
	e2 = _mm256_sub_epi32(e2, e1);
	const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(e1, default_exponent));
	const __m256 s2 = _mm256_castsi256_ps(_mm256_add_epi32(e2, default_exponent));
	const __m256 tf = _mm256_permutevar8x32_ps(table, _mm256_castps_si256(t));
	t = _mm256_sub_ps(t, magic_bias);
	const __m256 rx = _mm256_fmadd_ps(t, minus_ln2_lo, _mm256_fmadd_ps(t, minus_ln2_hi, x));
	const __m256 rf = _mm256_fmadd_ps(rx, _mm256_mul_ps(rx, _mm256_fmadd_ps(rx, c3, c2)), rx);
	__m256 f = _mm256_fmadd_ps(tf, rf, tf);
	f = _mm256_mul_ps(s2, _mm256_mul_ps(s1, f));
	/* Fixup underflow to zero */
	f = _mm256_andnot_ps(_mm256_cmp_ps(x, zero_cutoff, _CMP_LT_OS), f);
	/* Fixup overflow */
	f = _mm256_blendv_ps(f, plus_inf, _mm256_cmp_ps(x, inf_cutoff, _CMP_GT_OS));
	/* Fixup NaN */
	f = _mm256_blendv_ps(x, f, _mm256_cmp_ps(x, x, _CMP_EQ_OS));
	return f;
}

static inline uint32_t as_uint32(float x) {
	union {
		float x;
		uint32_t n;
	} data = {
		.x = x
	};
	return data.n;
}

static inline float as_float(uint32_t n) {
	union {
		float x;
		uint32_t n;
	} data = {
		.n = n
	};
	return data.x;
}

static inline float ulpf(float x) {
	const float absx = fabsf(x);
	if (absx < __builtin_inff()) {
		return as_float(as_uint32(absx) + 1) - absx;
	} else {
		return absx;
	}
}

int main() {
	float max_error = 0.0f;
	for (uint32_t n = INT32_MIN; n < as_uint32(-0x1.9FE368p+6f); n++) {
		const float x = as_float(n);
		const float ref_y = expf(x);
		const float opt_y = _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_exp_ps(_mm256_set1_ps(x))));
		const float error = fabsf(ref_y - opt_y) / ulpf(ref_y);
		if (error > max_error)
			max_error = error;
	}
	printf("Max error: %.2f ULP\n", max_error);

	max_error = 0.0f;
	for (uint32_t n = 0; n < as_uint32(0x1.62E42Ep+6f); n++) {
		const float x = as_float(n);
		const float ref_y = expf(x);
		const float opt_y = _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_exp_ps(_mm256_set1_ps(x))));
		const float error = fabsf(ref_y - opt_y) / ulpf(ref_y);
		if (error > max_error)
			max_error = error;
	}
	printf("Max error: %.2f ULP\n", max_error);
}
