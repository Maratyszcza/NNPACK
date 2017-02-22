#pragma once

#include <psimd.h>


static inline psimd_f32 psimd_exp_f32(psimd_f32 x) {
	const psimd_f32 magic_bias = psimd_splat_f32(0x1.800000p+23f);
    const psimd_f32 zero_cutoff = psimd_splat_f32(-0x1.9FE368p+6f); /* The smallest x for which expf(x) is non-zero */
    const psimd_f32 inf_cutoff = psimd_splat_f32(0x1.62E42Ep+6f); /* The largest x for which expf(x) is finite */
    const psimd_f32 log2e  = psimd_splat_f32(0x1.715476p+0f);
	const psimd_f32 ln2_hi = psimd_splat_f32(0x1.62E400p-1f); /* The lowest 7 bits are zeros */
	const psimd_f32 ln2_lo = psimd_splat_f32(0x1.7F7D1Cp-20f);
    const psimd_f32 plus_inf = psimd_splat_f32(__builtin_inff());

    const psimd_f32 c2 = psimd_splat_f32(0x1.FFFFFCp-2f);
    const psimd_f32 c3 = psimd_splat_f32(0x1.55548Cp-3f);
    const psimd_f32 c4 = psimd_splat_f32(0x1.555834p-5f);
    const psimd_f32 c5 = psimd_splat_f32(0x1.123CFEp-7f);
    const psimd_f32 c6 = psimd_splat_f32(0x1.6ADCAEp-10f);

    const psimd_s32 min_exponent = psimd_splat_s32((int32_t)((uint32_t) -126 << 23));
    const psimd_s32 max_exponent = psimd_splat_s32(127 << 23);
    const psimd_s32 default_exponent = psimd_splat_s32(0x3F800000);

    psimd_f32 t = x * log2e + magic_bias;
    psimd_s32 e1 = ((psimd_s32) t) << psimd_splat_s32(23);
    psimd_s32 e2 = e1;
    e1 = psimd_min_s32(psimd_max_s32(e1, min_exponent), max_exponent);
    e2 = e2 - e1;

    const psimd_f32 s1 = (psimd_f32) (e1 + default_exponent);
    const psimd_f32 s2 = (psimd_f32) (e2 + default_exponent);

    t = t - magic_bias;
	const psimd_f32 rx = (x - t * ln2_hi) - t * ln2_lo;
	const psimd_f32 rf = rx  + rx * rx * (c2 + rx * (c3 + rx * (c4 + rx * (c5 + rx * c6))));
	psimd_f32 f = s2 * (s1 * rf + s1);

    /* Fixup underflow to zero */
    f = psimd_andmask_f32(x > zero_cutoff, f);

    /* Fixup overflow */
    f = psimd_blend_f32(x > inf_cutoff, plus_inf, f);
    return f;
}
