#pragma once

#include <nnpack/simd.h>


static inline v4f v4f_exp(v4f x) {
	const v4f magic_bias = v4f_splat(0x1.800000p+23f);
    const v4f zero_cutoff = v4f_splat(-0x1.9FE368p+6f); /* The smallest x for which expf(x) is non-zero */
    const v4f inf_cutoff = v4f_splat(0x1.62E42Ep+6f); /* The largest x for which expf(x) is finite */
    const v4f log2e  = v4f_splat(0x1.715476p+0f);
	const v4f ln2_hi = v4f_splat(0x1.62E400p-1f); /* The lowest 7 bits are zeros */
	const v4f ln2_lo = v4f_splat(0x1.7F7D1Cp-20f);
    const v4f plus_inf = v4f_splat(__builtin_inff());

    const v4f c2 = v4f_splat(0x1.FFFFFCp-2f);
    const v4f c3 = v4f_splat(0x1.55548Cp-3f);
    const v4f c4 = v4f_splat(0x1.555834p-5f);
    const v4f c5 = v4f_splat(0x1.123CFEp-7f);
    const v4f c6 = v4f_splat(0x1.6ADCAEp-10f);

    const v4i min_exponent = v4i_splat((int32_t)((uint32_t) -126 << 23));
    const v4i max_exponent = v4i_splat(127 << 23);
    const v4i default_exponent = v4i_splat(0x3F800000);

    v4f t = x * log2e + magic_bias;
    v4i e1 = ((v4i) t) << v4i_splat(23);
    v4i e2 = e1;
    e1 = v4i_min(v4i_max(e1, min_exponent), max_exponent);
    e2 = e2 - e1;

    const v4f s1 = (v4f) (e1 + default_exponent);
    const v4f s2 = (v4f) (e2 + default_exponent);

    t = t - magic_bias;
	const v4f rx = (x - t * ln2_hi) - t * ln2_lo;
	const v4f rf = rx  + rx * rx * (c2 + rx * (c3 + rx * (c4 + rx * (c5 + rx * c6))));
	v4f f = s2 * (s1 * rf + s1);

    /* Fixup underflow to zero */
    f = v4f_andi(f, x > zero_cutoff);

    /* Fixup overflow */
    f = v4f_blend(x > inf_cutoff, plus_inf, f);
    return f;
}
