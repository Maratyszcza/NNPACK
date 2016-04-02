#pragma once

#include <nnpack/simd.h>


static inline void v4f_butterfly(v4f a[restrict static 1], v4f b[restrict static 1]) {
    const v4f new_a = *a + *b;
    const v4f new_b = *a - *b;
    *a = new_a;
    *b = new_b;
}

static inline void v4f_butterfly_and_negate_b(v4f a[restrict static 1], v4f b[restrict static 1]) {
    const v4f new_a = *a + *b;
    const v4f new_b = *b - *a;
    *a = new_a;
    *b = new_b;
}

static inline void v4f_butterfly_with_negated_b(v4f a[restrict static 1], v4f b[restrict static 1]) {
    const v4f new_a = *a - *b;
    const v4f new_b = *a + *b;
    *a = new_a;
    *b = new_b;
}
