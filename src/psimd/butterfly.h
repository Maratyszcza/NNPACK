#pragma once

#include <psimd.h>


static inline void psimd_butterfly_f32(
    psimd_f32 a[restrict static 1],
    psimd_f32 b[restrict static 1])
{
    const psimd_f32 new_a = *a + *b;
    const psimd_f32 new_b = *a - *b;
    *a = new_a;
    *b = new_b;
}

static inline void psimd_butterfly_and_negate_b_f32(
    psimd_f32 a[restrict static 1],
    psimd_f32 b[restrict static 1])
{
    const psimd_f32 new_a = *a + *b;
    const psimd_f32 new_b = *b - *a;
    *a = new_a;
    *b = new_b;
}

static inline void psimd_butterfly_with_negated_b_f32(
    psimd_f32 a[restrict static 1],
    psimd_f32 b[restrict static 1])
{
    const psimd_f32 new_a = *a - *b;
    const psimd_f32 new_b = *a + *b;
    *a = new_a;
    *b = new_b;
}
