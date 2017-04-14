#pragma once

#include <psimd.h>


static inline void psimd_transpose4x4_f32(
    const psimd_f32 row0, const psimd_f32 row1, const psimd_f32 row2, const psimd_f32 row3,
    psimd_f32 col0[restrict static 1],
    psimd_f32 col1[restrict static 1],
    psimd_f32 col2[restrict static 1],
    psimd_f32 col3[restrict static 1])
{
    /*
     * row0 = ( x00 x01 x02 x03 )
     * row1 = ( x10 x11 x12 x13 )
     * row2 = ( x20 x21 x22 x23 )
     * row3 = ( x30 x31 x32 x33 )
     */

    /*
     * row01lo = ( x00 x10 x01 x11 )
     * row01hi = ( x02 x12 x03 x13 )
     * row23lo = ( x20 x30 x21 x31 )
     * row23hi = ( x22 x32 x23 x33 )
     */
    const psimd_f32 row01lo = psimd_interleave_lo_f32(row0, row1);
    const psimd_f32 row01hi = psimd_interleave_hi_f32(row0, row1);
    const psimd_f32 row23lo = psimd_interleave_lo_f32(row2, row3);
    const psimd_f32 row23hi = psimd_interleave_hi_f32(row2, row3);

    /*
     * col0 = ( x00 x10 x20 x30 )
     * col1 = ( x01 x11 x21 x31 )
     * col2 = ( x02 x12 x22 x32 )
     * col3 = ( x03 x13 x23 x33 )
     */
    *col0 = psimd_concat_lo_f32(row01lo, row23lo);
    *col1 = psimd_concat_hi_f32(row01lo, row23lo);
    *col2 = psimd_concat_lo_f32(row01hi, row23hi);
    *col3 = psimd_concat_hi_f32(row01hi, row23hi);
}
