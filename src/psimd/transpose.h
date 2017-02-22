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
    const psimd_f32 row01lo = __builtin_shufflevector(row0, row1, 0, 4, 1, 5);
    const psimd_f32 row01hi = __builtin_shufflevector(row0, row1, 2, 6, 3, 7);
    const psimd_f32 row23lo = __builtin_shufflevector(row2, row3, 0, 4, 1, 5);
    const psimd_f32 row23hi = __builtin_shufflevector(row2, row3, 2, 6, 3, 7);

    /*
     * col0 = ( x00 x10 x20 x30 )
     * col1 = ( x01 x11 x21 x31 )
     * col2 = ( x02 x12 x22 x32 )
     * col3 = ( x03 x13 x23 x33 )
     */
    *col0 = __builtin_shufflevector(row01lo, row23lo, 0, 1, 4, 5);
    *col1 = __builtin_shufflevector(row01lo, row23lo, 2, 3, 6, 7);
    *col2 = __builtin_shufflevector(row01hi, row23hi, 0, 1, 4, 5);
    *col3 = __builtin_shufflevector(row01hi, row23hi, 2, 3, 6, 7);
}
