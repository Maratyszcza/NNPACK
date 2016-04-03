#pragma once

#include <nnpack/simd.h>


static inline void v4f_transpose4x4(
    const v4f row0, const v4f row1, const v4f row2, const v4f row3,
    v4f col0[restrict static 1],
    v4f col1[restrict static 1],
    v4f col2[restrict static 1],
    v4f col3[restrict static 1])
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
    const v4f row01lo = __builtin_shufflevector(row0, row1, 0, 4, 1, 5);
    const v4f row01hi = __builtin_shufflevector(row0, row1, 2, 6, 3, 7);
    const v4f row23lo = __builtin_shufflevector(row2, row3, 0, 4, 1, 5);
    const v4f row23hi = __builtin_shufflevector(row2, row3, 2, 6, 3, 7);

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
