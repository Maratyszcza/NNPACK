from __future__ import absolute_import
from __future__ import division


from peachpy import *
from peachpy.x86_64 import *


from common import _MM_SHUFFLE
from common import transpose2x2x128, transpose2x2x2x64, butterfly


def input_transform(ymm_d):
    assert isinstance(ymm_d, list) and len(ymm_d) == 8 and all(isinstance(ymm, YMMRegister) for ymm in ymm_d)

    ymm_wd = [YMMRegister() for _ in range(8)]

    # wd0 = (d0 - d6) + 5.25 * (d4 - d2)
    # wd1 = (d6 + d2 - 4.25 * d4) + (d1 + d5 - 4.25 * d3)
    # wd2 = (d6 + d2 - 4.25 * d4) - (d1 + d5 - 4.25 * d3)
    # wd3 = (d6 + 0.25 * d2 - 1.25 * d4) + 2.0 * (d5 + 0.25 * d1 - 1.25 * d3)
    # wd4 = (d6 + 0.25 * d2 - 1.25 * d4) - 2.0 * (d5 + 0.25 * d1 - 1.25 * d3)
    # wd5 = (d6 - 5.0 * d4 + 4.0 * d2) + 2.0 * (d1 + 0.25 * d5 - 1.25 * d3)
    # wd6 = (d6 - 5.0 * d4 + 4.0 * d2) - 2.0 * (d1 + 0.25 * d5 - 1.25 * d3)
    # wd7 = (d7 - d1) + 5.25 * (d3 - d5)

    ymm_0_25 = YMMRegister()
    VMOVAPS(ymm_0_25, Constant.float32x8(0.25))

    # Compute wd0 := d0 - d6
    VSUBPS(ymm_wd[0], ymm_d[0], ymm_d[6])
    ymm_d4_sub_d2 = YMMRegister()
    VSUBPS(ymm_d4_sub_d2, ymm_d[4], ymm_d[2])
    # Compute wd7 := d7 - d1
    VSUBPS(ymm_wd[7], ymm_d[7], ymm_d[1])
    ymm_d3_sub_d5 = YMMRegister()
    VSUBPS(ymm_d3_sub_d5, ymm_d[3], ymm_d[5])
    # Compute wd1 := d2 + d6
    VADDPS(ymm_wd[1], ymm_d[2], ymm_d[6])
    # Compute wd2 := d1 + d5
    VADDPS(ymm_wd[2], ymm_d[1], ymm_d[5])
    # Compute wd4 := d5 + 0.25 * d1
    VMOVAPS(ymm_wd[4], ymm_d[5])
    VFMADD231PS(ymm_wd[4], ymm_d[1], ymm_0_25)
    # Compute wd5 := d6 - 5.0 * d4
    VMOVAPS(ymm_wd[5], Constant.float32x8(5.0))
    VFNMADD132PS(ymm_wd[5], ymm_d[6], ymm_d[4])
    # Compute wd3 := d6 + 0.25 * d2
    VFMADD231PS(ymm_d[6], ymm_d[2], ymm_0_25)
    SWAP.REGISTERS(ymm_wd[3], ymm_d[6])
    # Compute wd6 := d1 + 0.25 * d5
    VFMADD231PS(ymm_d[1], ymm_d[5], ymm_0_25)
    SWAP.REGISTERS(ymm_wd[6], ymm_d[1])

    ymm_5_25 = YMMRegister()
    VMOVAPS(ymm_5_25, Constant.float32x8(5.25))
    # Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
    VFMADD231PS(ymm_wd[0], ymm_d4_sub_d2, ymm_5_25)
    # Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
    VFMADD231PS(ymm_wd[7], ymm_d3_sub_d5, ymm_5_25)

    ymm_4_25 = YMMRegister()
    VMOVAPS(ymm_4_25, Constant.float32x8(4.25))
    # Compute
    #   wd1 := (d6 + d2) - 4.25 * d4
    #   wd2 := (d1 + d5) - 4.25 * d3
    VFNMADD231PS(ymm_wd[1], ymm_d[4], ymm_4_25)
    VFNMADD231PS(ymm_wd[2], ymm_d[3], ymm_4_25)

    ymm_1_25 = YMMRegister()
    VMOVAPS(ymm_1_25, Constant.float32x8(1.25))
    # Compute
    #   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
    #   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
    #   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
    #   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
    VFNMADD231PS(ymm_wd[3], ymm_d[4], ymm_1_25)
    VFNMADD231PS(ymm_wd[4], ymm_d[3], ymm_1_25)
    VFMADD231PS(ymm_wd[5], ymm_d[2], Constant.float32x8(4.0))
    VFNMADD231PS(ymm_wd[6], ymm_d[3], ymm_1_25)

    ymm_2 = YMMRegister()
    VMOVAPS(ymm_2, Constant.float32x8(2.0))
    butterfly(ymm_wd[1], ymm_wd[2])
    butterfly(ymm_wd[3], ymm_wd[4], scale_b=ymm_2)
    butterfly(ymm_wd[5], ymm_wd[6], scale_b=ymm_2)

    return ymm_wd


def kernel_transform(g, rescale_coefficients=True):
    assert isinstance(g, list) and len(g) == 3 and \
        (all(isinstance(reg, XMMRegister) for reg in g) or all(isinstance(reg, YMMRegister) for reg in g))

    minus_2_over_9 = float.fromhex("-0x1.C71C72p-3")
    rcp_90         = float.fromhex( "0x1.6C16C2p-7")
    rcp_180        = float.fromhex( "0x1.6C16C2p-8")

    if isinstance(g[0], XMMRegister):
        wg = [XMMRegister() for _ in range(8)]
        const_2              = Constant.float32x4(2.0)
        const_4              = Constant.float32x4(4.0)
        const_minus_2_over_9 = Constant.float32x4(minus_2_over_9)
        const_rcp_90         = Constant.float32x4(rcp_90)
        const_rcp_180        = Constant.float32x4(rcp_180)
    else:
        wg = [YMMRegister() for _ in range(8)]
        const_2              = Constant.float32x8(2.0)
        const_4              = Constant.float32x8(4.0)
        const_minus_2_over_9 = Constant.float32x8(minus_2_over_9)
        const_rcp_90         = Constant.float32x8(rcp_90)
        const_rcp_180        = Constant.float32x8(rcp_180)

    # wg[0] = g0
    # wg[1] = ((g0 + g2) + g1) * (-2.0 / 9)
    # wg[2] = ((g0 + g2) - g1) * (-2.0 / 9)
    # wg[3] = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
    # wg[4] = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
    # wg[5] = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
    # wg[6] = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
    # wg[7] = g2

    # Compute wg[1] := g0 + g2
    VADDPS(wg[1],  g[0], g[2])
    # Compute
    #   wg[3] := g0 + 4 * g2
    #   wg[5] := g2 + 4 * g0
    VMOVAPS(wg[3], const_4)
    VMOVAPS(wg[5], wg[3])
    VFMADD132PS(wg[3], g[0], g[2])
    VFMADD132PS(wg[5], g[2], g[0])

    # Compute wg[1] and wg[2]
    VSUBPS(wg[2], wg[1], g[1])
    VADDPS(wg[1], wg[1], g[1])

    var_2 = YMMRegister() if isinstance(g[0], YMMRegister) else XMMRegister()
    VMOVAPS(var_2, const_2)

    # Compute wg[3] and wg[4]
    VMOVAPS(wg[4], wg[3])
    VFNMADD231PS(wg[4], g[1], var_2)
    VFMADD231PS(wg[3], g[1], var_2)

    # Compute wg[5] and wg[6]
    VMOVAPS(wg[6], wg[5])
    VFNMADD231PS(wg[6], g[1], var_2)
    VFMADD231PS(wg[5], g[1], var_2)

    SWAP.REGISTERS(wg[0], g[0])
    SWAP.REGISTERS(wg[7], g[2])

    if rescale_coefficients:
        VMULPS(wg[1], wg[1], const_minus_2_over_9)
        VMULPS(wg[2], wg[2], const_minus_2_over_9)
        VMULPS(wg[3], wg[3], const_rcp_90)
        VMULPS(wg[4], wg[4], const_rcp_90)
        VMULPS(wg[5], wg[5], const_rcp_180)
        VMULPS(wg[6], wg[6], const_rcp_180)

    return wg


def output_transform(ymm_m):
    assert isinstance(ymm_m, list) and len(ymm_m) == 8 and all(isinstance(ymm, YMMRegister) for ymm in ymm_m)

    ymm_s = [YMMRegister() for _ in range(6)]

    # s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
    # s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
    # s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
    # s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
    # s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
    # s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7

    ymm_m1_add_m2, ymm_m1_sub_m2 = YMMRegister(), YMMRegister()
    VADDPS(ymm_m1_add_m2, ymm_m[1], ymm_m[2])
    VSUBPS(ymm_m1_sub_m2, ymm_m[1], ymm_m[2])

    ymm_m3_add_m4, ymm_m3_sub_m4 = YMMRegister(), YMMRegister()
    VADDPS(ymm_m3_add_m4, ymm_m[3], ymm_m[4])
    VSUBPS(ymm_m3_sub_m4, ymm_m[3], ymm_m[4])

    ymm_m5_add_m6, ymm_m5_sub_m6 = YMMRegister(), YMMRegister()
    VADDPS(ymm_m5_add_m6, ymm_m[5], ymm_m[6])
    VSUBPS(ymm_m5_sub_m6, ymm_m[5], ymm_m[6])


    VADDPS(ymm_s[0], ymm_m[0], ymm_m1_add_m2)
    VADDPS(ymm_s[5], ymm_m[7], ymm_m1_sub_m2)


    ymm_16 = YMMRegister()
    VMOVAPS(ymm_16, Constant.float32x8(16.0))
    VMOVAPS(ymm_s[1], ymm_m1_sub_m2)
    VFMADD231PS(ymm_s[1], ymm_m5_sub_m6, ymm_16)
    VFMADD132PS(ymm_16, ymm_m1_add_m2, ymm_m3_add_m4)
    SWAP.REGISTERS(ymm_s[4], ymm_16)

    ymm_8 = YMMRegister()
    VMOVAPS(ymm_8, Constant.float32x8(8.0))
    VMOVAPS(ymm_s[2], ymm_m1_add_m2)
    VFMADD231PS(ymm_s[2], ymm_m5_add_m6, ymm_8)
    VFMADD132PS(ymm_8, ymm_m1_sub_m2, ymm_m3_sub_m4)
    SWAP.REGISTERS(ymm_s[3], ymm_8)

    ymm_32 = YMMRegister()
    VMOVAPS(ymm_32, Constant.float32x8(32.0))
    VFMADD231PS(ymm_s[0], ymm_m5_add_m6, ymm_32)
    VFMADD231PS(ymm_s[5], ymm_m3_sub_m4, ymm_32)

    ymm_2, ymm_4 = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_2, Constant.float32x8(2.0))
    VADDPS(ymm_s[0], ymm_s[0], ymm_m3_add_m4)
    VMOVAPS(ymm_4, Constant.float32x8(4.0))
    VFMADD231PS(ymm_s[1], ymm_m3_sub_m4, ymm_2)
    VFMADD231PS(ymm_s[4], ymm_m5_add_m6, ymm_2)
    VFMADD231PS(ymm_s[2], ymm_m3_add_m4, ymm_4)
    VFMADD231PS(ymm_s[3], ymm_m5_sub_m6, ymm_4)
    VADDPS(ymm_s[5], ymm_s[5], ymm_m5_sub_m6)

    return ymm_s


def transpose8x8(ymm_rows):
    assert isinstance(ymm_rows, list) and len(ymm_rows) == 8 and all(isinstance(ymm_row, YMMRegister) for ymm_row in ymm_rows)
    # ymm_rows[0] = ( g07, g06, g05, g04, g03, g02, g01, g00 )
    # ymm_rows[1] = ( g17, g16, g15, g14, g13, g12, g11, g10 )
    # ymm_rows[2] = ( g27, g26, g25, g24, g23, g22, g21, g20 )
    # ymm_rows[3] = ( g37, g36, g35, g34, g33, g32, g31, g30 )
    # ymm_rows[4] = ( g47, g46, g45, g44, g43, g42, g41, g40 )
    # ymm_rows[5] = ( g57, g56, g55, g54, g53, g52, g51, g50 )
    # ymm_rows[6] = ( g67, g66, g65, g64, g63, g62, g61, g60 )
    # ymm_rows[7] = ( g77, g76, g75, g74, g73, g72, g71, g70 )

    for ymm_even_row, ymm_odd_row in zip(ymm_rows[0::2], ymm_rows[1::2]):
        ymm_temp = YMMRegister()
        VUNPCKLPS(ymm_temp, ymm_even_row, ymm_odd_row)
        VUNPCKHPS(ymm_odd_row, ymm_even_row, ymm_odd_row)
        SWAP.REGISTERS(ymm_even_row, ymm_temp)

    # ymm_rows[0] = ( g15, g05, g14, g04, g11, g01, g10, g00 )
    # ymm_rows[1] = ( g17, g07, g16, g06, g13, g03, g12, g02 )
    # ymm_rows[2] = ( g35, g25, g34, g24, g31, g21, g30, g20 )
    # ymm_rows[3] = ( g37, g27, g36, g26, g33, g23, g32, g22 )
    # ymm_rows[4] = ( g55, g45, g54, g44, g51, g41, g50, g40 )
    # ymm_rows[5] = ( g57, g47, g56, g46, g53, g43, g52, g42 )
    # ymm_rows[6] = ( g75, g65, g74, g64, g71, g61, g70, g60 )
    # ymm_rows[7] = ( g77, g67, g76, g66, g73, g63, g72, g62 )

    transpose2x2x2x64(ymm_rows[0], ymm_rows[2])
    transpose2x2x2x64(ymm_rows[1], ymm_rows[3])
    transpose2x2x2x64(ymm_rows[4], ymm_rows[6])
    transpose2x2x2x64(ymm_rows[5], ymm_rows[7])

    # ymm_rows[0] = ( g34, g24, g14, g04, g30, g20, g10, g00 )
    # ymm_rows[1] = ( g36, g26, g16, g06, g32, g22, g12, g02 )
    # ymm_rows[2] = ( g35, g25, g15, g05, g31, g21, g11, g01 )
    # ymm_rows[3] = ( g37, g27, g17, g07, g33, g23, g13, g03 )
    # ymm_rows[4] = ( g74, g64, g54, g44, g70, g60, g50, g40 )
    # ymm_rows[5] = ( g76, g66, g56, g46, g72, g62, g52, g42 )
    # ymm_rows[6] = ( g75, g65, g55, g45, g71, g61, g51, g41 )
    # ymm_rows[7] = ( g77, g67, g57, g47, g73, g63, g53, g43 )

    transpose2x2x128(ymm_rows[0], ymm_rows[4])
    transpose2x2x128(ymm_rows[1], ymm_rows[5])
    transpose2x2x128(ymm_rows[2], ymm_rows[6])
    transpose2x2x128(ymm_rows[3], ymm_rows[7])

    SWAP.REGISTERS(ymm_rows[1], ymm_rows[2])
    SWAP.REGISTERS(ymm_rows[5], ymm_rows[6])


def transpose6x8(ymm_rows):
    assert isinstance(ymm_rows, list) and len(ymm_rows) == 6 and all(isinstance(ymm_row, YMMRegister) for ymm_row in ymm_rows)
    # ymm_rows[0] = ( g07, g06, g05, g04, g03, g02, g01, g00 )
    # ymm_rows[1] = ( g17, g16, g15, g14, g13, g12, g11, g10 )
    # ymm_rows[2] = ( g27, g26, g25, g24, g23, g22, g21, g20 )
    # ymm_rows[3] = ( g37, g36, g35, g34, g33, g32, g31, g30 )
    # ymm_rows[4] = ( g47, g46, g45, g44, g43, g42, g41, g40 )
    # ymm_rows[5] = ( g57, g56, g55, g54, g53, g52, g51, g50 )

    for ymm_even_row, ymm_odd_row in zip(ymm_rows[0::2], ymm_rows[1::2]):
        ymm_temp = YMMRegister()
        VUNPCKLPS(ymm_temp, ymm_even_row, ymm_odd_row)
        VUNPCKHPS(ymm_odd_row, ymm_even_row, ymm_odd_row)
        SWAP.REGISTERS(ymm_even_row, ymm_temp)

    # ymm_rows[0] = ( g15, g05, g14, g04, g11, g01, g10, g00 )
    # ymm_rows[1] = ( g17, g07, g16, g06, g13, g03, g12, g02 )
    # ymm_rows[2] = ( g35, g25, g34, g24, g31, g21, g30, g20 )
    # ymm_rows[3] = ( g37, g27, g36, g26, g33, g23, g32, g22 )
    # ymm_rows[4] = ( g55, g45, g54, g44, g51, g41, g50, g40 )
    # ymm_rows[5] = ( g57, g47, g56, g46, g53, g43, g52, g42 )

    ymm_zero_rows = [YMMRegister(), YMMRegister()]
    for ymm_zero in ymm_zero_rows:
        VXORPS(ymm_zero, ymm_zero, ymm_zero)
    ymm_rows += ymm_zero_rows

    # ymm_rows[6] = ( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 )
    # ymm_rows[7] = ( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 )

    transpose2x2x2x64(ymm_rows[0], ymm_rows[2])
    transpose2x2x2x64(ymm_rows[1], ymm_rows[3])
    transpose2x2x2x64(ymm_rows[4], ymm_rows[6])
    transpose2x2x2x64(ymm_rows[5], ymm_rows[7])

    # ymm_rows[0] = ( g34, g24, g14, g04, g30, g20, g10, g00 )
    # ymm_rows[1] = ( g36, g26, g16, g06, g32, g22, g12, g02 )
    # ymm_rows[2] = ( g35, g25, g15, g05, g31, g21, g11, g01 )
    # ymm_rows[3] = ( g37, g27, g17, g07, g33, g23, g13, g03 )
    # ymm_rows[4] = ( 0.0, 0.0, g54, g44, 0.0, 0.0, g50, g40 )
    # ymm_rows[5] = ( 0.0, 0.0, g56, g46, 0.0, 0.0, g52, g42 )
    # ymm_rows[6] = ( 0.0, 0.0, g55, g45, 0.0, 0.0, g51, g41 )
    # ymm_rows[7] = ( 0.0, 0.0, g57, g47, 0.0, 0.0, g53, g43 )

    transpose2x2x128(ymm_rows[0], ymm_rows[4])
    transpose2x2x128(ymm_rows[1], ymm_rows[5])
    transpose2x2x128(ymm_rows[2], ymm_rows[6])
    transpose2x2x128(ymm_rows[3], ymm_rows[7])

    SWAP.REGISTERS(ymm_rows[1], ymm_rows[2])
    SWAP.REGISTERS(ymm_rows[5], ymm_rows[6])

    return ymm_rows


def transpose8x3(xmm_rows):
    assert isinstance(xmm_rows, list) and len(xmm_rows) == 8 and all(isinstance(xmm_row, XMMRegister) for xmm_row in xmm_rows)
    # xmm_rows[0] = ( 0.0, g02, g01, g00 )
    # xmm_rows[1] = ( 0.0, g12, g11, g10 )
    # xmm_rows[2] = ( 0.0, g22, g21, g20 )
    # xmm_rows[3] = ( 0.0, g32, g31, g30 )
    # xmm_rows[4] = ( 0.0, g42, g41, g40 )
    # xmm_rows[5] = ( 0.0, g52, g51, g50 )
    # xmm_rows[6] = ( 0.0, g62, g61, g60 )
    # xmm_rows[7] = ( 0.0, g72, g71, g70 )

    ymm_rows = [YMMRegister() for _ in range(4)]

    VINSERTF128(ymm_rows[0], xmm_rows[0].as_ymm, xmm_rows[4], 1)
    VINSERTF128(ymm_rows[1], xmm_rows[1].as_ymm, xmm_rows[5], 1)
    VINSERTF128(ymm_rows[2], xmm_rows[2].as_ymm, xmm_rows[6], 1)
    VINSERTF128(ymm_rows[3], xmm_rows[3].as_ymm, xmm_rows[7], 1)

    # ymm_rows[0] = ( 0.0, g42, g41, g40, 0.0, g02, g01, g00 )
    # ymm_rows[1] = ( 0.0, g52, g51, g50, 0.0, g12, g11, g10 )
    # ymm_rows[2] = ( 0.0, g62, g61, g60, 0.0, g22, g21, g20 )
    # ymm_rows[3] = ( 0.0, g72, g71, g70, 0.0, g32, g31, g30 )

    ymm_new_rows = [YMMRegister() for _ in range(4)]
    VUNPCKLPS(ymm_new_rows[0], ymm_rows[0], ymm_rows[1])
    VUNPCKHPS(ymm_new_rows[1], ymm_rows[0], ymm_rows[1])
    VUNPCKLPS(ymm_new_rows[2], ymm_rows[2], ymm_rows[3])
    VUNPCKHPS(ymm_new_rows[3], ymm_rows[2], ymm_rows[3])
    for ymm_row, ymm_new_row in zip(ymm_rows, ymm_new_rows):
        SWAP.REGISTERS(ymm_row, ymm_new_row)

    # ymm_rows[0] = ( g51, g41, g50, g40, g11, g01, g10, g00 )
    # ymm_rows[1] = ( 0.0, 0.0, g52, g42, 0.0, 0.0, g12, g02 )
    # ymm_rows[2] = ( g71, g61, g70, g60, g31, g21, g30, g20 )
    # ymm_rows[3] = ( 0.0, 0.0, g72, g62, 0.0, 0.0, g32, g22 )

    # ymm_rows[0] = ( g70, g60, g50, g40, g30, g20, g10, g00 )
    # ymm_rows[2] = ( g71, g61, g51, g41, g31, g21, g11, g01 )
    transpose2x2x2x64(ymm_rows[0], ymm_rows[2])
    # ymm_rows[1] = ( g72, g62, g52, g42, g32, g22, g12, g02 )
    VUNPCKLPD(ymm_rows[1], ymm_rows[1], ymm_rows[3])
    SWAP.REGISTERS(ymm_rows[1], ymm_rows[2])

    return ymm_rows[0:3]


if __name__ == "__main__":
    import numpy
    numpy.set_printoptions(linewidth=120)
    import ctypes

    arg_i = Argument(ptr(const_float_))
    arg_o = Argument(ptr(float_))
    with Function("transpose8x3", (arg_i, arg_o)) as transpose8x3_asm:
        reg_i = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_i, arg_i)

        reg_o = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_o, arg_o)

        xmm_load_mask = XMMRegister()
        VMOVAPS(xmm_load_mask, Constant.float32x4(-0.0, -0.0, -0.0, +0.0))

        xmm_data = [XMMRegister() for i in range(8)]
        for i, xmm in enumerate(xmm_data):
            VMASKMOVPS(xmm, xmm_load_mask, [reg_i + i * 12])


        ymm_data = transpose8x3(xmm_data)

        for i, ymm in enumerate(ymm_data):
            VMOVUPS([reg_o + i * 32], ymm)

        RETURN()

    with Function("transpose8x8", (arg_i, arg_o)) as transpose8x8_asm:
        reg_i = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_i, arg_i)

        reg_o = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_o, arg_o)

        ymm_data = [YMMRegister() for i in range(8)]
        for i, ymm in enumerate(ymm_data):
            VMOVUPS(ymm, [reg_i + i * 32])

        transpose8x8(ymm_data)

        for i, ymm in enumerate(ymm_data):
            VMOVUPS([reg_o + i * 32], ymm)

        RETURN()

    transpose8x3_fn = transpose8x3_asm.finalize(abi.detect()).encode().load()
    transpose8x8_fn = transpose8x8_asm.finalize(abi.detect()).encode().load()

    i = numpy.random.random(8 * 3).astype(numpy.float32)
    o = numpy.empty(8 * 3, numpy.float32)

    transpose8x3_fn(
        i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    # print(i.reshape([8, 3]))
    # print(o.reshape([3, 8]).T)

    i = numpy.random.random(8 * 8).astype(numpy.float32)
    o = numpy.empty(8 * 8, numpy.float32)

    transpose8x8_fn(
        i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    print(i.reshape([8, 8]))
    print(o.reshape([8, 8]).T)
