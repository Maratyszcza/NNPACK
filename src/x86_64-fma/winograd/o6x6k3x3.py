from peachpy import *
from peachpy.x86_64 import *


from common import _MM_SHUFFLE
from common import transpose2x2x128, transpose2x2x2x64


def input_transform(ymm_d):
    assert isinstance(ymm_d, list) and len(ymm_d) == 8 and all(isinstance(ymm, YMMRegister) for ymm in ymm_d)

    ymm_wd = [YMMRegister() for _ in range(8)]

    # wd0 = d6 - 36 * d0 + 49 * d2 - 14 * d4
    # wd1 = (d6 + 36 * d2 - 13 * d4) + (d5 + 36 * d1 - 13 * d3)
    # wd2 = (d6 + 36 * d2 - 13 * d4) - (d5 + 36 * d1 - 13 * d3)
    # wd3 = (d6 + 9 * d2 - 10 * d4) + 2 * (d5 + 9 * d1 - 10 * d3)
    # wd4 = (d6 + 9 * d2 - 10 * d4) - 2 * (d5 + 9 * d1 - 10 * d3)
    # wd5 = (d6 + 4 * d2 - 5 * d4) + 3 * (d5 + 4 * d1 - 5 * d3)
    # wd6 = (d6 + 4 * d2 - 5 * d4) - 3 * (d5 + 4 * d1 - 5 * d3)
    # wd7 = d7 - 36 * d1 + 49 * d3 - 14 * d5

    ymm_36 = YMMRegister()
    VMOVAPS(ymm_36, Constant.float32x8(36.0))

    ymm_wd[0] = ymm_d[0]
    VFNMADD132PS(ymm_wd[0], ymm_d[6], ymm_36)
    ymm_wd[7] = ymm_d[7]
    VFNMADD231PS(ymm_wd[7], ymm_d[1], ymm_36)

    ymm_4, ymm_9 = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_9, Constant.float32x8(9.0))
    VMOVAPS(ymm_4, Constant.float32x8(4.0))

    VMOVAPS(ymm_wd[1], ymm_d[5])
    VFMADD231PS(ymm_wd[1], ymm_d[1], ymm_36)
    VMOVAPS(ymm_wd[2], ymm_d[6])
    VFMADD231PS(ymm_wd[2], ymm_d[2], ymm_36)

    VMOVAPS(ymm_wd[3], ymm_d[5])
    VFMADD231PS(ymm_wd[3], ymm_d[1], ymm_9)
    VMOVAPS(ymm_wd[4], ymm_d[6])
    VFMADD231PS(ymm_wd[4], ymm_d[2], ymm_9)

    ymm_49, ymm_13 = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_49, Constant.float32x8(49.0))
    VMOVAPS(ymm_13, Constant.float32x8(13.0))

    VMOVAPS(ymm_wd[5], ymm_d[5])
    VFMADD231PS(ymm_wd[5], ymm_d[1], ymm_4)
    VMOVAPS(ymm_wd[6], ymm_d[6])
    VFMADD231PS(ymm_wd[6], ymm_d[2], ymm_4)

    ymm_10, ymm_5 = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_10, Constant.float32x8(10.0))
    VMOVAPS(ymm_5, Constant.float32x8(5.0))

    VFMADD231PS(ymm_wd[0], ymm_d[2], ymm_49)
    VFMADD231PS(ymm_wd[7], ymm_d[3], ymm_49)
    VFNMADD231PS(ymm_wd[1], ymm_d[3], ymm_13)
    VFNMADD231PS(ymm_wd[2], ymm_d[4], ymm_13)

    ymm_14 = YMMRegister()
    VMOVAPS(ymm_14, Constant.float32x8(14.0))

    VFNMADD231PS(ymm_wd[3], ymm_d[3], ymm_10)
    VFNMADD231PS(ymm_wd[4], ymm_d[4], ymm_10)
    VFNMADD231PS(ymm_wd[5], ymm_d[3], ymm_5)
    VFNMADD231PS(ymm_wd[6], ymm_d[4], ymm_5)

    VFNMADD231PS(ymm_wd[0], ymm_d[4], ymm_14)
    VFNMADD231PS(ymm_wd[7], ymm_d[5], ymm_14)

    ymm_2 = YMMRegister()
    VMOVAPS(ymm_2, Constant.float32x8(2.0))

    ymm_new_wd1 = YMMRegister()
    VADDPS(ymm_new_wd1, ymm_wd[2], ymm_wd[1])
    VSUBPS(ymm_wd[2], ymm_wd[2], ymm_wd[1])
    SWAP.REGISTERS(ymm_wd[1], ymm_new_wd1)

    ymm_3 = YMMRegister()
    VMOVAPS(ymm_3, Constant.float32x8(3.0))

    ymm_new_wd3 = YMMRegister()
    VMOVAPS(ymm_new_wd3, ymm_wd[3])
    VFMADD132PS(ymm_new_wd3, ymm_wd[4], ymm_2)
    VFNMADD231PS(ymm_wd[4], ymm_wd[3], ymm_2)
    SWAP.REGISTERS(ymm_wd[3], ymm_new_wd3)

    ymm_new_wd5 = YMMRegister()
    VMOVAPS(ymm_new_wd5, ymm_wd[5])
    VFMADD132PS(ymm_new_wd5, ymm_wd[6], ymm_3)
    VFNMADD231PS(ymm_wd[6], ymm_wd[5], ymm_3)
    SWAP.REGISTERS(ymm_wd[5], ymm_new_wd5)

    return ymm_wd


def kernel_transform(g, rescale_coefficients=True):
    assert isinstance(g, list) and len(g) == 3 and \
        (all(isinstance(reg, XMMRegister) for reg in g) or all(isinstance(reg, YMMRegister) for reg in g))

    rcp_minus_36  = float.fromhex("-0x1.C71C72p-6")
    rcp_48        = float.fromhex( "0x1.555556p-6")
    rcp_minus_120 = float.fromhex("-0x1.111112p-7")
    rcp_720       = float.fromhex( "0x1.6C16C2p-10")

    if isinstance(g[0], XMMRegister):
        wg = [XMMRegister() for _ in range(8)]
        const_2 = Constant.float32x4(2.0)
        const_3 = Constant.float32x4(3.0)
        const_4 = Constant.float32x4(4.0)
        const_9 = Constant.float32x4(9.0)
        const_rcp_minus_36 = Constant.float32x4(rcp_minus_36)
        const_rcp_48 = Constant.float32x4(rcp_48)
        const_rcp_minus_120 = Constant.float32x4(rcp_minus_120)
        const_rcp_720 = Constant.float32x4(rcp_720)
    else:
        wg = [YMMRegister() for _ in range(8)]
        const_2 = Constant.float32x8(2.0)
        const_3 = Constant.float32x8(3.0)
        const_4 = Constant.float32x8(4.0)
        const_9 = Constant.float32x8(9.0)
        const_rcp_minus_36 = Constant.float32x8(rcp_minus_36)
        const_rcp_48 = Constant.float32x8(rcp_48)
        const_rcp_minus_120 = Constant.float32x8(rcp_minus_120)
        const_rcp_720 = Constant.float32x8(rcp_720)

    # wg[0] = g0 * (-1. / 36)
    # wg[1] = ((g0 + g2) + g1) * (1.0 / 48)
    # wg[2] = ((g0 + g2) - g1) * (1.0 / 48)
    # wg[3] = ((g0 + 4 * g2) + 2 * g1) * (-1. / 120)
    # wg[4] = ((g0 + 4 * g2) - 2 * g1) * (-1. / 120)
    # wg[5] = ((g0 + 9 * g2) + 3 * g1) * (1. / 720)
    # wg[6] = ((g0 + 9 * g2) - 3 * g1) * (1. / 720)
    # wg[7] = g2

    VADDPS(wg[2], g[0], g[2])
    VMOVAPS(wg[4], g[0])
    VFMADD231PS(wg[4], g[2], const_4)
    VMOVAPS(wg[6], g[0])
    VFMADD231PS(wg[6], g[2], const_9)

    VADDPS(wg[1], wg[2], g[1])
    VSUBPS(wg[2], wg[2], g[1])
    VMOVAPS(wg[3], wg[4])
    VFMADD231PS(wg[3], g[1], const_2)
    VFNMADD231PS(wg[4], g[1], const_2)
    VMOVAPS(wg[5], wg[6])
    VFMADD231PS(wg[5], g[1], const_3)
    VFNMADD231PS(wg[6], g[1], const_3)

    wg[0], wg[7] = g[0], g[2]

    if rescale_coefficients:
        VMULPS(wg[0], wg[0], const_rcp_minus_36)
        VMULPS(wg[1], wg[1], const_rcp_48)
        VMULPS(wg[2], wg[2], const_rcp_48)
        VMULPS(wg[3], wg[3], const_rcp_minus_120)
        VMULPS(wg[4], wg[4], const_rcp_minus_120)
        VMULPS(wg[5], wg[5], const_rcp_720)
        VMULPS(wg[6], wg[6], const_rcp_720)

    return wg


def output_transform(ymm_m):
    assert isinstance(ymm_m, list) and len(ymm_m) == 8 and all(isinstance(ymm, YMMRegister) for ymm in ymm_m)

    ymm_s = [YMMRegister() for _ in range(6)]

    # s0 = m0 + (m1 + m2) + (m3 + m4) + (m5 + m6)
    # s1 = (m1 - m2) + 2 * (m3 - m4) + 3 * (m5 - m6)
    # s2 = (m1 + m2) + 4 * (m3 + m4) + 9 * (m5 + m6)
    # s3 = (m1 - m2) + 8 * (m3 - m4) + 27 * (m5 - m6)
    # s4 = (m1 + m2) + 16 * (m3 + m4) + 81 * (m5 + m6)
    # s5 = m7 + (m1 - m2) + 32 * (m3 - m4) + 243 * (m5 - m6)

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

    VMOVAPS(ymm_s[1], ymm_m1_sub_m2)
    VFMADD231PS(ymm_s[1], ymm_m3_sub_m4, Constant.float32x8(2.0))
    VMOVAPS(ymm_s[2], ymm_m1_add_m2)
    VFMADD231PS(ymm_s[2], ymm_m3_add_m4, Constant.float32x8(4.0))

    ymm_s[3], ymm_s[4] = ymm_m1_sub_m2, ymm_m1_add_m2
    VFMADD231PS(ymm_s[3], ymm_m3_sub_m4, Constant.float32x8(8.0))
    VFMADD231PS(ymm_s[4], ymm_m3_add_m4, Constant.float32x8(16.0))

    VADDPS(ymm_s[0], ymm_s[0], ymm_m3_add_m4)
    VFMADD231PS(ymm_s[5], ymm_m3_sub_m4, Constant.float32x8(32.0))

    VFMADD231PS(ymm_s[1], ymm_m5_sub_m6, Constant.float32x8(3.0))
    VFMADD231PS(ymm_s[2], ymm_m5_add_m6, Constant.float32x8(9.0))
    VFMADD231PS(ymm_s[3], ymm_m5_sub_m6, Constant.float32x8(27.0))
    VFMADD231PS(ymm_s[4], ymm_m5_add_m6, Constant.float32x8(81.0))

    VADDPS(ymm_s[0], ymm_s[0], ymm_m5_add_m6)
    VFMADD231PS(ymm_s[5], ymm_m5_sub_m6, Constant.float32x8(243.0))

    return ymm_s


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
    VUNPCKLPD(ymm_rows[0], ymm_rows[0], ymm_rows[2])
    # ymm_rows[2] = ( g71, g61, g51, g41, g31, g21, g11, g01 )
    VUNPCKHPD(ymm_rows[0], ymm_rows[0], ymm_rows[2])
    # ymm_rows[1] = ( g72, g62, g52, g42, g32, g22, g12, g02 )
    VUNPCKLPD(ymm_rows[1], ymm_rows[1], ymm_rows[3])

    return ymm_rows[0:3]


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
