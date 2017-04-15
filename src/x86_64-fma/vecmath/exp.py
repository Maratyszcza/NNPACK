from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *

log2e = float.fromhex("+0x1.715476p+3")
magic_bias = float.fromhex("+0x1.800000p+23")
zero_cutoff = float.fromhex("-0x1.9FE368p+6")
inf_cutoff = float.fromhex("+0x1.62E42Ep+6")
minus_ln2_hi = float.fromhex("-0x1.62E430p-4")
minus_ln2_lo = float.fromhex("+0x1.05C610p-32")
plus_inf = float("inf")

c2 = float.fromhex("0x1.00088Ap-1")
c3 = float.fromhex("0x1.555A86p-3")
t0 = float.fromhex("0x1.000000p+0")
t1 = float.fromhex("0x1.172B84p+0")
t2 = float.fromhex("0x1.306FE0p+0")
t3 = float.fromhex("0x1.4BFDAEp+0")
t4 = float.fromhex("0x1.6A09E6p+0")
t5 = float.fromhex("0x1.8ACE54p+0")
t6 = float.fromhex("0x1.AE89FAp+0")
t7 = float.fromhex("0x1.D5818Ep+0")

min_exponent = (-126 << 23) & 0xFFFFFFFF
max_exponent = 127 << 23
default_exponent = 0x3F800000
mantissa_mask = 0x007FFFF8


def simd_exp(ymm_xs):
    assert isinstance(ymm_xs, list) and all(isinstance(ymm_x, YMMRegister) for ymm_x in ymm_xs)

    ymm_magic_bias = YMMRegister()
    VMOVAPS(ymm_magic_bias, Constant.float32x8(magic_bias))

    ymm_ys = [YMMRegister() for _ in ymm_xs]
    var_e2s = [LocalVariable(YMMRegister.size) for _ in ymm_xs]
    if len(ymm_xs) > 1:
        const_log2e, const_mantissa_mask, const_lut = None, None, None
        const_minus_ln2_hi, const_minus_ln2_lo = None, None
    else:
        const_lut = Constant.float32x8(t0, t1, t2, t3, t4, t5, t6, t7)
        const_log2e = Constant.float32x8(log2e)
        const_mantissa_mask = Constant.uint32x8(mantissa_mask)
        const_minus_ln2_hi = Constant.float32x8(minus_ln2_hi)
        const_minus_ln2_lo = Constant.float32x8(minus_ln2_lo)
    for ymm_y, ymm_x, var_e2 in zip(ymm_ys, ymm_xs, var_e2s):
        ymm_t = YMMRegister()
        VMOVAPS(ymm_t, ymm_x)
        if const_log2e is None:
            const_log2e = YMMRegister()
            VMOVAPS(const_log2e, Constant.float32x8(log2e))
        VFMADD132PS(ymm_t, ymm_magic_bias, const_log2e)

        if const_mantissa_mask is None:
            const_mantissa_mask = YMMRegister()
            VMOVDQA(const_mantissa_mask, Constant.uint32x8(mantissa_mask))
        ymm_e2 = YMMRegister()
        VPAND(ymm_e2, ymm_t, const_mantissa_mask)
        VPSLLD(ymm_e2, ymm_e2, 20)
        VMOVDQA(var_e2, ymm_e2)

        if const_lut is None:
            const_lut = YMMRegister()
            VMOVAPS(const_lut, Constant.float32x8(t0, t1, t2, t3, t4, t5, t6, t7))
        VPERMPS(ymm_y, ymm_t, const_lut)
        VSUBPS(ymm_t, ymm_t, ymm_magic_bias)

        # x = fma(t, minus_ln2_lo, fma(t, minus_ln2_hi, x))
        # x := t * minus_ln2_hi + x
        # x := t * minus_ln2_lo + rx
        if const_minus_ln2_hi is None:
            const_minus_ln2_hi = YMMRegister()
            VMOVAPS(const_minus_ln2_hi, Constant.float32x8(minus_ln2_hi))
        VFMADD231PS(ymm_x, ymm_t, const_minus_ln2_hi)
        if const_minus_ln2_lo is None:
            const_minus_ln2_lo = YMMRegister()
            VMOVAPS(const_minus_ln2_lo, Constant.float32x8(minus_ln2_lo))
        VFMADD231PS(ymm_x, ymm_t, const_minus_ln2_lo)

    if len(ymm_xs) > 1:
        const_c3 = YMMRegister()
        VMOVAPS(const_c3, Constant.float32x8(c3))
    else:
        const_c3 = Constant.float32x8(c3)
    for ymm_x, ymm_y in zip(ymm_xs, ymm_ys):
        # rf = fma(rx, rx * fma(rx, c3, c2), rx)
        # rf := rx * c3 + c2
        # rf := rx * rf
        # rf := rx * rf + rx
        ymm_rf = YMMRegister()
        VMOVAPS(ymm_rf, Constant.float32x8(c2))
        VFMADD231PS(ymm_rf, ymm_x, const_c3)
        VMULPS(ymm_rf, ymm_rf, ymm_x)
        VFMADD213PS(ymm_rf, ymm_x, ymm_x)

        # y = fma(y, rf, y)
        VFMADD231PS(ymm_y, ymm_y, ymm_rf)

    if len(ymm_xs) > 1:
        const_min_exponent, const_max_exponent = YMMRegister(), YMMRegister()
        VMOVDQA(const_min_exponent, Constant.uint32x8(min_exponent))
        VMOVDQA(const_max_exponent, Constant.uint32x8(max_exponent))
    else:
        const_min_exponent = Constant.uint32x8(min_exponent)
        const_max_exponent = Constant.uint32x8(max_exponent)
    ymm_default_exponent = YMMRegister()
    VMOVDQA(ymm_default_exponent, Constant.uint32x8(default_exponent))
    for ymm_x, ymm_y, var_e2 in zip(ymm_xs, ymm_ys, var_e2s):
        ymm_e1, ymm_e2 = YMMRegister(), YMMRegister()
        VMOVDQA(ymm_e2, var_e2)
        VPMAXSD(ymm_e1, ymm_e2, const_min_exponent)
        VPMINSD(ymm_e1, ymm_e1, const_max_exponent)

        VPSUBD(ymm_e2, ymm_e2, ymm_e1)

        VPADDD(ymm_e1, ymm_e1, ymm_default_exponent)
        VPADDD(ymm_e2, ymm_e2, ymm_default_exponent)

        VMULPS(ymm_y, ymm_y, ymm_e1)
        VMULPS(ymm_y, ymm_y, ymm_e2)

    return ymm_ys
