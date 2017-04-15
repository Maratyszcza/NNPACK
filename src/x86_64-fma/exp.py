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

x_arg = Argument(m256, "x")
with Function("_mm256_exp_ps", (x_arg,), m256,
              target=uarch.default + isa.fma3 + isa.avx2):
    ymm_x = YMMRegister()
    LOAD.ARGUMENT(ymm_x, x_arg)

    ymm_magic_bias = YMMRegister()
    VMOVAPS(ymm_magic_bias, Constant.float32x8(magic_bias))

    ymm_t = YMMRegister()
    VMOVAPS(ymm_t, ymm_x)
    VFMADD132PS(ymm_t, ymm_magic_bias, Constant.float32x8(log2e))

    ymm_e1, ymm_e2 = YMMRegister(), YMMRegister()
    VPAND(ymm_e2, ymm_t, Constant.uint32x8(mantissa_mask))
    VPSLLD(ymm_e2, ymm_e2, 20)

    ymm_tf = YMMRegister()
    VPERMPS(ymm_tf, ymm_t, Constant.float32x8(t0, t1, t2, t3, t4, t5, t6, t7))
    VSUBPS(ymm_t, ymm_t, ymm_magic_bias)

    # rx = fma(t, minus_ln2_lo, fma(t, minus_ln2_hi, x))
    # rx := t * minus_ln2_hi + x
    # rx := t * minus_ln2_lo + rx
    ymm_rx = YMMRegister()
    VMOVAPS(ymm_rx, ymm_x)
    VFMADD231PS(ymm_rx, ymm_t, Constant.float32x8(minus_ln2_hi))
    VFMADD231PS(ymm_rx, ymm_t, Constant.float32x8(minus_ln2_lo))

    VPMAXSD(ymm_e1, ymm_e2, Constant.uint32x8(min_exponent))
    VPMINSD(ymm_e1, ymm_e1, Constant.uint32x8(max_exponent))

    ymm_default_exponent = YMMRegister()
    VMOVDQA(ymm_default_exponent, Constant.uint32x8(default_exponent))
    VPSUBD(ymm_e2, ymm_e2, ymm_e1)

    VPADDD(ymm_e1, ymm_e1, ymm_default_exponent)
    VPADDD(ymm_e2, ymm_e2, ymm_default_exponent)

    # rf = fma(rx, rx * fma(rx, c3, c2), rx)
    # rf := rx * c3 + c2
    # rf := rx * rf
    # rf := rx * rf + rx
    ymm_rf = YMMRegister()
    VMOVAPS(ymm_rf, Constant.float32x8(c2))
    VFMADD231PS(ymm_rf, ymm_rx, Constant.float32x8(c3))
    VMULPS(ymm_rf, ymm_rf, ymm_rx)
    VFMADD213PS(ymm_rf, ymm_rx, ymm_rx)

    # f = fma(tf, rf, tf)
    VFMADD231PS(ymm_tf, ymm_tf, ymm_rf)
    ymm_f = ymm_tf

    VMULPS(ymm_f, ymm_f, ymm_e1)
    VMULPS(ymm_f, ymm_f, ymm_e2)

    RETURN(ymm_f)
