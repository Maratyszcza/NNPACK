from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *

from common import interleave


def ifft8_within_rows_preprocess(ymm_wr, ymm_wi):
    assert isinstance(ymm_wr, YMMRegister)
    assert isinstance(ymm_wi, YMMRegister)

    # w0.re, w1.re, w2.re, w3.re, w4.re, w5.re, w6.re, w7.re = \
    #    = f0.re, f2.re - f3.im, f4.re - f5.im, f6.re - f7.im, f0.im, f6.re + f7.im, f4.re + f5.im, f2.re + f3.im
    # w0.im, w1.im, w2.im, w3.im, w4.im, w5.im, w6.im, w7.im = \
    #    = f1.re, f3.re + f2.im, f5.re + f4.im, f7.re + f6.im, f1.im, f7.re - f6.im, f5.re - f4.im, f3.re - f2.im

    ymm_shuffle_02461642 = YMMRegister()
    VMOVDQA(ymm_shuffle_02461642, Constant.uint32x8(0, 2, 4, 6, 1, 6, 4, 2))
    ymm_shuffle_13570753 = YMMRegister()
    VMOVDQA(ymm_shuffle_13570753, Constant.uint32x8(1, 3, 5, 7, 0, 7, 5, 3))

    ymm_wr_02461642 = YMMRegister()
    VPERMPS(ymm_wr_02461642, ymm_shuffle_02461642, ymm_wr)
    ymm_wr_13570753 = YMMRegister()
    VPERMPS(ymm_wr_13570753, ymm_shuffle_13570753, ymm_wr)

    ymm_wi_02461642 = YMMRegister()
    VPERMPS(ymm_wi_02461642, ymm_shuffle_02461642, ymm_wi)
    ymm_wi_13570753 = YMMRegister()
    VPERMPS(ymm_wi_13570753, ymm_shuffle_13570753, ymm_wi)

    # wr02461642 = f0.re, f2.re - f3.im, f4.re - f5.im, f6.re - f7.im, -, f6.re + f7.im, f4.re + f5.im, f2.re + f3.im
    VFMADD231PS(ymm_wr_02461642, ymm_wi_13570753, Constant.float32x8(0.0, -1.0, -1.0, -1.0, 0.0, +1.0, +1.0, +1.0))
    # wi13570753 = f1.re, f3.re + f2.im, f5.re + f4.im, f7.re + f6.im, -, f7.re - f6.im, f5.re - f4.im, f3.re - f2.im
    VFMADD231PS(ymm_wr_13570753, ymm_wi_02461642, Constant.float32x8(0.0, +1.0, +1.0, +1.0, 0.0, -1.0, -1.0, -1.0))

    VBLENDPS(ymm_wr_02461642, ymm_wr_02461642, ymm_wi_13570753, 0b00010000)
    VBLENDPS(ymm_wr_13570753, ymm_wr_13570753, ymm_wi_02461642, 0b00010000)

    SWAP.REGISTERS(ymm_wr_02461642, ymm_wr)
    SWAP.REGISTERS(ymm_wr_13570753, ymm_wi)


def ifft16_within_rows_preprocess(ymm_wr, ymm_wi, bit_reversal=False):
    assert isinstance(ymm_wr, (list, tuple)) and len(ymm_wr) == 2 and all(isinstance(reg, YMMRegister) for reg in ymm_wr)
    assert isinstance(ymm_wi, (list, tuple)) and len(ymm_wi) == 2 and all(isinstance(reg, YMMRegister) for reg in ymm_wi)

    # w0.re, w1.re,  w2.re,  w3.re,  w4.re,  w5.re,  w6.re,  w7.re = \
    #    = f0.re, f2.re - f3.im, f4.re - f5.im, f6.re - f7.im, f8.re - f9.im, f10.re - f11.im, f12.re - f13.im, f14.re - f15.im
    # w8.re, w9.re, w10.re, w11.re, w12.re, w13.re, w14.re, w15.re = \
    #    = f0.im, f14.re + f15.im, f12.re + f13.im, f10.re + f11.im, f8.re + f9.im, f6.re + f7.im, f4.re + f5.im, f2.re + f3.im
    #
    # w0.im, w1.im,  w2.im,  w3.im,  w4.im,  w5.im,  w6.im,  w7.im = \
    #    = f1.re, f3.re + f2.im, f5.re + f4.im, f7.re + f6.im, r9.re + f8.im, f11.re + f10.im, f13.re + f12.im, f15.re + f14.im
    # w8.im, w9.im, w10.im, w11.im, w12.im, w13.im, w14.im, w15.im = \
    #    = f1.im, f15.re - f14.im, f13.re - f12.im, f11.re - f10.im, f9.re - f8.im, f7.re - f6.im, f5.re - f4.im, f3.re - f2.im

    # Step 1.A:
    #   w0.re, w1.re, w2.re, w3.re, -, w13.re, w14.re, w15.re = \
    #      = f0.re, f2.re - f3.im, f4.re - f5.im, f6.re - f7.im, -, f6.re + f7.im, f4.re + f5.im, f2.re + f3.im
    #   w0.im, w1.im, w2.im, w3.im, -, w13.im, w14.im, w15.im = \
    #      = f1.re, f3.re + f2.im, f5.re + f4.im, f7.re + f6.im, -, f7.re - f6.im, f5.re - f4.im, f3.re - f2.im

    ymm_shuffle_02461642 = YMMRegister()
    VMOVDQA(ymm_shuffle_02461642, Constant.uint32x8(0, 2, 4, 6, 1, 6, 4, 2))
    ymm_shuffle_13570753 = YMMRegister()
    VMOVDQA(ymm_shuffle_13570753, Constant.uint32x8(1, 3, 5, 7, 0, 7, 5, 3))

    ymm_fr_02461642, ymm_fi_13570753 = YMMRegister(), YMMRegister()
    VPERMPS(ymm_fr_02461642, ymm_shuffle_02461642, ymm_wr[0])
    VPERMPS(ymm_fi_13570753, ymm_shuffle_13570753, ymm_wi[0])
    VFMADD231PS(ymm_fr_02461642, ymm_fi_13570753, Constant.float32x8(0.0, -1.0, -1.0, -1.0, 0.0, +1.0, +1.0, +1.0))

    ymm_fr_13570753, ymm_fi_02461642 = YMMRegister(), YMMRegister()
    VPERMPS(ymm_fr_13570753, ymm_shuffle_13570753, ymm_wr[0])
    VPERMPS(ymm_fi_02461642, ymm_shuffle_02461642, ymm_wi[0])
    VFMADD231PS(ymm_fr_13570753, ymm_fi_02461642, Constant.float32x8(0.0, +1.0, +1.0, +1.0, 0.0, -1.0, -1.0, -1.0))

    ymm_wr_0123xDEF, ymm_wi_0123xDEF = ymm_fr_02461642, ymm_fr_13570753

    # Step 1.B:
    #   -, w9.re, w10.re, w11.re, w4.re, w5.re, w6.re, w7.re = \
    #      = -, f14.re + f15.im, f12.re + f13.im, f10.re + f11.im, r8.re - r9.im, r10.re - r11.im, r12.re - r13.im, r14.re - f15.im
    #   -, w9.im, w10.im, w11.im, w4.im, w5.im, w6.im, w7.im = \
    #      = -, f15.re - f14.im, f13.re - f12.im, f11.re - f10.im, r9.re + f8.im, f11.re + f10.im, f13.re + f12.im, f15.re + f14.im

    ymm_shuffle_06420246 = YMMRegister()
    VMOVDQA(ymm_shuffle_06420246, Constant.uint32x8(0, 6, 4, 2, 0, 2, 4, 6))
    ymm_shuffle_17531357 = YMMRegister()
    VMOVDQA(ymm_shuffle_17531357, Constant.uint32x8(1, 7, 5, 3, 1, 3, 5, 7))

    ymm_wr_xxxxCxxx, ymm_wi_xxxxCxxx = YMMRegister(), YMMRegister()
    ymm_wr_0123CDEF, ymm_wi_0123CDEF = YMMRegister(), YMMRegister()

    ymm_fr_8ECA8ACE, ymm_fi_9FDB9BDF = YMMRegister(), YMMRegister()
    VPERMPS(ymm_fr_8ECA8ACE, ymm_shuffle_06420246, ymm_wr[1])
    VPERMPS(ymm_fi_9FDB9BDF, ymm_shuffle_17531357, ymm_wi[1])
    VADDPS(ymm_wr_xxxxCxxx, ymm_fr_8ECA8ACE, ymm_fi_9FDB9BDF)
    VFMADD231PS(ymm_fr_8ECA8ACE, ymm_fi_9FDB9BDF, Constant.float32x8(0.0, +1.0, +1.0, +1.0, -1.0, -1.0, -1.0, -1.0))
    VBLENDPS(ymm_wr_0123CDEF, ymm_wr_0123xDEF, ymm_wr_xxxxCxxx, 0b00010000)

    ymm_fr_9FDB9BDF, ymm_fi_8ECA8ACE = YMMRegister(), YMMRegister()
    VPERMPS(ymm_fr_9FDB9BDF, ymm_shuffle_17531357, ymm_wr[1])
    VPERMPS(ymm_fi_8ECA8ACE, ymm_shuffle_06420246, ymm_wi[1])
    VSUBPS(ymm_wi_xxxxCxxx, ymm_fr_9FDB9BDF, ymm_fi_8ECA8ACE)
    VFMADD231PS(ymm_fr_9FDB9BDF, ymm_fi_8ECA8ACE, Constant.float32x8(0.0, -1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0))
    VBLENDPS(ymm_wi_0123CDEF, ymm_wi_0123xDEF, ymm_wi_xxxxCxxx, 0b00010000)

    ymm_wr_x9AB4567, ymm_wi_x9AB4567 = ymm_fr_8ECA8ACE, ymm_fr_9FDB9BDF
    ymm_wr_89AB4567, ymm_wi_89AB4567 = YMMRegister(), YMMRegister()
    VBLENDPS(ymm_wr_89AB4567, ymm_wr_x9AB4567, ymm_fi_02461642, 0b00000001)
    VBLENDPS(ymm_wi_89AB4567, ymm_wi_x9AB4567, ymm_fi_13570753, 0b00000001)

    ymm_wr_01234567, ymm_wr_89ABCDEF = YMMRegister(), YMMRegister()
    VBLENDPS(ymm_wr_01234567, ymm_wr_0123CDEF, ymm_wr_89AB4567, 0xF0)
    VBLENDPS(ymm_wr_89ABCDEF, ymm_wr_0123CDEF, ymm_wr_89AB4567, 0x0F)

    ymm_wi_01234567, ymm_wi_89ABCDEF = YMMRegister(), YMMRegister()
    VBLENDPS(ymm_wi_01234567, ymm_wi_0123CDEF, ymm_wi_89AB4567, 0xF0)
    VBLENDPS(ymm_wi_89ABCDEF, ymm_wi_0123CDEF, ymm_wi_89AB4567, 0x0F)

    SWAP.REGISTERS(ymm_wr[0], ymm_wr_01234567)
    SWAP.REGISTERS(ymm_wi[0], ymm_wi_01234567)
    SWAP.REGISTERS(ymm_wr[1], ymm_wr_89ABCDEF)
    SWAP.REGISTERS(ymm_wi[1], ymm_wi_89ABCDEF)

    if bit_reversal:
        # Bit reversal
        # w[0] = x0 x8 x4 x12 x2 x10 x6 x14
        # w[1] = x1 x9 x5 x13 x3 x11 x7 x15
        ymm_bit_reversal_mask = YMMRegister()
        VMOVDQA(ymm_bit_reversal_mask, Constant.uint32x8(0, 2, 4, 6, 1, 3, 5, 7))
        for ymm in interleave(ymm_wr, ymm_wi):
            VPERMPS(ymm, ymm_bit_reversal_mask, ymm)
