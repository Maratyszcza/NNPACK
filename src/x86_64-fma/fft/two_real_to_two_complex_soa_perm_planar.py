from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *


def fft8_within_rows_postprocess(ymm_wr, ymm_wi):
    assert isinstance(ymm_wr, YMMRegister)
    assert isinstance(ymm_wi, YMMRegister)

    ymm_shuffle_44112233 = YMMRegister()
    VMOVDQA(ymm_shuffle_44112233, Constant.uint32x8(4, 4, 1, 1, 2, 2, 3, 3))
    ymm_shuffle_00776655 = YMMRegister()
    VMOVDQA(ymm_shuffle_00776655, Constant.uint32x8(0, 0, 7, 7, 6, 6, 5, 5))

    ymm_wr_44112233 = YMMRegister()
    VPERMPS(ymm_wr_44112233, ymm_shuffle_44112233, ymm_wr)
    ymm_wr_00776655 = YMMRegister()
    VPERMPS(ymm_wr_00776655, ymm_shuffle_00776655, ymm_wr)

    ymm_wi_44112233 = YMMRegister()
    VPERMPS(ymm_wi_44112233, ymm_shuffle_44112233, ymm_wi)
    ymm_wi_00776655 = YMMRegister()
    VPERMPS(ymm_wi_00776655, ymm_shuffle_00776655, ymm_wi)

    # wr44776655 = wr0,   -, wr7 + wr1, wr7 - wr1, wr6 + wr2, wr6 - wr2, wr5 + wr3, wr5 - wr3
    VFMADD231PS(ymm_wr_00776655, ymm_wr_44112233, Constant.float32x8(0.0, 0.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0))
    # wi00112233 =   _, wi4, wi1 - wi7, wi1 + wi7, wi2 - wi6, wi2 + wi6, wi3 - wi5, wi3 + wi5
    VFMADD231PS(ymm_wi_44112233, ymm_wi_00776655, Constant.float32x8(0.0, 0.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0))

    # xhr = wr0,   -, wr1 + wr7, wi1 + wi7, wr2 + wr6, wi2 + wi6, wr3 + wr5, wi3 + wi5
    ymm_xhr = YMMRegister()
    VBLENDPS(ymm_xhr, ymm_wr_00776655, ymm_wi_44112233, 0b10101010)
    # xhI =   -, wi4, wi1 - wi7, wr7 - wr1, wi2 - wi6, wr6 - wr2, wi3 - wi5, wr5 - wr3
    ymm_xhi = YMMRegister()
    VBLENDPS(ymm_xhi, ymm_wr_00776655, ymm_wi_44112233, 0b01010110)

    # xhr = wr0, wi0, wr1 + wr7, wi1 + wi7, wr2 + wr6, wi2 + wi6, wr3 + wr5, wi3 + wi5
    VBLENDPS(ymm_xhr, ymm_xhr, ymm_wi_00776655, 0b00000010)

    # xhI = wr4, wi4, wi1 - wi7, wr7 - wr1, wi2 - wi6, wr6 - wr2, wi3 - wi5, wr5 - wr3
    VBLENDPS(ymm_xhi, ymm_xhi, ymm_wr_44112233, 0b00000001)

    ymm_scale_factor = YMMRegister()
    VMOVAPS(ymm_scale_factor, Constant.float32x8(1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    VMULPS(ymm_xhr, ymm_xhr, ymm_scale_factor)
    VMULPS(ymm_xhi, ymm_xhi, ymm_scale_factor)

    # wRe[0], wIm[0], (wRe[1] + wRe[7]) / 2, (wIm[1] + wIm[7]) / 2,
    #   (wRe[2] + wRe[6]) / 2, (wIm[2] + wIm[6]) / 2, (wRe[3] + wRe[5]) / 2, (wIm[3] + wIm[5]) / 2

    # wRe[4], wIm[4], (wIm[1] - wIm[7]) / 2, (wRe[7] - wRe[1]) / 2,
    #   (wIm[2] - wIm[6]) / 2, (wRe[6] - wRe[2]) / 2, (wIm[3] - wIm[5]) / 2, (wRe[5] - wRe[3]) / 2

    SWAP.REGISTERS(ymm_xhr, ymm_wr)
    SWAP.REGISTERS(ymm_xhi, ymm_wi)


def fft16_within_rows_postprocess(ymm_wr, ymm_wi, bit_reversal=False):
    assert isinstance(ymm_wr, (list, tuple)) and len(ymm_wr) == 2 and all(isinstance(reg, YMMRegister) for reg in ymm_wr)
    assert isinstance(ymm_wi, (list, tuple)) and len(ymm_wi) == 2 and all(isinstance(reg, YMMRegister) for reg in ymm_wi)

    if bit_reversal:
        ymm_shuffle_00112233 = YMMRegister()
        VMOVDQA(ymm_shuffle_00112233, Constant.uint32x8(0, 0, 4, 4, 1, 1, 5, 5))
        ymm_shuffle_44556677 = YMMRegister()
        VMOVDQA(ymm_shuffle_44556677, Constant.uint32x8(2, 2, 6, 6, 3, 3, 7, 7))
        ymm_shuffle_44332211 = YMMRegister()
        VMOVDQA(ymm_shuffle_44332211, Constant.uint32x8(2, 2, 5, 5, 1, 1, 4, 4))
        ymm_shuffle_00776655 = YMMRegister()
        VMOVDQA(ymm_shuffle_00776655, Constant.uint32x8(0, 0, 7, 7, 3, 3, 6, 6))
    else:
        ymm_shuffle_00112233 = YMMRegister()
        VMOVDQA(ymm_shuffle_00112233, Constant.uint32x8(0, 0, 1, 1, 2, 2, 3, 3))
        ymm_shuffle_44556677 = YMMRegister()
        VMOVDQA(ymm_shuffle_44556677, Constant.uint32x8(4, 4, 5, 5, 6, 6, 7, 7))
        ymm_shuffle_44332211 = YMMRegister()
        VMOVDQA(ymm_shuffle_44332211, Constant.uint32x8(4, 4, 3, 3, 2, 2, 1, 1))
        ymm_shuffle_00776655 = YMMRegister()
        VMOVDQA(ymm_shuffle_00776655, Constant.uint32x8(0, 0, 7, 7, 6, 6, 5, 5))

    ymm_wr_00112233, ymm_wr_44556677 = YMMRegister(), YMMRegister()
    VPERMPS(ymm_wr_00112233, ymm_shuffle_00112233, ymm_wr[0])
    VPERMPS(ymm_wr_44556677, ymm_shuffle_44556677, ymm_wr[0])

    ymm_wr_CCBBAA99, ymm_wr_88FFEEDD = YMMRegister(), YMMRegister()
    VPERMPS(ymm_wr_CCBBAA99, ymm_shuffle_44332211, ymm_wr[1])
    VPERMPS(ymm_wr_88FFEEDD, ymm_shuffle_00776655, ymm_wr[1])

    ymm_wi_00112233, ymm_wi_44556677 = YMMRegister(), YMMRegister()
    VPERMPS(ymm_wi_00112233, ymm_shuffle_00112233, ymm_wi[0])
    VPERMPS(ymm_wi_44556677, ymm_shuffle_44556677, ymm_wi[0])

    ymm_wi_CCBBAA99, ymm_wi_88FFEEDD = YMMRegister(), YMMRegister()
    VPERMPS(ymm_wi_CCBBAA99, ymm_shuffle_44332211, ymm_wi[1])
    VPERMPS(ymm_wi_88FFEEDD, ymm_shuffle_00776655, ymm_wi[1])

    # wr88FFEEDD = wr8,   -, wr15 + wr1, wr15 - wr1, wr14 + wr2, wr14 - wr2, wr13 + wr3, wr13 - wr3
    VFMADD231PS(ymm_wr_88FFEEDD, ymm_wr_00112233, Constant.float32x8(0.0, 0.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0))
    # wrCCBBAA99 = wr12 + wr4, wr12 - wr4, wr11 + wr5, wr11 - wr5, wr10 + wr6, wr10 - wr6, wr9 + wr7, wr9 - wr7
    VFMADD231PS(ymm_wr_CCBBAA99, ymm_wr_44556677, Constant.float32x8(+1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0))
    # wi00112233 =   _, wi0, wi1 - wi15, wi1 + wi15, wi2 - wi14, wi2 + wi14, wi3 - wi13, wi3 + wi13
    VFMADD231PS(ymm_wi_00112233, ymm_wi_88FFEEDD, Constant.float32x8(0.0, 0.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0))
    # wi44556677 = wi4 - wi12, wi4 + wi12, wi5 - wi11, wi5 + wi11, wi6 - wi10, wi6 + wi10, wi7 - wi9, wi7 + wi9
    VADDSUBPS(ymm_wi_44556677, ymm_wi_44556677, ymm_wi_CCBBAA99)

    # xhr_lo =   -, wi0, wr1 + wr15, wi1 + wi15, wr2 + wr14, wi2 + wi14, wr3 + wr13, wi3 + wi13
    ymm_xhr_lo, ymm_xhr_hi = YMMRegister(), YMMRegister()
    VBLENDPS(ymm_xhr_lo, ymm_wr_88FFEEDD, ymm_wi_00112233, 0b10101010)
    VBLENDPS(ymm_xhr_hi, ymm_wr_CCBBAA99, ymm_wi_44556677, 0b10101010)
    # xhi_lo = wr8,   -, wi1 - wi15, wr15 - wr1, wi2 - wi14, wr14 - wr2, wi3 - wi13, wr13 - wr3
    ymm_xhi_lo, ymm_xhi_hi = YMMRegister(), YMMRegister()
    VBLENDPS(ymm_xhi_lo, ymm_wr_88FFEEDD, ymm_wi_00112233, 0b01010110)
    VBLENDPS(ymm_xhi_hi, ymm_wr_CCBBAA99, ymm_wi_44556677, 0b01010101)

    # xhr_lo = wr0, wi0, wr1 + wr15, wi1 + wi15, wr2 + wr14, wi2 + wi14, wr3 + wr13, wi3 + wi13
    VBLENDPS(ymm_xhr_lo, ymm_xhr_lo, ymm_wr_00112233, 0b00000001)

    # xhi_lo = wr8, wi8, wi1 - wi7, wr7 - wr1, wi2 - wi6, wr6 - wr2, wi3 - wi5, wr5 - wr3
    VBLENDPS(ymm_xhi_lo, ymm_xhi_lo, ymm_wi_88FFEEDD, 0b00000010)

    ymm_scale_factor_lo = YMMRegister()
    VMOVAPS(ymm_scale_factor_lo, Constant.float32x8(1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    VMULPS(ymm_xhr_lo, ymm_xhr_lo, ymm_scale_factor_lo)
    VMULPS(ymm_xhi_lo, ymm_xhi_lo, ymm_scale_factor_lo)

    ymm_scale_factor_hi = YMMRegister()
    VMOVAPS(ymm_scale_factor_hi, Constant.float32x8(0.5))
    VMULPS(ymm_xhr_hi, ymm_xhr_hi, ymm_scale_factor_hi)
    VMULPS(ymm_xhi_hi, ymm_xhi_hi, ymm_scale_factor_hi)

    # wRe[0], wIm[0], (wRe[1] + wRe[15]) / 2, (wIm[1] + wIm[15]) / 2,
    #   (wRe[2] + wRe[14]) / 2, (wIm[2] + wIm[14]) / 2, (wRe[3] + wRe[13]) / 2, (wIm[3] + wIm[13]) / 2

    # (wRe[4] + wRe[12]) / 2, (wIm[4] + wIm[12]) / 2, (wRe[5] + wRe[11]) / 2, (wIm[5] + wIm[11]) / 2,
    #   (wRe[6] + wRe[10]) / 2, (wIm[6] + wIm[10]) / 2, (wRe[7] + wRe[9]) / 2, (wIm[7] + wIm[9]) / 2

    # wRe[8], wIm[8], (wIm[1] - wIm[15]) / 2, (wRe[15] - wRe[1]) / 2,
    #   (wIm[2] - wIm[14]) / 2, (wRe[14] - wRe[2]) / 2, (wIm[3] - wIm[13]) / 2, (wRe[13] - wRe[3]) / 2

    # (wIm[4] - wIm[12]) / 2, (wRe[12] - wRe[4]) / 2, (wIm[5] - wIm[11]) / 2, (wRe[11] - wRe[5]) / 2,
    #   (wIm[6] - wIm[10]) / 2, (wRe[6] - wRe[10]) / 2, (wIm[7] - wIm[9]) / 2, (wRe[13] - wRe[3]) / 2

    SWAP.REGISTERS(ymm_xhr_lo, ymm_wr[0])
    SWAP.REGISTERS(ymm_xhr_hi, ymm_wr[1])
    SWAP.REGISTERS(ymm_xhi_lo, ymm_wi[0])
    SWAP.REGISTERS(ymm_xhi_hi, ymm_wi[1])
