from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *

from common import sqrt2_over_2
from common import butterfly

import fft.complex_soa

def ifft8_across_rows(ymm_data, bias=None):
    assert isinstance(ymm_data, list) and len(ymm_data) == 8
    ymm_real = ymm_data[0::2]
    ymm_imag = ymm_data[1::2]

    if bias is None:
        # Do 1/N scaling before IFFT
        ymm_one_eighth = YMMRegister()
        VMOVAPS(ymm_one_eighth, Constant.float32x8(0.125))
        for ymm_row in ymm_data:
            if ymm_row is ymm_real[2]:
                VMULPS(ymm_row, ymm_row, Constant.float32x8(0.25))
            elif ymm_row is ymm_imag[2]:
                VMULPS(ymm_row, ymm_row, Constant.float32x8(-0.25))
            else:
                VMULPS(ymm_row, ymm_row, ymm_one_eighth)
    else:
        # Do 1/N scaling after FFT (merge with bias addition)
        VMULPS(ymm_real[2], ymm_real[2], Constant.float32x8(2.0))
        VMULPS(ymm_imag[2], ymm_imag[2], Constant.float32x8(-2.0))

    butterfly(ymm_real[0], ymm_imag[0])

    # H1.real, H1.imag = W1.real - W3.real, W1.imag + W3.imag
    ymm_h1_real, ymm_h1_imag = YMMRegister(), YMMRegister()
    VSUBPS(ymm_h1_real, ymm_real[1], ymm_real[3])
    VADDPS(ymm_h1_imag, ymm_imag[1], ymm_imag[3])

    # G1.real, G1.imag = W1.real + W3.real, W1.imag - W3.imag
    ymm_g1_real, ymm_g1_imag = YMMRegister(), YMMRegister()
    VADDPS(ymm_g1_real, ymm_real[1], ymm_real[3])
    VSUBPS(ymm_g1_imag, ymm_imag[1], ymm_imag[3])

    # H1+, H1- = H1.real + H1.imag, H1.real - H1.imag
    ymm_h1_plus, ymm_h1_minus = YMMRegister(), YMMRegister()
    VADDPS(ymm_h1_plus, ymm_h1_real, ymm_h1_imag)
    VSUBPS(ymm_h1_minus, ymm_h1_real, ymm_h1_imag)

    ymm_sqrt2_over_2 = YMMRegister()
    VMOVAPS(ymm_sqrt2_over_2, Constant.float32x8(sqrt2_over_2))

    # w1.real =  G1.real - SQRT2_OVER_2 * H1.plus;
    # w3.real =  G1.real + SQRT2_OVER_2 * H1.plus;
    VMOVAPS(ymm_real[1], ymm_g1_real)
    VFNMADD231PS(ymm_real[1], ymm_h1_plus, ymm_sqrt2_over_2)
    VFMADD231PS(ymm_g1_real, ymm_h1_plus, ymm_sqrt2_over_2)
    SWAP.REGISTERS(ymm_real[3], ymm_g1_real)

    # w1.imag =  G1.imag + SQRT2_OVER_2 * H1.minus;
    # w3.imag = -G1.imag + SQRT2_OVER_2 * H1.minus;
    VMOVAPS(ymm_imag[1], ymm_g1_imag)
    VFMADD231PS(ymm_imag[1], ymm_h1_minus, ymm_sqrt2_over_2)
    VFMSUB231PS(ymm_g1_imag, ymm_h1_minus, ymm_sqrt2_over_2)
    SWAP.REGISTERS(ymm_imag[3], ymm_g1_imag)

    fft.complex_soa.fft4_across_rows(ymm_real, ymm_imag, transformation="inverse")

    if bias is not None:
        ymm_bias = bias
        if not isinstance(bias, YMMRegister):
            ymm_bias = YMMRegister()
            VMOVAPS(ymm_bias, bias)

        ymm_one_eighth = YMMRegister()
        VMOVAPS(ymm_one_eighth, Constant.float32x8(0.125))

        # 1/N scaling
        for ymm_row in ymm_data:
            VFMADD132PS(ymm_row, ymm_bias, ymm_one_eighth)
