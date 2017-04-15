from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *

from common import sqrt2_over_2
from common import butterfly

import fft.complex_soa


def fft8_across_rows(ymm_data):
    assert isinstance(ymm_data, list) and len(ymm_data) == 8
    ymm_real = ymm_data[0::2]
    ymm_imag = ymm_data[1::2]

    fft.complex_soa.fft4_across_rows(ymm_real, ymm_imag)

    butterfly(ymm_real[0], ymm_imag[0])

    # const float two_gdata1_real = crealf(data1) + crealf(data3);
    # const float two_gdata1_imag = cimagf(data1) - cimagf(data3);
    ymm_two_gdata1_real, ymm_two_gdata1_imag = YMMRegister(), YMMRegister()
    VADDPS(ymm_two_gdata1_real, ymm_real[1], ymm_real[3])
    VSUBPS(ymm_two_gdata1_imag, ymm_imag[1], ymm_imag[3])

    # const float two_hdata1_real = cimagf(data1) + cimagf(data3);
    # const float two_hdata1_imag = crealf(data3) - crealf(data1);
    ymm_two_hdata1_real, ymm_two_hdata1_imag = YMMRegister(), YMMRegister()
    VADDPS(ymm_two_hdata1_real, ymm_imag[1], ymm_imag[3])
    VSUBPS(ymm_two_hdata1_imag, ymm_real[3], ymm_real[1])

    # const float two_hdata1_real_plus_imag = two_hdata1_real + two_hdata1_imag;
    # const float two_hdata1_real_minus_imag = two_hdata1_real - two_hdata1_imag;
    ymm_two_hdata1_plus, ymm_two_hdata1_minus = YMMRegister(), YMMRegister()
    VADDPS(ymm_two_hdata1_plus, ymm_two_hdata1_real, ymm_two_hdata1_imag)
    VSUBPS(ymm_two_hdata1_minus, ymm_two_hdata1_real, ymm_two_hdata1_imag)

    ymm_sqrt2_over_2 = YMMRegister()
    VMOVAPS(ymm_sqrt2_over_2, Constant.float32x8(sqrt2_over_2))

    # const float two_data1_real = two_gdata1_real + SQRT2_OVER_2 * two_hdata1_real_plus_imag;
    # const float two_data1_imag = two_gdata1_imag - SQRT2_OVER_2 * two_hdata1_real_minus_imag;
    # const float two_data3_real = two_gdata1_real - SQRT2_OVER_2 * two_hdata1_real_plus_imag;
    # const float two_data3_imag = -two_gdata1_imag - SQRT2_OVER_2 * two_hdata1_real_minus_imag;
    ymm_two_data1_real, ymm_two_data1_imag = YMMRegister(), YMMRegister()
    ymm_two_data3_real, ymm_two_data3_imag = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_two_data3_real, ymm_two_gdata1_real)
    VMOVAPS(ymm_two_data3_imag, ymm_two_gdata1_imag)
    VFMADD231PS(ymm_two_gdata1_real, ymm_two_hdata1_plus, ymm_sqrt2_over_2)
    VFNMADD231PS(ymm_two_gdata1_imag, ymm_two_hdata1_minus, ymm_sqrt2_over_2)
    SWAP.REGISTERS(ymm_two_data1_real, ymm_two_gdata1_real)
    SWAP.REGISTERS(ymm_two_data1_imag, ymm_two_gdata1_imag)
    VFNMADD231PS(ymm_two_data3_real, ymm_two_hdata1_plus, ymm_sqrt2_over_2)
    VFNMSUB231PS(ymm_two_data3_imag, ymm_two_hdata1_minus, ymm_sqrt2_over_2)

    # /* Store outputs */
    # fdata[0] = crealf(data0) + cimagf(data0);
    # fdata[1] = crealf(data0) - cimagf(data0);
    # fdata[2] = 0.5f * two_data1_real;
    # fdata[3] = 0.5f * two_data1_imag;
    # fdata[4] = crealf(data2);
    # fdata[5] = -cimagf(data2);
    # fdata[6] = 0.5f * two_data3_real;
    # fdata[7] = 0.5f * two_data3_imag;

    ymm_half = YMMRegister()
    VMOVAPS(ymm_half, Constant.float32x8(0.5))
    VMULPS(ymm_real[1], ymm_two_data1_real, ymm_half)
    VMULPS(ymm_imag[1], ymm_two_data1_imag, ymm_half)
    VXORPS(ymm_imag[2], ymm_imag[2], Constant.float32x8(-0.0))
    VMULPS(ymm_real[3], ymm_two_data3_real, ymm_half)
    VMULPS(ymm_imag[3], ymm_two_data3_imag, ymm_half)
