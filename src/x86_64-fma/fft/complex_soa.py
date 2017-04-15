from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *

from common import cos_npi_over_8, sin_npi_over_8, cos_npi_over_4, sin_npi_over_4
from common import _MM_SHUFFLE
from common import butterfly, transpose2x2x128, transpose2x2x2x64, interleave

def fft8_within_rows(ymm_real_rows, ymm_imag_rows, transformation="forward"):
    if isinstance(ymm_real_rows, YMMRegister) and isinstance(ymm_imag_rows, YMMRegister):
        return fft8_within_rows([ymm_real_rows], [ymm_imag_rows], transformation)

    assert isinstance(ymm_real_rows, list) and all(isinstance(ymm_real, YMMRegister) for ymm_real in ymm_real_rows)
    assert isinstance(ymm_imag_rows, list) and all(isinstance(ymm_imag, YMMRegister) for ymm_imag in ymm_imag_rows)
    assert transformation in {"forward", "inverse"}

    ymm_fft8_butterfly_factor = YMMRegister()
    VMOVAPS(ymm_fft8_butterfly_factor, Constant.float32x8(+1.0, +1.0, +1.0, +1.0, -1.0, -1.0, -1.0, -1.0))

    # FFT8: Butterfly
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_real_flipped = YMMRegister()
        VPERM2F128(ymm_real_flipped, ymm_real, ymm_real, 0x01)
        VFMADD132PS(ymm_real, ymm_real_flipped, ymm_fft8_butterfly_factor)
        ymm_imag_flipped = YMMRegister()
        VPERM2F128(ymm_imag_flipped, ymm_imag, ymm_imag, 0x01)
        VFMADD132PS(ymm_imag, ymm_imag_flipped, ymm_fft8_butterfly_factor)

    # FFT8: Multiplication by twiddle factors
    ymm_fft8_cos_twiddle_factor = YMMRegister()
    VMOVAPS(ymm_fft8_cos_twiddle_factor, Constant.float32x8(1.0, 1.0, 1.0, 1.0, cos_npi_over_4[0], cos_npi_over_4[1], cos_npi_over_4[2], cos_npi_over_4[3]))

    ymm_fft8_sin_twiddle_factor = YMMRegister()
    VMOVAPS(ymm_fft8_sin_twiddle_factor, Constant.float32x8(0.0, 0.0, 0.0, 0.0, sin_npi_over_4[0], sin_npi_over_4[1], sin_npi_over_4[2], sin_npi_over_4[3]))

    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real, ymm_new_imag = YMMRegister(), YMMRegister()
        VMULPS(ymm_new_real, ymm_real, ymm_fft8_cos_twiddle_factor)
        VMULPS(ymm_new_imag, ymm_imag, ymm_fft8_cos_twiddle_factor)

        if transformation == "forward":
            VFMADD231PS(ymm_new_real, ymm_imag, ymm_fft8_sin_twiddle_factor)
            VFNMADD231PS(ymm_new_imag, ymm_real, ymm_fft8_sin_twiddle_factor)
        else:
            VFNMADD231PS(ymm_new_real, ymm_imag, ymm_fft8_sin_twiddle_factor)
            VFMADD231PS(ymm_new_imag, ymm_real, ymm_fft8_sin_twiddle_factor)

        SWAP.REGISTERS(ymm_real, ymm_new_real)
        SWAP.REGISTERS(ymm_imag, ymm_new_imag)

    # 2x FFT4: Butterfly
    ymm_fft4_butterfly_factor = YMMRegister()
    VMOVAPS(ymm_fft4_butterfly_factor, Constant.float32x8(+1.0, +1.0, -1.0, -1.0, +1.0, +1.0, -1.0, -1.0))

    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_real_flipped = YMMRegister()
        VPERMILPS(ymm_real_flipped, ymm_real, _MM_SHUFFLE(1, 0, 3, 2))
        VFMADD132PS(ymm_real, ymm_real_flipped, ymm_fft4_butterfly_factor)
        ymm_imag_flipped = YMMRegister()
        VPERMILPS(ymm_imag_flipped, ymm_imag, _MM_SHUFFLE(1, 0, 3, 2))
        VFMADD132PS(ymm_imag, ymm_imag_flipped, ymm_fft4_butterfly_factor)

    # 2x FFT4: Multiplication by twiddle factors
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real, ymm_new_imag = YMMRegister(), YMMRegister()
        VBLENDPS(ymm_new_real, ymm_real, ymm_imag, 0b10001000)
        VBLENDPS(ymm_new_imag, ymm_imag, ymm_real, 0b10001000)
        if transformation == "forward":
            VXORPS(ymm_new_imag, ymm_new_imag, Constant.float32x8(+0.0, +0.0, +0.0, -0.0, +0.0, +0.0, +0.0, -0.0))
        else:
            VXORPS(ymm_new_real, ymm_new_real, Constant.float32x8(+0.0, +0.0, +0.0, -0.0, +0.0, +0.0, +0.0, -0.0))
        SWAP.REGISTERS(ymm_real, ymm_new_real)
        SWAP.REGISTERS(ymm_imag, ymm_new_imag)

    # 4x FFT2: Butterfly
    ymm_fft2_butterfly_factor = YMMRegister()
    VMOVAPS(ymm_fft2_butterfly_factor, Constant.float32x8(+1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0))

    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_real_flipped = YMMRegister()
        VPERMILPS(ymm_real_flipped, ymm_real, _MM_SHUFFLE(2, 3, 0, 1))
        VFMADD132PS(ymm_real, ymm_real_flipped, ymm_fft2_butterfly_factor)
        ymm_imag_flipped = YMMRegister()
        VPERMILPS(ymm_imag_flipped, ymm_imag, _MM_SHUFFLE(2, 3, 0, 1))
        VFMADD132PS(ymm_imag, ymm_imag_flipped, ymm_fft2_butterfly_factor)

    # Bit reversal
    ymm_bit_reversal_mask = YMMRegister()
    VMOVAPS(ymm_bit_reversal_mask, Constant.uint32x8(0, 4, 2, 6, 1, 5, 3, 7))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        VPERMPS(ymm_real, ymm_bit_reversal_mask, ymm_real)
        VPERMPS(ymm_imag, ymm_bit_reversal_mask, ymm_imag)

    # Scale
    if transformation == "inverse":
        ymm_scale_factor = YMMRegister()
        VMOVAPS(ymm_scale_factor, Constant.float32x8(0.125))
        for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
            VMULPS(ymm_real, ymm_real, ymm_scale_factor)
            VMULPS(ymm_imag, ymm_imag, ymm_scale_factor)


def fft16_within_rows(ymm_real_rows, ymm_imag_rows, bit_reversal=True):
    if isinstance(ymm_real_rows, tuple) and isinstance(ymm_imag_rows, tuple):
        return fft16_within_rows([ymm_real_rows], [ymm_imag_rows])

    assert isinstance(ymm_real_rows, list) and all(isinstance(ymm_real, tuple) and all(isinstance(ymm, YMMRegister) for ymm in ymm_real) for ymm_real in ymm_real_rows)
    assert isinstance(ymm_imag_rows, list) and all(isinstance(ymm_imag, tuple) and all(isinstance(ymm, YMMRegister) for ymm in ymm_imag) for ymm_imag in ymm_imag_rows)

    # FFT16: Butterfly
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1])
        butterfly(ymm_imag[0], ymm_imag[1])

    # FFT16: Multiplication by twiddle factors
    ymm_fft16_cos_twiddle_factor, ymm_fft16_sin_twiddle_factor = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_fft16_cos_twiddle_factor, Constant.float32x8(*cos_npi_over_8))
    VMOVAPS(ymm_fft16_sin_twiddle_factor, Constant.float32x8(*sin_npi_over_8))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real1, ymm_new_imag1 = YMMRegister(), YMMRegister()
        VMULPS(ymm_new_real1, ymm_real[1], ymm_fft16_cos_twiddle_factor)
        VMULPS(ymm_new_imag1, ymm_imag[1], ymm_fft16_cos_twiddle_factor)

        VFMADD231PS(ymm_new_real1, ymm_imag[1], ymm_fft16_sin_twiddle_factor)
        VFNMADD231PS(ymm_new_imag1, ymm_real[1], ymm_fft16_sin_twiddle_factor)

        SWAP.REGISTERS(ymm_real[1], ymm_new_real1)
        SWAP.REGISTERS(ymm_imag[1], ymm_new_imag1)

    # 2x FFT8: Butterfly
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        transpose2x2x128(ymm_real[0], ymm_real[1])
        transpose2x2x128(ymm_imag[0], ymm_imag[1])
    # w[0] = x0 x1 x2 x3  x8  x9 x10 x11
    # w[1] = x4 x5 x6 x7 x12 x13 x14 x15
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1])
        butterfly(ymm_imag[0], ymm_imag[1])

    # 2x FFT8: Multiplication by twiddle factors
    ymm_fft8_cos_twiddle_factor, ymm_fft8_sin_twiddle_factor = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_fft8_cos_twiddle_factor, Constant.float32x8(*(cos_npi_over_4 * 2)))
    VMOVAPS(ymm_fft8_sin_twiddle_factor, Constant.float32x8(*(sin_npi_over_4 * 2)))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real1, ymm_new_imag1 = YMMRegister(), YMMRegister()
        VMULPS(ymm_new_real1, ymm_real[1], ymm_fft8_cos_twiddle_factor)
        VMULPS(ymm_new_imag1, ymm_imag[1], ymm_fft8_cos_twiddle_factor)

        VFMADD231PS(ymm_new_real1, ymm_imag[1], ymm_fft8_sin_twiddle_factor)
        VFNMADD231PS(ymm_new_imag1, ymm_real[1], ymm_fft8_sin_twiddle_factor)

        SWAP.REGISTERS(ymm_real[1], ymm_new_real1)
        SWAP.REGISTERS(ymm_imag[1], ymm_new_imag1)

    # 4x FFT4: Butterfly
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        transpose2x2x2x64(ymm_real[0], ymm_real[1])
        transpose2x2x2x64(ymm_imag[0], ymm_imag[1])
    # w[0] = x0 x1 x4 x5  x8  x9 x12 x13
    # w[1] = x2 x3 x6 x7 x10 x11 x14 x15
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1])
        butterfly(ymm_imag[0], ymm_imag[1])

    # 4x FFT4: Multiplication by twiddle factors and 8x FFT2: Butterfly
    ymm_fft4_twiddle_factor = YMMRegister()
    VMOVAPS(ymm_fft4_twiddle_factor, Constant.float32x8(+1.0, +1.0, -1.0, -1.0, +1.0, +1.0, -1.0, -1.0))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real = YMMRegister(), YMMRegister()
        VSHUFPS(ymm_new_real[0], ymm_real[0], ymm_real[1], _MM_SHUFFLE(2, 0, 2, 0))
        VSHUFPS(ymm_new_real[1], ymm_real[0], ymm_imag[1], _MM_SHUFFLE(3, 1, 3, 1))
        butterfly(ymm_new_real[0], ymm_new_real[1])

        ymm_new_imag = YMMRegister(), YMMRegister()
        VSHUFPS(ymm_new_imag[0], ymm_imag[0], ymm_imag[1], _MM_SHUFFLE(2, 0, 2, 0))
        VSHUFPS(ymm_new_imag[1], ymm_imag[0], ymm_real[1], _MM_SHUFFLE(3, 1, 3, 1))
        butterfly(ymm_new_imag[0], ymm_new_imag[1], scale_b=ymm_fft4_twiddle_factor)

        SWAP.REGISTERS(ymm_real[0], ymm_new_real[0])
        SWAP.REGISTERS(ymm_real[1], ymm_new_real[1])
        SWAP.REGISTERS(ymm_imag[0], ymm_new_imag[0])
        SWAP.REGISTERS(ymm_imag[1], ymm_new_imag[1])

    # w[0] = x0 x4 x2 x6 x8 x12 x10 x14
    # w[1] = x1 x5 x3 x7 x9 x11 x13 x15

    if bit_reversal:
        # Bit reversal
        ymm_bit_reversal_mask = YMMRegister()
        VMOVDQA(ymm_bit_reversal_mask, Constant.uint32x8(0, 4, 1, 5, 2, 6, 3, 7))
        for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
            for i in range(2):
                VPERMPS(ymm_real[i], ymm_bit_reversal_mask, ymm_real[i])
                VPERMPS(ymm_imag[i], ymm_bit_reversal_mask, ymm_imag[i])


def ifft16_within_rows(ymm_real_rows, ymm_imag_rows, bit_reversal=True):
    if isinstance(ymm_real_rows, tuple) and isinstance(ymm_imag_rows, tuple):
        return ifft16_within_rows([ymm_real_rows], [ymm_imag_rows])

    assert isinstance(ymm_real_rows, list) and all(isinstance(ymm_real, tuple) and all(isinstance(ymm, YMMRegister) for ymm in ymm_real) for ymm_real in ymm_real_rows)
    assert isinstance(ymm_imag_rows, list) and all(isinstance(ymm_imag, tuple) and all(isinstance(ymm, YMMRegister) for ymm in ymm_imag) for ymm_imag in ymm_imag_rows)

    if bit_reversal:
        # Bit reversal
        # w[0] = x0 x8 x4 x12 x2 x10 x6 x14
        # w[1] = x1 x9 x5 x13 x3 x11 x7 x15
        ymm_bit_reversal_mask = YMMRegister()
        VMOVDQA(ymm_bit_reversal_mask, Constant.uint32x8(0, 2, 4, 6, 1, 3, 5, 7))
        for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
            for i in range(2):
                VPERMPS(ymm_real[i], ymm_bit_reversal_mask, ymm_real[i])
                VPERMPS(ymm_imag[i], ymm_bit_reversal_mask, ymm_imag[i])

    # 8x FFT2: Butterfly
    # w[0] = x0 x4 x2 x6 x8 x12 x10 x14
    # w[1] = x1 x5 x3 x7 x9 x13 x11 x15
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1])
        butterfly(ymm_imag[0], ymm_imag[1])

        ymm_new_real = YMMRegister(), YMMRegister()
        VUNPCKLPS(ymm_new_real[0], ymm_real[0], ymm_real[1])
        VUNPCKHPS(ymm_new_real[1], ymm_real[0], ymm_imag[1])

        ymm_new_imag = YMMRegister(), YMMRegister()
        VUNPCKLPS(ymm_new_imag[0], ymm_imag[0], ymm_imag[1])
        VUNPCKHPS(ymm_new_imag[1], ymm_imag[0], ymm_real[1])

        SWAP.REGISTERS(ymm_imag[0], ymm_new_imag[0])
        SWAP.REGISTERS(ymm_imag[1], ymm_new_imag[1])
        SWAP.REGISTERS(ymm_real[0], ymm_new_real[0])
        SWAP.REGISTERS(ymm_real[1], ymm_new_real[1])
    # w[0] = x0 x1 x4 x5 x8  x9  x12 x13
    # w[1] = x2 x3 x6 x7 x10 x11 x14 x15

    # 4x FFT4: Butterfly and multiplication by twiddle factors
    ymm_fft4_twiddle_factor = YMMRegister()
    VMOVAPS(ymm_fft4_twiddle_factor, Constant.float32x8(+1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1], scale_b=ymm_fft4_twiddle_factor)
        butterfly(ymm_imag[0], ymm_imag[1])
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        transpose2x2x2x64(ymm_real[0], ymm_real[1])
        transpose2x2x2x64(ymm_imag[0], ymm_imag[1])
    # w[0] = x0 x1 x2 x3  x8  x9 x10 x11
    # w[1] = x4 x5 x6 x7 x12 x13 x14 x15

    # 2x FFT8: Multiplication by twiddle factors
    ymm_fft8_cos_twiddle_factor, ymm_fft8_sin_twiddle_factor = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_fft8_cos_twiddle_factor, Constant.float32x8(*(cos_npi_over_4 * 2)))
    VMOVAPS(ymm_fft8_sin_twiddle_factor, Constant.float32x8(*(sin_npi_over_4 * 2)))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real1, ymm_new_imag1 = YMMRegister(), YMMRegister()
        VMULPS(ymm_new_real1, ymm_real[1], ymm_fft8_cos_twiddle_factor)
        VMULPS(ymm_new_imag1, ymm_imag[1], ymm_fft8_cos_twiddle_factor)

        VFNMADD231PS(ymm_new_real1, ymm_imag[1], ymm_fft8_sin_twiddle_factor)
        VFMADD231PS(ymm_new_imag1, ymm_real[1], ymm_fft8_sin_twiddle_factor)

        SWAP.REGISTERS(ymm_real[1], ymm_new_real1)
        SWAP.REGISTERS(ymm_imag[1], ymm_new_imag1)

    # 2x FFT8: Butterfly
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1])
        butterfly(ymm_imag[0], ymm_imag[1])
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        transpose2x2x128(ymm_real[0], ymm_real[1])
        transpose2x2x128(ymm_imag[0], ymm_imag[1])
    # w[0] = x0 x1  x2  x3  x4  x5  x6  x7
    # w[1] = x8 x9 x10 x11 x12 x13 x14 x15

    # FFT16: Multiplication by twiddle factors and scale
    scale_factor = 0.0625
    ymm_fft16_cos_scale_twiddle_factor, ymm_fft16_sin_scale_twiddle_factor = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_fft16_cos_scale_twiddle_factor, Constant.float32x8(*[cos * scale_factor for cos in cos_npi_over_8]))
    VMOVAPS(ymm_fft16_sin_scale_twiddle_factor, Constant.float32x8(*[sin * scale_factor for sin in sin_npi_over_8]))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        ymm_new_real1, ymm_new_imag1 = YMMRegister(), YMMRegister()
        VMULPS(ymm_new_real1, ymm_real[1], ymm_fft16_cos_scale_twiddle_factor)
        VMULPS(ymm_new_imag1, ymm_imag[1], ymm_fft16_cos_scale_twiddle_factor)

        VFNMADD231PS(ymm_new_real1, ymm_imag[1], ymm_fft16_sin_scale_twiddle_factor)
        VFMADD231PS(ymm_new_imag1, ymm_real[1], ymm_fft16_sin_scale_twiddle_factor)

        SWAP.REGISTERS(ymm_real[1], ymm_new_real1)
        SWAP.REGISTERS(ymm_imag[1], ymm_new_imag1)

    # FFT16: Butterfly and scale
    ymm_scale_factor = YMMRegister()
    VMOVAPS(ymm_scale_factor, Constant.float32x8(scale_factor))
    for ymm_real, ymm_imag in zip(ymm_real_rows, ymm_imag_rows):
        butterfly(ymm_real[0], ymm_real[1], scale_a=ymm_scale_factor)
        butterfly(ymm_imag[0], ymm_imag[1], scale_a=ymm_scale_factor)


def fft4_across_rows(ymm_real, ymm_imag, transformation="forward"):
    assert isinstance(ymm_real, list) and len(ymm_real) == 4
    assert isinstance(ymm_imag, list) and len(ymm_imag) == 4
    assert transformation in {"forward", "inverse"}
    ymm_data = sum(zip(ymm_real, ymm_imag), ())

    # FFT-4 Butterfly
    for i in range(4):
        butterfly(ymm_data[i], ymm_data[i + 4])

    # Multiply by FFT-4 twiddle factors
    SWAP.REGISTERS(ymm_real[3], ymm_imag[3])

    # 2x FFT-2 Butterfly
    butterfly(ymm_data[0], ymm_data[2])
    butterfly(ymm_data[1], ymm_data[3])
    if transformation == "forward":
        butterfly(ymm_data[4], ymm_data[6])
        butterfly(ymm_data[5], ymm_data[7], negate_b=True)
    else:
        butterfly(ymm_data[4], ymm_data[6], negate_b=True)
        butterfly(ymm_data[5], ymm_data[7])

    # Bit reversal: not needed
    SWAP.REGISTERS(ymm_real[1], ymm_real[2])
    SWAP.REGISTERS(ymm_imag[1], ymm_imag[2])
