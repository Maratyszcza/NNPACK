from __future__ import absolute_import
from __future__ import division

import fft.complex_soa

arg_t = Argument(ptr(const_float_), name="t")
arg_f = Argument(ptr(float_), name="f")


with Function("nnp_fft4_8aos__fma3",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_data = [YMMRegister() for _ in range(8)]
    ymm_real, ymm_imag = ymm_data[0::2], ymm_data[1::2]

    for i, ymm_i in enumerate(ymm_data):
        VMOVUPS(ymm_i, [reg_t + i * YMMRegister.size])

    fft.complex_soa.fft4_across_rows(ymm_real, ymm_imag)

    for i, ymm_i in enumerate(ymm_data):
        VMOVUPS([reg_f + i * YMMRegister.size], ymm_i)

    RETURN()


from common import butterfly, sqrt2_over_2


def fft8_bitreverse(n):
    return int(format(n, "03b")[::-1], 2)


with Function("nnp_fft8_8aos__fma3",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    data = [YMMRegister() for _ in range(16)]
    data[0] = LocalVariable(data[0])
    data[8] = LocalVariable(data[8])
    real, imag = data[0::2], data[1::2]

    for i, (data_lo, data_hi) in enumerate(zip(data[0:8], data[8:16])):
        ymm_data_lo, ymm_data_hi = data_lo, data_hi
        if isinstance(data_lo, LocalVariable):
            ymm_data_lo = YMMRegister()
        if isinstance(data_hi, LocalVariable):
            ymm_data_hi = YMMRegister()

        VMOVUPS(ymm_data_lo, [reg_t + i * YMMRegister.size])
        VMOVUPS(ymm_data_hi, [reg_t + (i + 8) * YMMRegister.size])

        butterfly(ymm_data_lo, ymm_data_hi)

        if isinstance(data_lo, LocalVariable):
            VMOVAPS(data_lo, ymm_data_lo)
        if isinstance(data_hi, LocalVariable):
            VMOVAPS(data_hi, ymm_data_hi)

    # FFT8: multiplication by twiddle factors
    fft4_scale_b, fft4_negate_b = {}, {}
    fft2_scale_b, fft2_negate_b = {}, {}

    # w6.re, w6.im = w6.im, -w6.re
    SWAP.REGISTERS(real[6], imag[6])
    fft4_negate_b[id(imag[6])] = True

    # w5.re, w5.im =  SQRT2_OVER_2 * (w5.re + w5.im),  SQRT2_OVER_2 * (w5.im - w5.re)
    butterfly(imag[5], real[5])
    SWAP.REGISTERS(real[5], imag[5])

    # w7.re, w7.im = -SQRT2_OVER_2 * (w7.re - w7.im), -SQRT2_OVER_2 * (w7.re + w7.im)
    butterfly(real[7], imag[7], negate_b=True)
    fft4_negate_b[id(real[7])] = True
    fft4_negate_b[id(imag[7])] = True

    # Propogate multiplication by sqrt2_over_2 until the last butterfly in FFT2
    ymm_sqrt2_over_2 = YMMRegister()
    fft2_scale_b[id(real[5])] = ymm_sqrt2_over_2
    fft2_scale_b[id(imag[5])] = ymm_sqrt2_over_2
    fft2_scale_b[id(real[7])] = ymm_sqrt2_over_2
    fft2_scale_b[id(imag[7])] = ymm_sqrt2_over_2

    # 2x FFT4: butterfly
    for data_lo, data_hi in zip(data[0:4] + data[8:12], data[4:8] + data[12:16]):
        butterfly(data_lo, data_hi, negate_b=fft4_negate_b.get(id(data_hi), False), scale_b=fft4_scale_b.get(id(data_hi)))

    # 2x FFT4: multiplication by twiddle factors

    # w3.re, w3.im = w3.im, -w3.re
    # w7.re, w7.im = w7.im, -w7.re
    SWAP.REGISTERS(real[3], imag[3])
    SWAP.REGISTERS(real[7], imag[7])
    fft2_negate_b[id(imag[3])] = True
    fft2_negate_b[id(imag[7])] = True

    # 4x FFT2: butterfly
    for i, (data_lo, data_hi) in enumerate(zip(data[0:2] + data[4:6] + data[8:10] + data[12:14], data[2:4] + data[6:8] + data[10:12] + data[14:16])):
        ymm_data_lo, ymm_data_hi = \
            butterfly(data_lo, data_hi,
                negate_b=fft2_negate_b.get(id(data_hi), False), scale_b=fft2_scale_b.get(id(data_hi)),
                writeback=False)

        index_lo = (i // 2) * 2
        index_hi = index_lo + 1

        VMOVUPS([reg_f + (fft8_bitreverse(index_lo) * 2 + i % 2) * YMMRegister.size], ymm_data_lo)
        VMOVUPS([reg_f + (fft8_bitreverse(index_hi) * 2 + i % 2) * YMMRegister.size], ymm_data_hi)

        if i == 0:
            VMOVAPS(ymm_sqrt2_over_2, Constant.float32x8(sqrt2_over_2))

    RETURN()


with Function("nnp_ifft8_8aos__fma3",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    data = [YMMRegister() for _ in range(16)]
    data[0] = LocalVariable(data[0])
    data[8] = LocalVariable(data[8])
    real, imag = data[0::2], data[1::2]

    for i, (data_lo, data_hi) in enumerate(zip(data[0:8], data[8:16])):
        ymm_data_lo, ymm_data_hi = data_lo, data_hi
        if isinstance(data_lo, LocalVariable):
            ymm_data_lo = YMMRegister()
        if isinstance(data_hi, LocalVariable):
            ymm_data_hi = YMMRegister()

        VMOVUPS(ymm_data_lo, [reg_t + i * YMMRegister.size])
        VMOVUPS(ymm_data_hi, [reg_t + (i + 8) * YMMRegister.size])

        butterfly(ymm_data_lo, ymm_data_hi)

        if isinstance(data_lo, LocalVariable):
            VMOVAPS(data_lo, ymm_data_lo)
        if isinstance(data_hi, LocalVariable):
            VMOVAPS(data_hi, ymm_data_hi)

    # FFT8: multiplication by twiddle factors
    fft4_scale_b, fft4_negate_b = {}, {}
    fft2_scale_b, fft2_negate_b = {}, {}

    # w6.re, w6.im = -w6.im, w6.re
    SWAP.REGISTERS(real[6], imag[6])
    fft4_negate_b[id(real[6])] = True

    # w5.re, w5.im =  SQRT2_OVER_2 * (w5.re - w5.im), SQRT2_OVER_2 * (w5.re + w5.im)
    butterfly(real[5], imag[5], negate_b=True)

    # w7.re, w7.im = -SQRT2_OVER_2 * (w7.re + w7.im), SQRT2_OVER_2 * (w7.re - w7.im)
    butterfly(real[7], imag[7])
    fft4_negate_b[id(real[7])] = True

    # Propogate multiplication by sqrt2_over_2 until the last butterfly in FFT2
    fft2_scale_b[id(real[5])] = Constant.float32x8(sqrt2_over_2)
    fft2_scale_b[id(imag[5])] = Constant.float32x8(sqrt2_over_2)
    fft2_scale_b[id(real[7])] = Constant.float32x8(sqrt2_over_2)
    fft2_scale_b[id(imag[7])] = Constant.float32x8(sqrt2_over_2)

    # 2x FFT4: butterfly
    for data_lo, data_hi in zip(data[0:4] + data[8:12], data[4:8] + data[12:16]):
        butterfly(data_lo, data_hi, negate_b=fft4_negate_b.get(id(data_hi), False), scale_b=fft4_scale_b.get(id(data_hi)))

    # 2x FFT4: multiplication by twiddle factors

    # w3.re, w3.im = -w3.im, w3.re
    # w7.re, w7.im = -w7.im, w7.re
    SWAP.REGISTERS(real[3], imag[3])
    SWAP.REGISTERS(real[7], imag[7])
    fft2_negate_b[id(real[3])] = True
    fft2_negate_b[id(real[7])] = True

    # 4x FFT2: butterfly
    for i, (data_lo, data_hi) in enumerate(zip(data[0:2] + data[4:6] + data[8:10] + data[12:14], data[2:4] + data[6:8] + data[10:12] + data[14:16])):
        ymm_data_lo, ymm_data_hi = \
            butterfly(data_lo, data_hi,
                negate_b=fft2_negate_b.get(id(data_hi), False), scale_b=fft2_scale_b.get(id(data_hi)),
                writeback=False)

        index_lo = (i // 2) * 2
        index_hi = index_lo + 1

        VMULPS(ymm_data_lo, ymm_data_lo, Constant.float32x8(0.125))
        VMULPS(ymm_data_hi, ymm_data_hi, Constant.float32x8(0.125))
        VMOVUPS([reg_f + (fft8_bitreverse(index_lo) * 2 + i % 2) * YMMRegister.size], ymm_data_lo)
        VMOVUPS([reg_f + (fft8_bitreverse(index_hi) * 2 + i % 2) * YMMRegister.size], ymm_data_hi)

    RETURN()
