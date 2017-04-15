from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *
from common import butterfly, sqrt2_over_2


from common import butterfly, sqrt2_over_2, cos_npi_over_8, interleave


def fft8_bitreverse(n):
    return int(format(n, "03b")[::-1], 2)


def load_ymm_variable(variable):
    assert isinstance(variable, (YMMRegister, LocalVariable))
    ymm_variable = variable
    if isinstance(variable, LocalVariable):
        assert variable.size == YMMRegister.size
        ymm_variable = YMMRegister()
        VMOVAPS(ymm_variable, variable)
    return ymm_variable


def store_ymm_result(variable, result):
    assert isinstance(result, YMMRegister)

    if isinstance(variable, YMMRegister):
        SWAP.REGISTERS(variable, result)
    else:
        VMOVAPS(variable, result)


def forward_vfft(reg_t0, reg_t8, reg_t_stride, data_out, reg_row_start=None, reg_row_end=None, ymm_load_mask=None):
    assert isinstance(reg_t0, GeneralPurposeRegister64)
    assert isinstance(reg_t8, GeneralPurposeRegister64)
    assert isinstance(reg_t_stride, GeneralPurposeRegister64)
    assert isinstance(data_out, list) and len(data_out) == 16
    assert ymm_load_mask is None or isinstance(ymm_load_mask, YMMRegister)

    out_real, out_imag = data_out[0::2], data_out[1::2]

    real, imag = [YMMRegister() for _ in range(8)], [YMMRegister() for _ in range(8)]
    imag[0] = LocalVariable(YMMRegister.size)
    imag[4] = LocalVariable(YMMRegister.size)
    data = interleave(real, imag)

    for i, (data_lo, data_hi) in enumerate(zip(data[0:8], data[8:16])):
        row_lo = i
        row_hi = row_lo + 8

        ymm_data_lo, ymm_data_hi = data_lo, data_hi
        if isinstance(data_lo, LocalVariable):
            ymm_data_lo = YMMRegister()
        if isinstance(data_hi, LocalVariable):
            ymm_data_hi = YMMRegister()

        VXORPS(ymm_data_lo, ymm_data_lo, ymm_data_lo)
        skip_data_lo = Label()
        if reg_row_start:
            CMP(reg_row_start, row_lo)
            JA(skip_data_lo)
        if reg_row_end:
            CMP(reg_row_end, row_lo)
            JBE(skip_data_lo)
        if ymm_load_mask is None:
            VMOVUPS(ymm_data_lo, [reg_t0])
        else:
            VMASKMOVPS(ymm_data_lo, ymm_load_mask, [reg_t0])
        if i + 1 != 8:
            ADD(reg_t0, reg_t_stride)
        LABEL(skip_data_lo)

        VMOVAPS(ymm_data_hi, ymm_data_lo)
        skip_data_hi = Label()
        if reg_row_start:
            CMP(reg_row_start, row_hi)
            JA(skip_data_hi)
        if reg_row_end:
            CMP(reg_row_end, row_hi)
            JBE(skip_data_hi)
        if ymm_load_mask is None:
            VMOVUPS(ymm_data_hi, [reg_t8])
            butterfly(ymm_data_lo, ymm_data_hi)
        else:
            ymm_temp_hi = YMMRegister()
            VMASKMOVPS(ymm_temp_hi, ymm_load_mask, [reg_t8])
            VSUBPS(ymm_data_hi, ymm_data_lo, ymm_temp_hi)
            VADDPS(ymm_data_lo, ymm_data_lo, ymm_temp_hi)
        if i + 1 != 8:
            ADD(reg_t8, reg_t_stride)
        LABEL(skip_data_hi)

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

    # Process the first two elements separately
    ymm_real0, ymm_real1 = butterfly(real[0], real[1], writeback=False)
    store_ymm_result(out_real[4], ymm_real1) # bit-reversal: 1->4
    ymm_imag0, ymm_imag1 = butterfly(imag[0], imag[1], negate_out_b=True, writeback=False)
    store_ymm_result(out_imag[4], ymm_imag1) # bit-reversal: 1->4

    VMOVAPS(ymm_sqrt2_over_2, Constant.float32x8(sqrt2_over_2))
    for i, (data_lo, data_hi) in enumerate(zip(data[4:6] + data[8:10] + data[12:14], data[6:8] + data[10:12] + data[14:16])):
        butterfly(data_lo, data_hi,
            negate_b=fft2_negate_b.get(id(data_hi), False), scale_b=fft2_scale_b.get(id(data_hi)))

    butterfly(ymm_real0, ymm_imag0)
    store_ymm_result(out_real[0], ymm_real0)
    store_ymm_result(out_imag[0], ymm_imag0)

    # Bit reversal
    for i in range(8):
        new_i = fft8_bitreverse(i)
        if new_i > i:
            real[i], real[new_i] = real[new_i], real[i]
            imag[i], imag[new_i] = imag[new_i], imag[i]
    data = interleave(real, imag)

    ymm_two_g2_real, ymm_two_g2_imag = YMMRegister(), YMMRegister()
    ymm_two_h2_real, ymm_two_h2_imag = YMMRegister(), YMMRegister()

    VADDPS(ymm_two_g2_real, real[2], real[6])
    VSUBPS(ymm_two_h2_imag, real[6], real[2])

    VSUBPS(ymm_two_g2_imag, imag[2], imag[6])
    VADDPS(ymm_two_h2_real, imag[2], imag[6])

    ymm_two_g1_real, ymm_two_g1_imag = YMMRegister(), YMMRegister()
    ymm_two_h1_real, ymm_two_h1_imag = YMMRegister(), YMMRegister()
    ymm_real1 = load_ymm_variable(real[1])
    VADDPS(ymm_two_g1_real, ymm_real1, real[7])
    VSUBPS(ymm_two_h1_imag, real[7], ymm_real1)

    ymm_imag1 = load_ymm_variable(imag[1])
    VSUBPS(ymm_two_g1_imag, ymm_imag1, imag[7])
    VADDPS(ymm_two_h1_real, ymm_imag1, imag[7])

    ymm_two_h2_add, ymm_two_h2_sub = YMMRegister(), YMMRegister()
    VADDPS(ymm_two_h2_add, ymm_two_h2_real, ymm_two_h2_imag)
    VSUBPS(ymm_two_h2_sub, ymm_two_h2_imag, ymm_two_h2_real)

    ymm_two_g3_real, ymm_two_g3_imag = YMMRegister(), YMMRegister()
    ymm_two_h3_real, ymm_two_h3_imag = YMMRegister(), YMMRegister()
    VADDPS(ymm_two_g3_real, real[3], real[5])
    VSUBPS(ymm_two_h3_imag, real[5], real[3])

    VSUBPS(ymm_two_g3_imag, imag[3], imag[5])
    VADDPS(ymm_two_h3_real, imag[3], imag[5])

    # const float two_w2_real = two_g2_real + SQRT2_OVER_2 * (two_h2_real + two_h2_imag);
    # const float two_w2_imag = two_g2_imag + SQRT2_OVER_2 * (two_h2_imag - two_h2_real);
    # const float two_w6_real = two_g2_real - SQRT2_OVER_2 * (two_h2_real + two_h2_imag);
    # const float two_w6_imag = -two_g2_imag + SQRT2_OVER_2 * (two_h2_imag - two_h2_real);

    ymm_sqrt2_over_2 = YMMRegister()
    VMOVAPS(ymm_sqrt2_over_2, Constant.float32x8(sqrt2_over_2))

    ymm_two_w2_real, ymm_two_w6_real = YMMRegister(), ymm_two_g2_real
    VMOVAPS(ymm_two_w2_real, ymm_two_g2_real)
    VFMADD231PS(ymm_two_w2_real, ymm_two_h2_add, ymm_sqrt2_over_2)
    VFNMADD231PS(ymm_two_w6_real, ymm_two_h2_add, ymm_sqrt2_over_2)

    ymm_two_w2_imag, ymm_two_w6_imag = YMMRegister(), ymm_two_g2_imag
    VMOVAPS(ymm_two_w2_imag, ymm_two_g2_imag)
    VFMADD231PS(ymm_two_w2_imag, ymm_two_h2_sub, ymm_sqrt2_over_2)
    VFMSUB231PS(ymm_two_w6_imag, ymm_two_h2_sub, ymm_sqrt2_over_2)

    ymm_half = YMMRegister()
    VMOVAPS(ymm_half, Constant.float32x8(0.5))

    VMULPS(ymm_two_w2_real, ymm_two_w2_real, ymm_half)
    store_ymm_result(out_real[2], ymm_two_w2_real)
    VMULPS(ymm_two_w6_real, ymm_two_w6_real, ymm_half)
    store_ymm_result(out_real[6], ymm_two_w6_real)

    VMULPS(ymm_two_w2_imag, ymm_two_w2_imag, ymm_half)
    store_ymm_result(out_imag[2], ymm_two_w2_imag)
    VMULPS(ymm_two_w6_imag, ymm_two_w6_imag, ymm_half)
    store_ymm_result(out_imag[6], ymm_two_w6_imag)

    # const float two_w1_real = two_g1_real + two_h1_real * COS_1PI_OVER_8 + two_h1_imag * COS_3PI_OVER_8;
    # const float two_w1_imag = two_g1_imag + two_h1_imag * COS_1PI_OVER_8 - two_h1_real * COS_3PI_OVER_8;
    # const float two_w7_real = two_g1_real - two_h1_real * COS_1PI_OVER_8 - two_h1_imag * COS_3PI_OVER_8;
    # const float two_w7_imag = -two_g1_imag + two_h1_imag * COS_1PI_OVER_8 - two_h1_real * COS_3PI_OVER_8;

    # const float two_w3_real = two_g3_real + two_h3_real * COS_3PI_OVER_8 + two_h3_imag * COS_1PI_OVER_8;
    # const float two_w3_imag = two_g3_imag + two_h3_imag * COS_3PI_OVER_8 - two_h3_real * COS_1PI_OVER_8;
    # const float two_w5_real = two_g3_real - two_h3_real * COS_3PI_OVER_8 - two_h3_imag * COS_1PI_OVER_8;
    # const float two_w5_imag = -two_g3_imag + two_h3_imag * COS_3PI_OVER_8 - two_h3_real * COS_1PI_OVER_8;

    ymm_cos_1pi_over_8 = YMMRegister()
    VMOVAPS(ymm_cos_1pi_over_8, Constant.float32x8(cos_npi_over_8[1]))

    ymm_two_w1_real, ymm_two_w7_real = YMMRegister(), ymm_two_g1_real
    VMOVAPS(ymm_two_w1_real, ymm_two_g1_real)
    VFMADD231PS(ymm_two_w1_real, ymm_two_h1_real, ymm_cos_1pi_over_8)
    VFNMADD231PS(ymm_two_w7_real, ymm_two_h1_real, ymm_cos_1pi_over_8)

    ymm_two_w1_imag, ymm_two_w7_imag = YMMRegister(), ymm_two_g1_imag
    VMOVAPS(ymm_two_w1_imag, ymm_two_g1_imag)
    VFMADD231PS(ymm_two_w1_imag, ymm_two_h1_imag, ymm_cos_1pi_over_8)
    VFMSUB231PS(ymm_two_w7_imag, ymm_two_h1_imag, ymm_cos_1pi_over_8)

    ymm_two_w3_real, ymm_two_w5_real = YMMRegister(), ymm_two_g3_real
    VMOVAPS(ymm_two_w3_real, ymm_two_g3_real)
    VFMADD231PS(ymm_two_w3_real, ymm_two_h3_imag, ymm_cos_1pi_over_8)
    VFNMADD231PS(ymm_two_w5_real, ymm_two_h3_imag, ymm_cos_1pi_over_8)

    ymm_two_w3_imag, ymm_two_w5_imag = YMMRegister(), ymm_two_g3_imag
    VMOVAPS(ymm_two_w3_imag, ymm_two_g3_imag)
    VFNMADD231PS(ymm_two_w3_imag, ymm_two_h3_real, ymm_cos_1pi_over_8)
    VFNMSUB231PS(ymm_two_w5_imag, ymm_two_h3_real, ymm_cos_1pi_over_8)

    ymm_cos_3pi_over_8 = YMMRegister()
    VMOVAPS(ymm_cos_3pi_over_8, Constant.float32x8(cos_npi_over_8[3]))

    VFMADD231PS(ymm_two_w1_real, ymm_two_h1_imag, ymm_cos_3pi_over_8)
    VFNMADD231PS(ymm_two_w7_real, ymm_two_h1_imag, ymm_cos_3pi_over_8)

    VFNMADD231PS(ymm_two_w1_imag, ymm_two_h1_real, ymm_cos_3pi_over_8)
    VFNMADD231PS(ymm_two_w7_imag, ymm_two_h1_real, ymm_cos_3pi_over_8)

    VFMADD231PS(ymm_two_w3_real, ymm_two_h3_real, ymm_cos_3pi_over_8)
    VFNMADD231PS(ymm_two_w5_real, ymm_two_h3_real, ymm_cos_3pi_over_8)

    VFMADD231PS(ymm_two_w3_imag, ymm_two_h3_imag, ymm_cos_3pi_over_8)
    VFMADD231PS(ymm_two_w5_imag, ymm_two_h3_imag, ymm_cos_3pi_over_8)

    ymm_half = YMMRegister()
    VMOVAPS(ymm_half, Constant.float32x8(0.5))

    VMULPS(ymm_two_w1_real, ymm_two_w1_real, ymm_half)
    store_ymm_result(out_real[1], ymm_two_w1_real)
    VMULPS(ymm_two_w7_real, ymm_two_w7_real, ymm_half)
    store_ymm_result(out_real[7], ymm_two_w7_real)

    VMULPS(ymm_two_w1_imag, ymm_two_w1_imag, ymm_half)
    store_ymm_result(out_imag[1], ymm_two_w1_imag)
    VMULPS(ymm_two_w7_imag, ymm_two_w7_imag, ymm_half)
    store_ymm_result(out_imag[7], ymm_two_w7_imag)

    VMULPS(ymm_two_w3_real, ymm_two_w3_real, ymm_half)
    store_ymm_result(out_real[3], ymm_two_w3_real)
    VMULPS(ymm_two_w5_real, ymm_two_w5_real, ymm_half)
    store_ymm_result(out_real[5], ymm_two_w5_real)

    VMULPS(ymm_two_w3_imag, ymm_two_w3_imag, ymm_half)
    store_ymm_result(out_imag[3], ymm_two_w3_imag)
    VMULPS(ymm_two_w5_imag, ymm_two_w5_imag, ymm_half)
    store_ymm_result(out_imag[5], ymm_two_w5_imag)


def inverse_vfft(reg_t0, reg_t8, reg_t_stride, data_in, reg_row_start=None, reg_row_end=None, store_mask=None, relu=False):
    assert isinstance(reg_t0, GeneralPurposeRegister64)
    assert isinstance(reg_t8, GeneralPurposeRegister64)
    assert isinstance(reg_t_stride, GeneralPurposeRegister64)
    assert isinstance(data_in, list) and len(data_in) == 16
    assert reg_row_end is None or isinstance(reg_row_end, GeneralPurposeRegister32)
    assert store_mask is None or isinstance(store_mask, LocalVariable) and store_mask.size == YMMRegister.size

    in_real, in_imag = data_in[0::2], data_in[1::2]

    ymm_scale_factor = YMMRegister()
    VMOVAPS(ymm_scale_factor, Constant.float32x8(0.0625))

    ymm_W1_real, ymm_W1_imag = YMMRegister(), YMMRegister()
    VMULPS(ymm_W1_real, ymm_scale_factor, in_real[1])
    VMULPS(ymm_W1_imag, ymm_scale_factor, in_imag[1])

    ymm_W2_real, ymm_W2_imag = YMMRegister(), YMMRegister()
    VMULPS(ymm_W2_real, ymm_scale_factor, in_real[2])
    VMULPS(ymm_W2_imag, ymm_scale_factor, in_imag[2])

    ymm_W3_real, ymm_W3_imag = YMMRegister(), YMMRegister()
    VMULPS(ymm_W3_real, ymm_scale_factor, in_real[3])
    VMULPS(ymm_W3_imag, ymm_scale_factor, in_imag[3])

    # G[n].real, H[n].real = W[n].real + W[8-n].real, W[n].real - W[8-n].real
    # G[n].imag, H[n].imag = W[n].imag - W[8-n].imag, W[n].imag + W[8-n].imag
    ymm_W7_real, ymm_W7_imag = YMMRegister(), YMMRegister()
    VMOVUPS(ymm_W7_real, in_real[7])
    ymm_G1_real, ymm_H1_real = butterfly(ymm_W1_real, ymm_W7_real, scale_b=ymm_scale_factor)
    VMOVUPS(ymm_W7_imag, in_imag[7])
    ymm_G1_imag, ymm_H1_imag = butterfly(ymm_W1_imag, ymm_W7_imag, scale_b=ymm_scale_factor, negate_b=True)

    ymm_W6_real, ymm_W6_imag = YMMRegister(), YMMRegister()
    VMOVUPS(ymm_W6_real, in_real[6])
    ymm_G2_real, ymm_H2_real = butterfly(ymm_W2_real, ymm_W6_real, scale_b=ymm_scale_factor)
    VMOVUPS(ymm_W6_imag, in_imag[6])
    ymm_G2_imag, ymm_H2_imag = butterfly(ymm_W2_imag, ymm_W6_imag, scale_b=ymm_scale_factor, negate_b=True)

    ymm_W5_real, ymm_W5_imag = YMMRegister(), YMMRegister()
    VMOVUPS(ymm_W5_real, in_real[5])
    ymm_G3_real, ymm_H3_real = butterfly(ymm_W3_real, ymm_W5_real, scale_b=ymm_scale_factor)
    VMOVUPS(ymm_W5_imag, in_imag[5])
    ymm_G3_imag, ymm_H3_imag = butterfly(ymm_W3_imag, ymm_W5_imag, scale_b=ymm_scale_factor, negate_b=True)

    # H[2]+, H[2]- = H[2].real + H[2].imag, H[2].real - H[2].imag
    ymm_H2_add, ymm_H2_sub = butterfly(ymm_H2_real, ymm_H2_imag)

    # w[   n].real =  G[  n].real - H[  n].real * cos((N-n)*pi/2N) - H[  n].imag * cos(n*pi/2N)
    # w[2N-n].real =  G[  n].real + H[  n].real * cos((N-n)*pi/2N) + H[  n].imag * cos(n*pi/2N)
    # w[   n].imag =  G[  n].imag + H[  n].real * cos(n*pi/2N)     - H[  n].imag * cos((N-n)*pi/2N)
    # w[2N-n].imag = -G[  n].imag + H[  n].real * cos(n*pi/2N)     - H[  n].imag * cos((N-n)*pi/2N)
    # w[ N-n].real =  G[N-n].real - H[N-n].real * cos(n*pi/2N)     - H[N-n].imag * cos((N-n)*pi/2N)
    # w[ N+n].real =  G[N-n].real + H[N-n].real * cos(n*pi/2N)     + H[N-n].imag * cos((N-n)*pi/2N)
    # w[ N-n].imag =  G[N-n].imag + H[N-n].real * cos((N-n)*pi/2N) - H[N-n].imag * cos(n*pi/2N)
    # w[ N+n].imag = -G[N-n].imag + H[N-n].real * cos((N-n)*pi/2N) - H[N-n].imag * cos(n*pi/2N)

    ymm_cos_1pi_over_8, ymm_cos_3pi_over_8 = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_cos_3pi_over_8, Constant.float32x8(cos_npi_over_8[3]))
    VMOVAPS(ymm_cos_1pi_over_8, Constant.float32x8(cos_npi_over_8[1]))

    ymm_w1_real, ymm_w7_real = YMMRegister(), ymm_G1_real
    VMOVAPS(ymm_w1_real, ymm_G1_real)
    VFNMADD231PS(ymm_w1_real, ymm_H1_real, ymm_cos_3pi_over_8)
    VFMADD231PS(ymm_w7_real, ymm_H1_real, ymm_cos_3pi_over_8)

    ymm_w1_imag, ymm_w7_imag = YMMRegister(), ymm_G1_imag
    VMOVAPS(ymm_w1_imag, ymm_G1_imag)
    VFMADD231PS(ymm_w1_imag, ymm_H1_real, ymm_cos_1pi_over_8)
    VFMSUB231PS(ymm_w7_imag, ymm_H1_real, ymm_cos_1pi_over_8)

    ymm_w3_real, ymm_w5_real = YMMRegister(), ymm_G3_real
    VMOVAPS(ymm_w3_real, ymm_G3_real)
    VFNMADD231PS(ymm_w3_real, ymm_H3_real, ymm_cos_1pi_over_8)
    VFMADD231PS(ymm_w5_real, ymm_H3_real, ymm_cos_1pi_over_8)

    ymm_w3_imag, ymm_w5_imag = YMMRegister(), ymm_G3_imag
    VMOVAPS(ymm_w3_imag, ymm_G3_imag)
    VFMADD231PS(ymm_w3_imag, ymm_H3_real, ymm_cos_3pi_over_8)
    VFMSUB231PS(ymm_w5_imag, ymm_H3_real, ymm_cos_3pi_over_8)

    ymm_sqrt2_over_2 = YMMRegister()
    VMOVAPS(ymm_sqrt2_over_2, Constant.float32x8(sqrt2_over_2))

    # w[ N/2].real =  G[N/2].real - H[N/2]+ * sqrt(2)/2
    # w[ N/2].imag =  G[N/2].imag + H[N/2]- * sqrt(2)/2
    # w[3N/2].real =  G[N/2].real + H[N/2]+ * sqrt(2)/2
    # w[3N/2].imag = -G[N/2].imag + H[N/2]- * sqrt(2)/2
    ymm_w2_real, ymm_w6_real = YMMRegister(), ymm_G2_real
    VMOVAPS(ymm_w2_real, ymm_G2_real)
    VFNMADD231PS(ymm_w2_real, ymm_H2_add, ymm_sqrt2_over_2)
    VFMADD231PS(ymm_w6_real, ymm_H2_add, ymm_sqrt2_over_2)

    ymm_w2_imag, ymm_w6_imag = YMMRegister(), ymm_G2_imag
    VMOVAPS(ymm_w2_imag, ymm_G2_imag)
    VFMADD231PS(ymm_w2_imag, ymm_H2_sub, ymm_sqrt2_over_2)
    VFMSUB231PS(ymm_w6_imag, ymm_H2_sub, ymm_sqrt2_over_2)

    # w[   n].real =  G[  n].real - H[  n].real * cos((N-n)*pi/2N) - H[  n].imag * cos(n*pi/2N)
    # w[2N-n].real =  G[  n].real + H[  n].real * cos((N-n)*pi/2N) + H[  n].imag * cos(n*pi/2N)
    # w[   n].imag =  G[  n].imag + H[  n].real * cos(n*pi/2N)     - H[  n].imag * cos((N-n)*pi/2N)
    # w[2N-n].imag = -G[  n].imag + H[  n].real * cos(n*pi/2N)     - H[  n].imag * cos((N-n)*pi/2N)
    # w[ N-n].real =  G[N-n].real - H[N-n].real * cos(n*pi/2N)     - H[N-n].imag * cos((N-n)*pi/2N)
    # w[ N+n].real =  G[N-n].real + H[N-n].real * cos(n*pi/2N)     + H[N-n].imag * cos((N-n)*pi/2N)
    # w[ N-n].imag =  G[N-n].imag + H[N-n].real * cos((N-n)*pi/2N) - H[N-n].imag * cos(n*pi/2N)
    # w[ N+n].imag = -G[N-n].imag + H[N-n].real * cos((N-n)*pi/2N) - H[N-n].imag * cos(n*pi/2N)

    ymm_cos_1pi_over_8, ymm_cos_3pi_over_8 = YMMRegister(), YMMRegister()
    VMOVAPS(ymm_cos_1pi_over_8, Constant.float32x8(cos_npi_over_8[1]))
    VMOVAPS(ymm_cos_3pi_over_8, Constant.float32x8(cos_npi_over_8[3]))

    VFNMADD231PS(ymm_w1_real, ymm_H1_imag, ymm_cos_1pi_over_8)
    VFMADD231PS(ymm_w7_real, ymm_H1_imag, ymm_cos_1pi_over_8)
    VFNMADD231PS(ymm_w1_imag, ymm_H1_imag, ymm_cos_3pi_over_8)
    VFNMADD231PS(ymm_w7_imag, ymm_H1_imag, ymm_cos_3pi_over_8)
    VFNMADD231PS(ymm_w3_real, ymm_H3_imag, ymm_cos_3pi_over_8)
    VFMADD231PS(ymm_w5_real, ymm_H3_imag, ymm_cos_3pi_over_8)
    VFNMADD231PS(ymm_w3_imag, ymm_H3_imag, ymm_cos_1pi_over_8)
    VFNMADD231PS(ymm_w5_imag, ymm_H3_imag, ymm_cos_1pi_over_8)

    data = [
        LocalVariable(YMMRegister.size), YMMRegister(),
        ymm_w1_real, ymm_w1_imag,
        ymm_w2_real, ymm_w2_imag,
        ymm_w3_real, ymm_w3_imag,
        LocalVariable(YMMRegister.size), LocalVariable(YMMRegister.size),
        ymm_w5_real, ymm_w5_imag,
        ymm_w6_real, ymm_w6_imag,
        ymm_w7_real, ymm_w7_imag
    ]
    real, imag = data[0::2], data[1::2]

    # TODO: optimize
    ymm_w0_real, ymm_w0_imag = YMMRegister(), imag[0]
    VMOVUPS(ymm_w0_real, in_real[0])
    VMOVUPS(ymm_w0_imag, in_imag[0])
    VMULPS(ymm_w0_real, ymm_w0_real, Constant.float32x8(0.0625))
    butterfly(ymm_w0_real, ymm_w0_imag, scale_b=Constant.float32x8(0.0625))
    VMOVAPS(real[0], ymm_w0_real)

    # TODO: optimize
    ymm_w4_real, ymm_w4_imag = YMMRegister(), YMMRegister()
    VMOVUPS(ymm_w4_real, in_real[4])
    VMOVUPS(ymm_w4_imag, in_imag[4])
    VMULPS(ymm_w4_real, ymm_w4_real, Constant.float32x8(0.125))
    VMULPS(ymm_w4_imag, ymm_w4_imag, Constant.float32x8(-0.125))
    VMOVAPS(real[4], ymm_w4_real)
    VMOVAPS(imag[4], ymm_w4_imag)

    # Bit reversal
    for i in range(8):
        new_i = fft8_bitreverse(i)
        if new_i > i:
            real[i], real[new_i] = real[new_i], real[i]
            imag[i], imag[new_i] = imag[new_i], imag[i]
    data = interleave(real, imag)

    # 4x FFT2: butterfly
    for i, (data_lo, data_hi) in enumerate(zip(data[0:2] + data[4:6] + data[8:10] + data[12:14], data[2:4] + data[6:8] + data[10:12] + data[14:16])):
        butterfly(data_lo, data_hi)

    # 2x FFT4: multiplication by twiddle factors
    fft4_scale_b, fft4_negate_b = {}, {}
    fft8_scale_b, fft8_negate_b = {}, {}

    # w3.re, w3.im = -w3.im, w3.re
    # w7.re, w7.im = -w7.im, w7.re
    SWAP.REGISTERS(real[3], imag[3])
    fft4_negate_b[id(real[3])] = True
    SWAP.REGISTERS(real[7], imag[7])
    fft4_negate_b[id(real[7])] = True

    # 2x FFT4: butterfly
    for data_lo, data_hi in zip(data[0:4] + data[8:12], data[4:8] + data[12:16]):
        butterfly(data_lo, data_hi, negate_b=fft4_negate_b.get(id(data_hi), False))

    # FFT8: multiplication by twiddle factors

    # w6.re, w6.im = -w6.im, w6.re
    SWAP.REGISTERS(real[6], imag[6])
    fft8_negate_b[id(real[6])] = True

    # w5.re, w5.im =  SQRT2_OVER_2 * (w5.re - w5.im), SQRT2_OVER_2 * (w5.re + w5.im)
    butterfly(real[5], imag[5], negate_b=True)
    fft8_scale_b[id(real[5])] = Constant.float32x8(sqrt2_over_2)
    fft8_scale_b[id(imag[5])] = Constant.float32x8(sqrt2_over_2)

    # w7.re, w7.im = -SQRT2_OVER_2 * (w7.re + w7.im), SQRT2_OVER_2 * (w7.re - w7.im)
    butterfly(real[7], imag[7])
    fft8_scale_b[id(real[7])] = Constant.float32x8(sqrt2_over_2)
    fft8_negate_b[id(real[7])] = True
    fft8_scale_b[id(imag[7])] = Constant.float32x8(sqrt2_over_2)

    ymm_store_mask = YMMRegister()
    if store_mask:
        VMOVAPS(ymm_store_mask, store_mask)

    # FFT8: butterfly
    with Block() as store_data:
        for i, (data_lo, data_hi) in enumerate(zip(data[0:8], data[8:16])):
            row_lo = i
            row_hi = row_lo + 8

            ymm_data_lo, ymm_data_hi = \
                butterfly(data_lo, data_hi,
                    scale_b=fft8_scale_b.get(id(data_hi)),
                    negate_b=fft8_negate_b.get(id(data_hi), False),
                    writeback=False)

            if relu:
                ymm_zero = YMMRegister()
                VMOVAPS(ymm_zero, Constant.float32x8(-0.0))

            with Block() as store_data_lo:
                if reg_row_start:
                    CMP(reg_row_start, row_lo)
                    JA(store_data_lo.end)
                    if reg_row_end:
                        CMP(reg_row_end, row_lo)
                        JBE(store_data_lo.end)
                elif reg_row_end:
                    CMP(reg_row_end, row_lo)
                    JBE(store_data.end)
                if relu:
                    VMAXPS(ymm_data_lo, ymm_zero, ymm_data_lo)
                if store_mask:
                    VMASKMOVPS([reg_t0], ymm_store_mask, ymm_data_lo)
                else:
                    VMOVUPS([reg_t0], ymm_data_lo)
                if i + 1 != 8:
                    ADD(reg_t0, reg_t_stride)

            with Block() as store_data_hi:
                if reg_row_start:
                    CMP(reg_row_start, row_hi)
                    JA(store_data_hi.end)
                if reg_row_end:
                    CMP(reg_row_end, row_hi)
                    JBE(store_data_hi.end)
                if relu:
                    VMAXPS(ymm_data_hi, ymm_zero, ymm_data_hi)
                if store_mask:
                    VMASKMOVPS([reg_t8], ymm_store_mask, ymm_data_hi)
                else:
                    VMOVUPS([reg_t8], ymm_data_hi)
                if i + 1 != 8:
                    ADD(reg_t8, reg_t_stride)
