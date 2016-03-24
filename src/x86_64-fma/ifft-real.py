import fft.complex_soa_perm_to_real
from common import butterfly, cos_npi_over_8, sqrt2_over_2


def fft8_bitreverse(n):
    return int(format(n, "03b")[::-1], 2)


arg_f = Argument(ptr(const_float_), name="f")
arg_t = Argument(ptr(float_), name="t")


with Function("nnp_ifft8_8real__fma3",
    (arg_f, arg_t),
    target=uarch.default + isa.fma3):

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    ymm_data = [YMMRegister() for _ in range(8)]
    ymm_real, ymm_imag = ymm_data[0::2], ymm_data[1::2]

    for i, ymm_i in enumerate(ymm_data):
        VMOVUPS(ymm_i, [reg_f + i * YMMRegister.size])

    fft.complex_soa_perm_to_real.ifft8_across_rows(ymm_data)

    for i, ymm_i in enumerate(ymm_data):
        VMOVUPS([reg_t + i * YMMRegister.size], ymm_i)

    RETURN()


import fft16x16


with Function("nnp_ifft16_8real__fma3",
    (arg_f, arg_t),
    target=uarch.default + isa.fma3):

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    reg_t0 = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t0, arg_t)

    reg_stride = GeneralPurposeRegister64()
    MOV(reg_stride, YMMRegister.size)

    reg_t8 = GeneralPurposeRegister64()
    LEA(reg_t8, [reg_t0 + 8 * YMMRegister.size])

    fft16x16.inverse_vfft(reg_t0, reg_t8, reg_stride,
        data_in=[yword[reg_f + YMMRegister.size * i] for i in range(16)])

    RETURN()
