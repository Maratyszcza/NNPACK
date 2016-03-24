import fft.real_to_complex_soa_perm

arg_t = Argument(ptr(const_float_), name="t")
arg_f = Argument(ptr(float_), name="f")


with Function("nnp_fft8_8real__fma3",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_data = [YMMRegister() for _ in range(8)]

    for i, ymm_i in enumerate(ymm_data):
        VMOVUPS(ymm_i, [reg_t + i * YMMRegister.size])

    fft.real_to_complex_soa_perm.fft8_across_rows(ymm_data)

    for i, ymm_i in enumerate(ymm_data):
        VMOVUPS([reg_f + i * YMMRegister.size], ymm_i)

    RETURN()


import fft16x16


with Function("nnp_fft16_8real__fma3",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3):

    reg_t0 = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t0, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    reg_stride = GeneralPurposeRegister64()
    MOV(reg_stride, YMMRegister.size)

    reg_t8 = GeneralPurposeRegister64()
    LEA(reg_t8, [reg_t0 + 8 * YMMRegister.size])

    fft16x16.forward_vfft(reg_t0, reg_t8, reg_stride,
        data_out=[yword[reg_f + YMMRegister.size * i] for i in range(16)])

    RETURN()
