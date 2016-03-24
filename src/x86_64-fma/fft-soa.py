import fft.complex_soa

arg_t = Argument(ptr(const_float_), name="t")
arg_f = Argument(ptr(float_), name="f")


with Function("nnp_fft16_soa__avx2",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_real = YMMRegister(), YMMRegister()
    ymm_imag = YMMRegister(), YMMRegister()

    for i, ymm_data in enumerate(ymm_real + ymm_imag):
        VMOVUPS(ymm_data, [reg_t + i * YMMRegister.size])

    fft.complex_soa.fft16_within_rows(ymm_real, ymm_imag)

    for i, ymm_data in enumerate(ymm_real + ymm_imag):
        VMOVUPS([reg_f + i * YMMRegister.size], ymm_data)

    RETURN()


with Function("nnp_fft8_soa__avx2",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_real, ymm_imag = YMMRegister(), YMMRegister()

    VMOVUPS(ymm_real, [reg_t])
    VMOVUPS(ymm_imag, [reg_t + YMMRegister.size])

    fft.complex_soa.fft8_within_rows(ymm_real, ymm_imag)

    VMOVUPS([reg_f], ymm_real)
    VMOVUPS([reg_f + YMMRegister.size], ymm_imag)

    RETURN()


with Function("nnp_ifft8_soa__avx2",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_real, ymm_imag = YMMRegister(), YMMRegister()

    VMOVUPS(ymm_real, [reg_t])
    VMOVUPS(ymm_imag, [reg_t + YMMRegister.size])

    fft.complex_soa.fft8_within_rows(ymm_real, ymm_imag, transformation="inverse")

    VMOVUPS([reg_f], ymm_real)
    VMOVUPS([reg_f + YMMRegister.size], ymm_imag)

    RETURN()


with Function("nnp_ifft16_soa__avx2",
    (arg_f, arg_t),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    ymm_real = YMMRegister(), YMMRegister()
    ymm_imag = YMMRegister(), YMMRegister()

    for i, ymm_data in enumerate(ymm_real + ymm_imag):
        VMOVUPS(ymm_data, [reg_f + i * YMMRegister.size])

    fft.complex_soa.ifft16_within_rows(ymm_real, ymm_imag)

    for i, ymm_data in enumerate(ymm_real + ymm_imag):
        VMOVUPS([reg_t + i * YMMRegister.size], ymm_data)

    RETURN()
