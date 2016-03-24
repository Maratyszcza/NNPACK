import fft.complex_soa
import fft.two_complex_soa_perm_to_two_real_planar


arg_f = Argument(ptr(const_float_), name="f")
arg_t = Argument(ptr(float_), name="t")


with Function("nnp_ifft8_dualreal__avx2",
    (arg_f, arg_t),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    ymm_xhr, ymm_xhi = YMMRegister(), YMMRegister()
    VMOVUPS(ymm_xhr, [reg_f])
    VMOVUPS(ymm_xhi, [reg_f + YMMRegister.size])

    fft.two_complex_soa_perm_to_two_real_planar.ifft8_within_rows_preprocess(ymm_xhr, ymm_xhi)
    ymm_wr, ymm_wi = ymm_xhr, ymm_xhi

    fft.complex_soa.fft8_within_rows(ymm_wr, ymm_wi, transformation="inverse")
    ymm_seq_a, ymm_seq_b = ymm_wr, ymm_wi

    VMOVUPS([reg_t], ymm_seq_a)
    VMOVUPS([reg_t + YMMRegister.size], ymm_seq_b)

    RETURN()


with Function("nnp_ifft16_dualreal__avx2",
    (arg_f, arg_t),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    ymm_wr = YMMRegister(), YMMRegister()
    ymm_wi = YMMRegister(), YMMRegister()

    for i, ymm_w in enumerate(ymm_wr + ymm_wi):
        VMOVUPS(ymm_w, [reg_f + i * YMMRegister.size])

    fft.two_complex_soa_perm_to_two_real_planar.ifft16_within_rows_preprocess(ymm_wr, ymm_wi)

    fft.complex_soa.ifft16_within_rows(ymm_wr, ymm_wi)

    for i, ymm_w in enumerate(ymm_wr + ymm_wi):
        VMOVUPS([reg_t + i * YMMRegister.size], ymm_w)

    RETURN()
