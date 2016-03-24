import fft.complex_soa
import fft.two_real_to_two_complex_soa_perm_planar


arg_t = Argument(ptr(const_float_), name="t")
arg_f = Argument(ptr(float_), name="f")


with Function("nnp_fft8_dualreal__avx2",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_seq_a, ymm_seq_b = YMMRegister(), YMMRegister()

    VMOVUPS(ymm_seq_a, [reg_t])
    VMOVUPS(ymm_seq_b, [reg_t + YMMRegister.size])

    fft.complex_soa.fft8_within_rows(ymm_seq_a, ymm_seq_b)
    ymm_wr, ymm_wi = ymm_seq_a, ymm_seq_b

    fft.two_real_to_two_complex_soa_perm_planar.fft8_within_rows_postprocess(ymm_wr, ymm_wi)
    ymm_xhr, ymm_xhi = ymm_wr, ymm_wi

    VMOVUPS([reg_f], ymm_xhr)
    VMOVUPS([reg_f + YMMRegister.size], ymm_xhi)

    RETURN()


with Function("nnp_fft16_dualreal__avx2",
    (arg_t, arg_f),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_t = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_t, arg_t)

    reg_f = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_f, arg_f)

    ymm_seq_a = YMMRegister(), YMMRegister()
    ymm_seq_b = YMMRegister(), YMMRegister()
    for i, ymm_a in enumerate(ymm_seq_a + ymm_seq_b):
        VMOVUPS(ymm_a, [reg_t + i * YMMRegister.size])

    fft.complex_soa.fft16_within_rows(ymm_seq_a, ymm_seq_b)
    ymm_wr, ymm_wi = ymm_seq_a, ymm_seq_b

    fft.two_real_to_two_complex_soa_perm_planar.fft16_within_rows_postprocess(ymm_wr, ymm_wi)

    for i, ymm_w in enumerate(ymm_wr + ymm_wi):
        VMOVUPS([reg_f + i * YMMRegister.size], ymm_w)

    RETURN()
