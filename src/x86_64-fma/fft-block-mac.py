from __future__ import absolute_import
from __future__ import division

from blockmac import multiply_accumulate_complex_aos_perm

arg_acc = Argument(ptr(float_), name="acc")
arg_x = Argument(ptr(const_float_), name="x")
arg_y = Argument(ptr(const_float_), name="y")


with Function("nnp_s8x8gemm__fma3",
    (arg_acc, arg_x, arg_y),
    target=uarch.default + isa.fma3):

    reg_acc = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_acc, arg_acc)

    reg_x = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_x, arg_x)

    reg_y = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_y, arg_y)

    for i in range(8):
        ymm_acc, ymm_x, ymm_y = YMMRegister(), YMMRegister(), YMMRegister()
        VMOVAPS(ymm_acc, [reg_acc])
        VMOVAPS(ymm_x, [reg_x])
        ADD(reg_x, YMMRegister.size)
        VFMADD231PS(ymm_acc, ymm_x, [reg_y])
        ADD(reg_y, YMMRegister.size)
        VMOVAPS([reg_acc], ymm_acc)
        ADD(reg_acc, YMMRegister.size)

    RETURN()


with Function("nnp_ft8x8gemmc__fma3",
    (arg_acc, arg_x, arg_y),
    target=uarch.default + isa.fma3):

    reg_acc = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_acc, arg_acc)

    reg_x = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_x, arg_x)

    reg_y = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_y, arg_y)

    for i in range(4):
        ymm_accr, ymm_acci = YMMRegister(), YMMRegister()
        VMOVAPS(ymm_accr, [reg_acc])
        VMOVAPS(ymm_acci, [reg_acc + YMMRegister.size])

        ymm_xr, ymm_xi = YMMRegister(), YMMRegister()
        VMOVAPS(ymm_xr, [reg_x])
        VMOVAPS(ymm_xi, [reg_x + YMMRegister.size])
        ADD(reg_x, YMMRegister.size * 2)

        ymm_yr, ymm_yi = YMMRegister(), YMMRegister()
        VMOVAPS(ymm_yr, [reg_y])
        VMOVAPS(ymm_yi, [reg_y + YMMRegister.size])
        ADD(reg_y, YMMRegister.size * 2)

        multiply_accumulate_complex_aos_perm(
            (ymm_accr, ymm_acci), (ymm_xr, ymm_xi), (ymm_yr, ymm_yi),
            first_tuple=i==0,
            conjugate_y=True)

        VMOVAPS([reg_acc], ymm_accr)
        VMOVAPS([reg_acc + YMMRegister.size], ymm_acci)
        ADD(reg_acc, YMMRegister.size * 2)

    RETURN()


with Function("nnp_ft16x16gemmc__fma3",
    (arg_acc, arg_x, arg_y),
    target=uarch.default + isa.fma3):

    reg_acc = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_acc, arg_acc)

    reg_x = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_x, arg_x)

    reg_y = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_y, arg_y)

    for i in range(8):
        for j in range(2):
            ymm_accr, ymm_acci = YMMRegister(), YMMRegister()
            VMOVAPS(ymm_accr, [reg_acc + (j * 2) * YMMRegister.size])
            VMOVAPS(ymm_acci, [reg_acc + (j * 2 + 1) * YMMRegister.size])

            ymm_xr, ymm_xi = YMMRegister(), YMMRegister()
            VMOVAPS(ymm_xr, [reg_x + (j * 2) * YMMRegister.size])
            VMOVAPS(ymm_xi, [reg_x + (j * 2 + 1) * YMMRegister.size])

            ymm_yr, ymm_yi = YMMRegister(), YMMRegister()
            VMOVAPS(ymm_yr, [reg_y + (j * 2) * YMMRegister.size])
            VMOVAPS(ymm_yi, [reg_y + (j * 2 + 1) * YMMRegister.size])

            multiply_accumulate_complex_aos_perm(
                (ymm_accr, ymm_acci), (ymm_xr, ymm_xi), (ymm_yr, ymm_yi),
                first_tuple=(i==0 and j==0),
                conjugate_y=True)

            VMOVAPS([reg_acc + (j * 2) * YMMRegister.size], ymm_accr)
            VMOVAPS([reg_acc + (j * 2 + 1) * YMMRegister.size], ymm_acci)

        SUB(reg_x, -YMMRegister.size * 4)
        SUB(reg_y, -YMMRegister.size * 4)
        SUB(reg_acc, -YMMRegister.size * 4)

    RETURN()
