from peachpy import *
from peachpy.x86_64 import *


def multiply_accumulate():
    VMOVAPS(ymm_acc, [mem_acc])
    VMOVAPS(ymm_x, [mem_x])
    VMOVSLDUP(ymm_yr, [mem_y])
    VMOVSHDUP(ymm_yi, [mem_y])

    # acc.re += x.re * y.re - x.im * y.im
    # acc.im += x.im * y.re + x.re * y.im

    VFMADD231PS(ymm_acc, ymm_x, ymm_yr)
    VPERMILPS(ymm_x, ymm_x, _MM_SHUFFLE(2, 3, 0, 1))
    VFMADDSUB231PS(ymm_acc, ymm_x, ymm_yi)

    VMOVAPS([mem_acc], ymm_acc)


def multiply_accumuate_conj():
    VMOVAPS(ymm_acc, [mem_acc])
    VMOVAPS(ymm_x, [mem_x])
    VMOVSLDUP(ymm_yr, [mem_y])
    VMOVSHDUP(ymm_yi, [mem_y])

    # acc.re += x.re * y.re + x.im * y.im
    # acc.im += x.im * y.re - x.re * y.im

    VFMADD231PS(ymm_acc, ymm_x, ymm_yr)
    VPERMILPS(ymm_x, ymm_x, _MM_SHUFFLE(2, 3, 0, 1))
    VFMSUBADD231PS(ymm_acc, ymm_x, ymm_yi)

    VMOVAPS([mem_acc], ymm_acc)


def multiply_accumulate_complex_aos_perm(ymm_acc, ymm_x, ymm_y, first_tuple=False, conjugate_y=False, overwrite=False):
    ymm_accr, ymm_acci = ymm_acc
    ymm_xr, ymm_xi = ymm_x
    ymm_yr, ymm_yi = ymm_y

    VFMADD231PS(ymm_accr, ymm_xr, ymm_yr)
    if first_tuple:
        # First row: the first two elements are real numbers

        # Don't be fooled: elements 0-1 are all real numbers,
        # not imag components of complex numbers.
        # Compute acc.im += x.im * y.im for elements 0-1.
        # y.re is not used after this snippet. Use it for the output
        if overwrite:
            ymm_yiirrrrrr = ymm_yr
        else:
            ymm_yiirrrrrr = YMMRegister()
        VBLENDPS(ymm_yiirrrrrr, ymm_yr, ymm_yi, 0b00000011)
        VFMADD231PS(ymm_acci, ymm_xi, ymm_yiirrrrrr)

        # Overwrite ymm_xi (instead of ymm_accr), then copy elements 2-7 to ymm_accr
        if not overwrite:
            ymm_xi_copy = YMMRegister()
            VMOVAPS(ymm_xi_copy, ymm_xi)
        if not conjugate_y:
            VFNMADD132PS(ymm_xi, ymm_accr, ymm_yi)
        else:
            VFMADD132PS(ymm_xi, ymm_accr, ymm_yi)
        VBLENDPS(ymm_accr, ymm_accr, ymm_xi, 0b11111100)

        # Overwrite ymm_xr (instead of ymm_acci), then copy elements 2-7 to ymm_acci
        if not overwrite:
            ymm_xr_copy = YMMRegister()
            VMOVAPS(ymm_xr_copy, ymm_xr)
        if not conjugate_y:
            VFMADD132PS(ymm_xr, ymm_acci, ymm_yi)
        else:
            VFNMADD132PS(ymm_xr, ymm_acci, ymm_yi)
        VBLENDPS(ymm_acci, ymm_acci, ymm_xr, 0b11111100)

        if not overwrite:
            # Restore xi and xr from copy
            SWAP.REGISTERS(ymm_xi, ymm_xi_copy)
            SWAP.REGISTERS(ymm_xr, ymm_xr_copy)
    else:
        VFMADD231PS(ymm_acci, ymm_xi, ymm_yr)
        if not conjugate_y:
            VFNMADD231PS(ymm_accr, ymm_xi, ymm_yi)
            VFMADD231PS(ymm_acci, ymm_xr, ymm_yi)
        else:
            VFMADD231PS(ymm_accr, ymm_xi, ymm_yi)
            VFNMADD231PS(ymm_acci, ymm_xr, ymm_yi)

