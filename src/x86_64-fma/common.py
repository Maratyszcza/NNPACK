from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *

sqrt2_over_2 = float.fromhex("0x1.6A09E6p-1")

cos_1pi_over_8 = float.fromhex("0x1.D906BCp-1")
cos_3pi_over_8 = float.fromhex("0x1.87DE2Ap-2")
tan_1pi_over_8 = float.fromhex("0x1.A8279Ap-2")
tan_3pi_over_8 = float.fromhex("0x1.3504F4p+1")

cos_npi_over_8 = [
    1.0,
    cos_1pi_over_8,
    sqrt2_over_2,
    cos_3pi_over_8,
    0.0,
    -cos_3pi_over_8,
    -sqrt2_over_2,
    -cos_1pi_over_8,
]

cos_npi_over_4 = [
    1.0,
    sqrt2_over_2,
    0.0,
    -sqrt2_over_2
]

sin_npi_over_8 = [
    0.0,
    cos_3pi_over_8,
    sqrt2_over_2,
    cos_1pi_over_8,
    1.0,
    cos_1pi_over_8,
    sqrt2_over_2,
    cos_3pi_over_8
]

sin_npi_over_4 = [
    0.0,
    sqrt2_over_2,
    1.0,
    sqrt2_over_2
]


def _MM_SHUFFLE(z, y, x, w):
    assert z & ~0b11 == 0
    assert y & ~0b11 == 0
    assert x & ~0b11 == 0
    assert w & ~0b11 == 0
    return (z << 6) | (y << 4) | (x << 2) | w


def _MM_SHUFFLE2(x, y):
    assert x & ~1 == 0
    assert y & ~1 == 0
    return (x << 1) | y


def butterfly(a, b, negate_a=False, negate_b=False, scale_a=None, scale_b=None, negate_out_b=False, writeback=True):
    assert isinstance(a, YMMRegister) or isinstance(a, LocalVariable) and a.size == YMMRegister.size
    assert isinstance(b, YMMRegister) or isinstance(b, LocalVariable) and b.size == YMMRegister.size
    assert isinstance(negate_a, bool)
    assert isinstance(negate_b, bool)
    assert isinstance(negate_out_b, bool)
    assert scale_b is None or \
        isinstance(scale_b, YMMRegister) or \
        isinstance(scale_b, (LocalVariable, Constant)) and scale_b.size == YMMRegister.size
    assert scale_a is None or \
        isinstance(scale_a, YMMRegister) or \
        isinstance(scale_a, (LocalVariable, Constant)) and scale_a.size == YMMRegister.size
    assert scale_a is None or scale_b is None
    assert isinstance(writeback, bool)

    assert not negate_out_b or not negate_a and not negate_b and scale_a is None and scale_b is None

    ymm_a, ymm_b = a, b
    if isinstance(a, LocalVariable):
        ymm_a = YMMRegister()
        VMOVAPS(ymm_a, a)
    if isinstance(b, LocalVariable):
        ymm_b = YMMRegister()
        VMOVAPS(ymm_b, b)

    if scale_b is None and scale_a is None:
        assert not negate_a, "Negation of a is supported only in combination with scaling"

        ymm_new_a = YMMRegister()
        VADDPS(ymm_new_a, ymm_a, ymm_b)

        ymm_new_b = YMMRegister()
        if not negate_out_b:
            VSUBPS(ymm_new_b, ymm_a, ymm_b)
        else:
            VSUBPS(ymm_new_b, ymm_b, ymm_a)

        if not negate_b:
            SWAP.REGISTERS(ymm_new_a, ymm_a)
            SWAP.REGISTERS(ymm_new_b, ymm_b)
        else:
            SWAP.REGISTERS(ymm_new_a, ymm_b)
            SWAP.REGISTERS(ymm_new_b, ymm_a)
    elif scale_a is not None:
        ymm_a_copy = YMMRegister()
        VMOVAPS(ymm_a_copy, ymm_a)

        if not negate_a and not negate_b:
            VFMADD132PS(ymm_a, ymm_b, scale_a)
            VFMSUB132PS(ymm_a_copy, ymm_b, scale_a)
        elif not negate_a and negate_b:
            VFMSUB132PS(ymm_a, ymm_b, scale_a)
            VFMADD132PS(ymm_a_copy, ymm_b, scale_a)
        elif negate_a and not negate_b:
            VFMMADD132PS(ymm_a, ymm_b, scale_a)
            VFNMSUB132PS(ymm_a_copy, ymm_b, scale_a)
        elif negate_a and negate_b:
            VFNMSUB132PS(ymm_a, ymm_b, scale_a)
            VFNMADD132PS(ymm_a_copy, ymm_b, scale_a)

        SWAP.REGISTERS(ymm_b, ymm_a_copy)
    elif scale_b is not None:
        ymm_a_copy = YMMRegister()
        VMOVAPS(ymm_a_copy, ymm_a)

        if not negate_a and not negate_b:
            VFMADD231PS(ymm_a, ymm_b, scale_b)
            VFNMADD231PS(ymm_a_copy, ymm_b, scale_b)
        elif not negate_a and negate_b:
            VFNMADD231PS(ymm_a, ymm_b, scale_b)
            VFMADD231PS(ymm_a_copy, ymm_b, scale_b)
        elif negate_a and not negate_b:
            VFMSUB231PS(ymm_a, ymm_b, scale_b)
            VFNMSUB231PS(ymm_a_copy, ymm_b, scale_b)
        elif negate_a and negate_b:
            VFNMSUB231PS(ymm_a, ymm_b, scale_b)
            VFMSUB231PS(ymm_a_copy, ymm_b, scale_b)

        SWAP.REGISTERS(ymm_b, ymm_a_copy)

    if writeback and isinstance(a, LocalVariable):
        VMOVAPS(a, ymm_a)
    if writeback and isinstance(b, LocalVariable):
        VMOVAPS(b, ymm_b)

    return ymm_a, ymm_b

def transpose2x2x128(ymm_a, ymm_b, use_blend=True):
    # ymm_a      = (a.lo, a.hi)
    # ymm_b      = (b.lo, b.hi)
    if use_blend:
        # ymm_ab = (a.hi, b.lo)
        ymm_ab = YMMRegister()
        VPERM2F128(ymm_ab, ymm_a, ymm_b, 0x21)

        # ymm_a  = (a.lo, b.lo)
        VBLENDPS(ymm_a, ymm_a, ymm_ab, 0xF0)

        # ymm_b  = (a.hi, b.hi)
        VBLENDPS(ymm_b, ymm_b, ymm_ab, 0x0F)
    else:
        # ymm_new_a = (a.lo, b.lo)
        ymm_new_a = YMMRegister()
        VINSERTF128(ymm_new_a, ymm_a, ymm_b.as_xmm, 1)

        # ymm_new_b = (a.hi, b.hi)
        ymm_new_b = YMMRegister()
        VPERM2F128(ymm_new_b, ymm_a, ymm_b, 0x31)

        SWAP.REGISTERS(ymm_a, ymm_new_a)
        SWAP.REGISTERS(ymm_b, ymm_new_b)


def transpose2x2x2x64(ymm_a, ymm_b, use_blend=True):
    # ymm_a      = (a0, a1, a2, a3)
    # ymm_b      = (b0, b1, a2, b3)
    if use_blend:
        # ymm_ab = (a1, b0, a3, b2)
        ymm_ab = YMMRegister()
        VSHUFPD(ymm_ab, ymm_a, ymm_b, 0b0101)

        # ymm_a  = (a0, b0, a2, b2)
        VBLENDPS(ymm_a, ymm_a, ymm_ab, 0b11001100)

        # ymm_b  = (a1, b1, a3, b3)
        VBLENDPS(ymm_b, ymm_b, ymm_ab, 0b00110011)
    else:
        # ymm_new_a = (a0, b0, a2, b2)
        ymm_new_a = YMMRegister()
        VUNPCKLPD(ymm_new_a, ymm_a, ymm_b)

        # ymm_new_b = (a1, b1, a3, b3)
        ymm_new_b = YMMRegister()
        VUNPCKHPD(ymm_new_b, ymm_a, ymm_b)

        SWAP.REGISTERS(ymm_a, ymm_new_a)
        SWAP.REGISTERS(ymm_b, ymm_new_b)


def compute_masks(masks, reg_column_offset, reg_column_count):
    assert isinstance(masks, list) and all(isinstance(mask, (YMMRegister, LocalVariable)) for mask in masks)
    assert isinstance(reg_column_offset, GeneralPurposeRegister64)
    assert isinstance(reg_column_count, GeneralPurposeRegister64)

    
def interleave(sequence_a, sequence_b):
    assert isinstance(sequence_a, list) and isinstance(sequence_b, list) or isinstance(sequence_a, tuple) and isinstance(sequence_b, tuple)

    if isinstance(sequence_a, list):
        return list(sum(zip(sequence_a, sequence_b), ()))
    else:
        return sum(zip(sequence_a, sequence_b), ())
