from __future__ import absolute_import
from __future__ import division

from peachpy import *
from peachpy.x86_64 import *


def load_with_padding(ymm_data, reg_data, reg_stride, reg_row_offset, reg_row_count, reg_column_offset, reg_column_count):
    assert isinstance(ymm_data, list) and all(isinstance(ymm_row, YMMRegister) for ymm_row in ymm_data)
    assert isinstance(reg_data, GeneralPurposeRegister64)
    assert isinstance(reg_stride, GeneralPurposeRegister64)
    assert isinstance(reg_row_offset, GeneralPurposeRegister32)
    assert isinstance(reg_row_count, GeneralPurposeRegister32)
    assert isinstance(reg_column_offset, GeneralPurposeRegister32)
    assert isinstance(reg_column_count, GeneralPurposeRegister32)

    reg_column_end = GeneralPurposeRegister64()
    LEA(reg_column_end, [reg_column_offset.as_qword + reg_column_count.as_qword * 1])

    ymm_before_column_end_mask = YMMRegister()
    VMOVD(ymm_before_column_end_mask.as_xmm, reg_column_end.as_dword)
    ymm_before_column_start_mask = YMMRegister()
    VMOVD(ymm_before_column_start_mask.as_xmm, reg_column_offset.as_dword)

    ymm_column_index_mask = YMMRegister()
    VMOVAPD(ymm_column_index_mask, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))

    VPBROADCASTD(ymm_before_column_end_mask, ymm_before_column_end_mask.as_xmm)
    VPCMPGTD(ymm_before_column_end_mask, ymm_before_column_end_mask, ymm_column_index_mask)
    VPBROADCASTD(ymm_before_column_start_mask, ymm_before_column_start_mask.as_xmm)
    VPCMPGTD(ymm_before_column_start_mask, ymm_before_column_start_mask, ymm_column_index_mask)

    ymm_load_mask = YMMRegister()
    VPANDN(ymm_load_mask, ymm_before_column_start_mask, ymm_before_column_end_mask)

    # Multiply by sizeof(float) to get offset in bytes
    SHL(reg_column_offset, 2)
    # data points to the first element, which is loaded into lane `reg_column_offset`
    # However, VMASKMOVPS expects pointer to the first lane, even if it is not loaded.
    # Adjust the pointer by subtracting column_offset, in bytes
    SUB(reg_data, reg_column_offset.as_qword)

    # stride is in elements; multiply by sizeof(float) to get stride in bytes
    SHL(reg_stride, 2)

    # Zero all elements. Rows which are not loaded are initialized here.
    for ymm_row in ymm_data:
        VXORPS(ymm_row, ymm_row, ymm_row)

    with Block() as load_rows:
        for i, ymm_row in enumerate(ymm_data):

            with Block() as load_row:
                CMP(reg_row_offset, i)
                JA(load_row.end)

                VMASKMOVPS(ymm_row, ymm_load_mask, [reg_data])
                if i + 1 != len(ymm_data):
                    ADD(reg_data, reg_stride)

                    SUB(reg_row_count, 1)
                    JZ(load_rows.end)


def store_packed(ymm_data, reg_data, reg_stride, reg_row_count, reg_column_end, reg_row_offset=None, reg_column_start=None, relu=False):
    assert isinstance(ymm_data, list) and all(isinstance(ymm_row, YMMRegister) for ymm_row in ymm_data)
    assert isinstance(reg_data, GeneralPurposeRegister64)
    assert isinstance(reg_stride, GeneralPurposeRegister64)
    assert isinstance(reg_row_count, GeneralPurposeRegister32)
    assert isinstance(reg_column_end, GeneralPurposeRegister32)
    assert reg_row_offset is None or isinstance(reg_row_offset, GeneralPurposeRegister32)
    assert reg_column_start is None or isinstance(reg_column_start, GeneralPurposeRegister32)

    if reg_column_start is None:
        ymm_store_mask = YMMRegister()
        VMOVD(ymm_store_mask.as_xmm, reg_column_end)
        VPBROADCASTD(ymm_store_mask, ymm_store_mask.as_xmm)
        VPCMPGTD(ymm_store_mask, ymm_store_mask, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))
    else:
        ymm_before_column_end_mask = YMMRegister()
        VMOVD(ymm_before_column_end_mask.as_xmm, reg_column_end)
        ymm_before_column_start_mask = YMMRegister()
        VMOVD(ymm_before_column_start_mask.as_xmm, reg_column_start)

        SHL(reg_column_start, 2)
        SUB(reg_data, reg_column_start.as_qword)

        ymm_column_index_mask = YMMRegister()
        VMOVDQA(ymm_column_index_mask, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))

        VPBROADCASTD(ymm_before_column_end_mask, ymm_before_column_end_mask.as_xmm)
        VPCMPGTD(ymm_before_column_end_mask, ymm_before_column_end_mask, ymm_column_index_mask)
        VPBROADCASTD(ymm_before_column_start_mask, ymm_before_column_start_mask.as_xmm)
        VPCMPGTD(ymm_before_column_start_mask, ymm_before_column_start_mask, ymm_column_index_mask)

        ymm_store_mask = YMMRegister()
        VPANDN(ymm_store_mask, ymm_before_column_start_mask, ymm_before_column_end_mask)

    # stride is in elements; multiply by sizeof(float) to get stride in bytes
    SHL(reg_stride, 2)

    if relu:
        ymm_zero = YMMRegister()
        VMOVAPS(ymm_zero, Constant.float32x8(-0.0))

    with Block() as store_rows:
        for i, ymm_row in enumerate(ymm_data):
            with Block() as store_row:
                if reg_row_offset is not None:
                    CMP(reg_row_offset, i)
                    JA(store_row.end)

                if relu:
                    VMAXPS(ymm_row, ymm_zero, ymm_row)

                VMASKMOVPS([reg_data], ymm_store_mask, ymm_row)

                if ymm_row is not ymm_data[-1]:
                    ADD(reg_data, reg_stride)

                    SUB(reg_row_count, 1)
                    JZ(store_rows.end)
