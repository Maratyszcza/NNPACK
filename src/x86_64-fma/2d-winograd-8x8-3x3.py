from __future__ import absolute_import
from __future__ import division

import winograd.o6x6k3x3
import block8x8
from common import _MM_SHUFFLE


for post_operation in ["store", "stream"]:
    arg_d_pointer = Argument(ptr(const_float_), name="d_pointer")
    arg_wd_pointer = Argument(ptr(float_), name="wd_pointer")
    arg_d_stride = Argument(size_t, name="d_stride")
    arg_wd_stride = Argument(size_t, name="wd_stride")
    arg_row_count = Argument(uint32_t, name="row_count")
    arg_column_count = Argument(uint32_t, name="column_count")
    arg_row_offset = Argument(uint32_t, name="row_offset")
    arg_column_offset = Argument(uint32_t, name="column_offset")
    with Function("nnp_iwt8x8_3x3_with_offset_and_{post_operation}__avx2".format(post_operation=post_operation),
        (arg_d_pointer, arg_wd_pointer, arg_d_stride, arg_wd_stride, arg_row_count, arg_column_count, arg_row_offset, arg_column_offset),
        target=uarch.default + isa.fma3 + isa.avx2):

        reg_d = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_d, arg_d_pointer)

        reg_wd = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_wd, arg_wd_pointer)

        reg_stride_d = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_stride_d, arg_d_stride)

        reg_stride_wd = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_stride_wd, arg_wd_stride)

        reg_row_cnt = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_cnt, arg_row_count)

        reg_col_cnt = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_col_cnt, arg_column_count)

        reg_row_off = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_off, arg_row_offset)

        reg_col_off = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_col_off, arg_column_offset)

        ymm_data = [YMMRegister() for _ in range(8)]

        block8x8.load_with_padding(ymm_data, reg_d, reg_stride_d, reg_row_off, reg_row_cnt, reg_col_off, reg_col_cnt)

        ymm_data = winograd.o6x6k3x3.input_transform(ymm_data)
        winograd.o6x6k3x3.transpose8x8(ymm_data)
        ymm_data = winograd.o6x6k3x3.input_transform(ymm_data)

        VSTOREPS = {"store": VMOVAPS, "stream": VMOVNTPS}[post_operation]
        for ymm_row in ymm_data:
            VSTOREPS([reg_wd], ymm_row)
            if ymm_row is not ymm_data[-1]:
                ADD(reg_wd, reg_stride_wd)

        RETURN()


for reverse_kernel in [False, True]:
    for post_operation in ["store", "stream"]:
        arg_g_pointer = Argument(ptr(const_float_), name="d_pointer")
        arg_wg_pointer = Argument(ptr(float_), name="wd_pointer")
        arg_g_stride = Argument(size_t, name="d_stride")
        arg_wg_stride = Argument(size_t, name="wd_stride")
        arg_row_count = Argument(uint32_t, name="row_count")
        arg_column_count = Argument(uint32_t, name="column_count")
        arg_row_offset = Argument(uint32_t, name="row_offset")
        arg_column_offset = Argument(uint32_t, name="column_offset")

        kwt_arguments = (arg_g_pointer, arg_wg_pointer, arg_g_stride, arg_wg_stride, arg_row_count, arg_column_count, arg_row_offset, arg_column_offset)
        with Function("nnp_kwt8x8_3{reverse}x3{reverse}_and_{post_operation}__avx2".format(
                reverse="R" if reverse_kernel else "", post_operation=post_operation),
            kwt_arguments, target=uarch.default + isa.fma3 + isa.avx2):

            reg_g = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_g, arg_g_pointer)

            reg_wg = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_wg, arg_wg_pointer)

            reg_stride_g = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_stride_g, arg_g_stride)

            reg_stride_wg = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_stride_wg, arg_wg_stride)

            # stride is in elements; multiply by sizeof(float) to get stride in bytes
            SHL(reg_stride_g, 2)

            xmm_load_mask = XMMRegister()
            VMOVAPS(xmm_load_mask.as_ymm, Constant.float32x8(-0.0, -0.0, -0.0, +0.0, +0.0, +0.0, +0.0, +0.0))
            xmm_g = [XMMRegister() for _ in range(3)]
            for xmm in xmm_g:
                VMASKMOVPS(xmm, xmm_load_mask, [reg_g])
                if xmm is not xmm_g[-1]:
                    ADD(reg_g, reg_stride_g)

            if reverse_kernel:
                xmm_g = xmm_g[::-1]
            ymm_wg_rows = winograd.o6x6k3x3.kernel_transform([xmm.as_ymm for xmm in xmm_g], rescale_coefficients=False)
            ymm_g_rows = winograd.o6x6k3x3.transpose8x3([ymm.as_xmm for ymm in ymm_wg_rows])
            if reverse_kernel:
                ymm_g_rows = ymm_g_rows[::-1]
            ymm_wg_rows = winograd.o6x6k3x3.kernel_transform(ymm_g_rows, rescale_coefficients=False)

            rcp_9     = float.fromhex("0x1.C71C72p-4")
            rcp_81    = float.fromhex("0x1.948B10p-7")
            rcp_90    = float.fromhex("0x1.6C16C2p-7")
            rcp_180   = float.fromhex("0x1.6C16C2p-8")
            rcp_810   = float.fromhex("0x1.43A274p-10")
            rcp_1620  = float.fromhex("0x1.43A274p-11")
            rcp_8100  = float.fromhex("0x1.02E85Cp-13")
            rcp_16200 = float.fromhex("0x1.02E85Cp-14")
            rcp_32400 = float.fromhex("0x1.02E85Cp-15")

            ymm_edge_scale  = YMMRegister()
            VMOVAPS(ymm_edge_scale, Constant.float32x8( 1.0,           -2.0 * rcp_9,    -2.0 * rcp_9,          rcp_90,           rcp_90,          rcp_180,          rcp_180,    1.0))
            VMULPS(ymm_wg_rows[0], ymm_wg_rows[0], ymm_edge_scale)
            VMULPS(ymm_wg_rows[7], ymm_wg_rows[7], ymm_edge_scale)

            ymm_row12_scale = YMMRegister()
            VMOVAPS(ymm_row12_scale, Constant.float32x8(-2.0 * rcp_9,    4.0 * rcp_81,    4.0 * rcp_81,  -2.0 * rcp_810,   -2.0 * rcp_810,  -2.0 * rcp_1620,  -2.0 * rcp_1620,  -2.0 * rcp_9))
            VMULPS(ymm_wg_rows[1], ymm_wg_rows[1], ymm_row12_scale)
            VMULPS(ymm_wg_rows[2], ymm_wg_rows[2], ymm_row12_scale)

            ymm_row34_scale = YMMRegister()
            VMOVAPS(ymm_row34_scale, Constant.float32x8(       rcp_90,  -2.0 * rcp_810,  -2.0 * rcp_810,        rcp_8100,         rcp_8100,        rcp_16200,        rcp_16200,        rcp_90))
            VMULPS(ymm_wg_rows[3], ymm_wg_rows[3], ymm_row34_scale)
            VMULPS(ymm_wg_rows[4], ymm_wg_rows[4], ymm_row34_scale)

            ymm_row56_scale = YMMRegister()
            VMOVAPS(ymm_row56_scale, Constant.float32x8(       rcp_180, -2.0 * rcp_1620, -2.0 * rcp_1620,       rcp_16200,        rcp_16200,       rcp_32400,        rcp_32400,        rcp_180))
            VMULPS(ymm_wg_rows[5], ymm_wg_rows[5], ymm_row56_scale)
            VMULPS(ymm_wg_rows[6], ymm_wg_rows[6], ymm_row56_scale)

            # Write output with stride
            VSTOREPS = {"store": VMOVAPS, "stream": VMOVNTPS}[post_operation]
            for ymm_wg_row in ymm_wg_rows:
                VSTOREPS([reg_wg], ymm_wg_row)
                if ymm_wg_row is not ymm_wg_rows[-1]:
                    ADD(reg_wg, reg_stride_wg)

            RETURN()


arg_m_pointer = Argument(ptr(const_float_), name="m_pointer")
arg_s_pointer = Argument(ptr(float_), name="s_pointer")
arg_bias = Argument(ptr(const_float_), name="bias_pointer")
arg_m_stride = Argument(size_t, name="m_stride")
arg_s_stride = Argument(size_t, name="s_stride")
arg_row_count = Argument(uint32_t, name="row_count")
arg_column_count = Argument(uint32_t, name="column_count")
arg_row_offset = Argument(uint32_t, name="row_offset")
arg_column_offset = Argument(uint32_t, name="column_offset")
for with_offset, with_bias, with_relu in [(True, False, False), (False, True, False), (False, True, True)]:
    if with_bias:
        owt8x8_arguments = (arg_m_pointer, arg_s_pointer, arg_bias, arg_m_stride, arg_s_stride, arg_row_count, arg_column_count)
    else:
        owt8x8_arguments = (arg_m_pointer, arg_s_pointer, arg_m_stride, arg_s_stride, arg_row_count, arg_column_count)
    if with_offset:
        # Note: the version with offset has offset arguments, but they are never used (assumed 0).
        owt8x8_arguments += (arg_row_offset, arg_column_offset)
    with Function("nnp_owt8x8_3x3{with_bias}{with_relu}__avx2".format(
            with_bias="_with_bias" if with_bias else "",
            with_relu="_with_relu" if with_relu else ""),
        owt8x8_arguments, target=uarch.default + isa.fma3 + isa.avx2):

        reg_m = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_m, arg_m_pointer)

        reg_s = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_s, arg_s_pointer)

        if with_bias:
            reg_bias = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_bias, arg_bias)

            xmm_bias = XMMRegister()
            VINSERTPS(xmm_bias, xmm_bias, [reg_bias], 0b1101 | 1<<4)

        reg_m_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_m_stride, arg_m_stride)

        reg_s_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_s_stride, arg_s_stride)

        reg_row_count = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_count, arg_row_count)

        reg_column_count = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_column_count, arg_column_count)

        ymm_m = [YMMRegister() for _ in range(8)]
        for ymm in ymm_m:
            if with_bias and ymm is ymm_m[1]:
                VADDPS(ymm, xmm_bias.as_ymm, [reg_m])
            else:
                VMOVAPS(ymm, [reg_m])

            if ymm is not ymm_m[-1]:
                ADD(reg_m, reg_m_stride)

        ymm_t = winograd.o6x6k3x3.output_transform(ymm_m)

        ymm_tt = winograd.o6x6k3x3.transpose6x8(ymm_t)

        ymm_s = winograd.o6x6k3x3.output_transform(ymm_tt)

        block8x8.store_packed(ymm_s, reg_s, reg_s_stride, reg_row_count, reg_column_count, None, None, with_relu)

        RETURN()
