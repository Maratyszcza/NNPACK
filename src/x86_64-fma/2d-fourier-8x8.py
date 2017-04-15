from __future__ import absolute_import
from __future__ import division

import block8x8
import fft.complex_soa
import fft.real_to_complex_soa_perm
import fft.complex_soa_perm_to_real
import fft.two_real_to_two_complex_soa_perm_planar
import fft.two_complex_soa_perm_to_two_real_planar


arg_t_pointer = Argument(ptr(const_float_), name="t_pointer")
arg_f_pointer = Argument(ptr(float_), name="f_pointer")
arg_x_pointer = Argument(ptr(const_float_), name="x_pointer")
arg_t_stride = Argument(size_t, name="t_stride")
arg_f_stride = Argument(size_t, name="f_stride")
arg_row_offset = Argument(uint32_t, name="row_offset")
arg_row_count = Argument(uint32_t, name="row_count")
arg_column_offset = Argument(uint32_t, name="column_offset")
arg_column_count = Argument(uint32_t, name="column_count")
for post_operation in ["stream", "store"]:
    fft8x8_arguments = (arg_t_pointer, arg_f_pointer, arg_t_stride, arg_f_stride, arg_row_count, arg_column_count, arg_row_offset, arg_column_offset)
    with Function("nnp_fft8x8_with_offset_and_{post_operation}__avx2".format(post_operation=post_operation),
        fft8x8_arguments, target=uarch.default + isa.fma3 + isa.avx2):

        reg_t = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t, arg_t_pointer)

        reg_f = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f, arg_f_pointer)

        reg_inct = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_inct, arg_t_stride)

        reg_incf = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_incf, arg_f_stride)

        reg_row_cnt = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_cnt, arg_row_count)

        reg_col_cnt = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_col_cnt, arg_column_count)

        reg_row_off = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_off, arg_row_offset)

        reg_col_off = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_col_off, arg_column_offset)

        ymm_data = [YMMRegister(i) for i in range(8)]
        ymm_real, ymm_imag = ymm_data[0::2], ymm_data[1::2]

        block8x8.load_with_padding(ymm_data, reg_t, reg_inct, reg_row_off, reg_row_cnt, reg_col_off, reg_col_cnt)

        fft.real_to_complex_soa_perm.fft8_across_rows(ymm_data)
        fft.complex_soa.fft8_within_rows(ymm_real, ymm_imag)
        fft.two_real_to_two_complex_soa_perm_planar.fft8_within_rows_postprocess(ymm_real[0], ymm_imag[0])

        VSTOREPS = {"store": VMOVAPS, "stream": VMOVNTPS}[post_operation]
        for ymm_re, ymm_im in zip(ymm_real, ymm_imag):
            VSTOREPS([reg_f], ymm_re)
            VSTOREPS([reg_f + YMMRegister.size], ymm_im)
            if ymm_re is not ymm_real[-1]:
                ADD(reg_f, reg_incf)

        RETURN()


arg_f_pointer = Argument(ptr(const_float_), name="f_pointer")
arg_t_pointer = Argument(ptr(float_), name="t_pointer")
arg_bias = Argument(ptr(const_float_), name="bias_pointer")
arg_f_stride = Argument(size_t, name="f_stride")
arg_t_stride = Argument(size_t, name="t_stride")
arg_row_offset = Argument(uint32_t, name="row_offset")
arg_row_count = Argument(uint32_t, name="row_count")
arg_column_offset = Argument(uint32_t, name="column_offset")
arg_column_count = Argument(uint32_t, name="column_count")
for with_offset, with_bias, with_relu in [(True, False, False), (False, True, False), (False, True, True)]:
    if with_bias:
        ifft8x8_arguments = (arg_f_pointer, arg_t_pointer, arg_bias, arg_f_stride, arg_t_stride, arg_row_count, arg_column_count)
    else:
        ifft8x8_arguments = (arg_f_pointer, arg_t_pointer, arg_f_stride, arg_t_stride, arg_row_count, arg_column_count)
    if with_offset:
        ifft8x8_arguments += arg_row_offset, arg_column_offset
    with Function("nnp_ifft8x8{with_offset}{with_bias}{with_relu}__avx2".format(
            with_offset="_with_offset" if with_offset else "",
            with_bias="_with_bias" if with_bias else "",
            with_relu="_with_relu" if with_relu else ""),
        ifft8x8_arguments,
        target=uarch.default + isa.fma3 + isa.avx2):

        reg_f = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f, arg_f_pointer)

        reg_t = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t, arg_t_pointer)

        if with_bias:
            reg_bias = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_bias, arg_bias)

        reg_f_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f_stride, arg_f_stride)

        reg_t_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t_stride, arg_t_stride)

        reg_row_count = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_count, arg_row_count)

        reg_column_end = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_column_end, arg_column_count)

        if with_offset:
            reg_row_start = GeneralPurposeRegister32()
            LOAD.ARGUMENT(reg_row_start, arg_row_offset)

            reg_column_start = GeneralPurposeRegister32()
            LOAD.ARGUMENT(reg_column_start, arg_column_offset)
            ADD(reg_column_end, reg_column_start)
        else:
            reg_row_start = None
            reg_column_start = None

        ymm_data = [YMMRegister(i) for i in range(8)]
        ymm_real, ymm_imag = ymm_data[0::2], ymm_data[1::2]

        if with_bias:
            ymm_bias = YMMRegister()
            VMOVSS(ymm_bias.as_xmm, [reg_bias])

        for ymm_re, ymm_im in zip(ymm_real, ymm_imag):
            VMOVAPS(ymm_re, [reg_f])
            VMOVAPS(ymm_im, [reg_f + YMMRegister.size])
            if with_bias and ymm_re is ymm_real[0]:
                VFMADD231PS(ymm_re, ymm_bias, Constant.float32x8(64.0))

            if ymm_im is not ymm_imag[-1]:
                ADD(reg_f, reg_f_stride)

        fft.two_complex_soa_perm_to_two_real_planar.ifft8_within_rows_preprocess(ymm_real[0], ymm_imag[0])
        fft.complex_soa.fft8_within_rows(ymm_real, ymm_imag, transformation="inverse")
        fft.complex_soa_perm_to_real.ifft8_across_rows(ymm_data)

        block8x8.store_packed(ymm_data, reg_t, reg_t_stride, reg_row_count, reg_column_end, reg_row_start, reg_column_start, with_relu)

        RETURN()
