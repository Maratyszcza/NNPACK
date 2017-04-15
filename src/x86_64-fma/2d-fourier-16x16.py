from __future__ import absolute_import
from __future__ import division

import fft16x16
import fft.complex_soa
import fft.two_real_to_two_complex_soa_perm_planar
import fft.two_complex_soa_perm_to_two_real_planar


arg_t_pointer = Argument(ptr(const_float_), name="t")
arg_f_pointer = Argument(ptr(float_), name="f")
arg_x_pointer = Argument(ptr(const_float_), name="x")
arg_t_stride = Argument(size_t, name="stride_t")
arg_f_stride = Argument(size_t, name="stride_f")
arg_row_count = Argument(uint32_t, name="row_count")
arg_column_count = Argument(uint32_t, name="column_count")
arg_row_offset = Argument(uint32_t, name="row_offset")
arg_column_offset = Argument(uint32_t, name="column_offset")
for post_operation in ["stream", "store"]:
    fft16x16_arguments = (arg_t_pointer, arg_f_pointer, arg_t_stride, arg_f_stride, arg_row_count, arg_column_count, arg_row_offset, arg_column_offset)
    with Function("nnp_fft16x16_with_offset_and_{post_operation}__avx2".format(post_operation=post_operation),
        fft16x16_arguments, target=uarch.default + isa.fma3 + isa.avx2):

        reg_t0 = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t0, arg_t_pointer)

        reg_f = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f, arg_f_pointer)

        reg_t_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t_stride, arg_t_stride)

        reg_f_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f_stride, arg_f_stride)

        reg_row_end = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_end, arg_row_count)

        reg_column_end = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_column_end, arg_column_count)

        reg_row_start = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_start, arg_row_offset)
        ADD(reg_row_end, reg_row_start)

        reg_column_start = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_column_start, arg_column_offset)
        ADD(reg_column_end, reg_column_start)

        ymm_column_start, ymm_column_end = YMMRegister(), YMMRegister()
        VMOVD(ymm_column_start.as_xmm, reg_column_start.as_dword)
        VMOVD(ymm_column_end.as_xmm, reg_column_end.as_dword)
        VPBROADCASTD(ymm_column_start, ymm_column_start.as_xmm)
        VPBROADCASTD(ymm_column_end, ymm_column_end.as_xmm)

        ymm_column_01234567 = YMMRegister()
        VMOVDQA(ymm_column_01234567, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))
        ymm_column_start_gt_01234567, ymm_column_end_gt_01234567 = YMMRegister(), YMMRegister()
        VPCMPGTD(ymm_column_start_gt_01234567, ymm_column_start, ymm_column_01234567)
        VPCMPGTD(ymm_column_end_gt_01234567, ymm_column_end, ymm_column_01234567)

        ymm_column_89ABCDEF = YMMRegister()
        VMOVDQA(ymm_column_89ABCDEF, Constant.uint32x8(8, 9, 10, 11, 12, 13, 14, 15))
        ymm_column_start_gt_89ABCDEF, ymm_column_end_gt_89ABCDEF = YMMRegister(), YMMRegister()
        VPCMPGTD(ymm_column_start_gt_89ABCDEF, ymm_column_start, ymm_column_89ABCDEF)
        VPCMPGTD(ymm_column_end_gt_89ABCDEF, ymm_column_end, ymm_column_89ABCDEF)

        ymm_load_mask_columns_0_to_8 = YMMRegister()
        VPANDN(ymm_load_mask_columns_0_to_8, ymm_column_start_gt_01234567, ymm_column_end_gt_01234567)

        ymm_load_mask_columns_8_to_16 = YMMRegister()
        VPANDN(ymm_load_mask_columns_8_to_16, ymm_column_start_gt_89ABCDEF, ymm_column_end_gt_89ABCDEF)
        load_mask_columns_8_to_16 = LocalVariable(ymm_load_mask_columns_8_to_16)
        VMOVDQA(load_mask_columns_8_to_16, ymm_load_mask_columns_8_to_16)

        # data points to the first element, which is loaded into lane `reg_column_start`
        # However, VMASKMOVPS expects pointer to the first lane, even if it is not loaded.
        # Adjust the pointer by subtracting column_offset, in bytes
        SHL(reg_column_start, 2)
        SUB(reg_t0, reg_column_start.as_qword)

        # Multiply stride by sizeof(float) to convert from elements to bytes
        SHL(reg_t_stride, 2)

        # t8_offset = stride * (8 - row_start)
        reg_t8_offset = GeneralPurposeRegister64()
        MOV(reg_t8_offset.as_dword, 8)
        SUB(reg_t8_offset.as_dword, reg_row_start)
        IMUL(reg_t8_offset, reg_t_stride)
        reg_t8 = GeneralPurposeRegister64()
        LEA(reg_t8, [reg_t0 + reg_t8_offset * 1])
        CMP(reg_row_start, 8)
        CMOVAE(reg_t8, reg_t0)

        reg_t0_column_8, reg_t8_column_8 = GeneralPurposeRegister64(), GeneralPurposeRegister64()
        LEA(reg_t0_column_8, [reg_t0 + YMMRegister.size])
        LEA(reg_t8_column_8, [reg_t8 + YMMRegister.size])

        vfft_columns_0_to_8 = [LocalVariable(YMMRegister.size) for _ in range(16)]
        vfft_columns_8_to_16 = [YMMRegister() if i < 4 else LocalVariable(YMMRegister.size) for i in range(16)]

        fft16x16.forward_vfft(reg_t0, reg_t8, reg_t_stride, data_out=vfft_columns_0_to_8,
            reg_row_start=reg_row_start, reg_row_end=reg_row_end, ymm_load_mask=ymm_load_mask_columns_0_to_8)

        ymm_load_mask_columns_8_to_16 = YMMRegister()
        VMOVDQA(ymm_load_mask_columns_8_to_16, load_mask_columns_8_to_16)

        fft16x16.forward_vfft(reg_t0_column_8, reg_t8_column_8, reg_t_stride, data_out=vfft_columns_8_to_16,
            reg_row_start=reg_row_start, reg_row_end=reg_row_end, ymm_load_mask=ymm_load_mask_columns_8_to_16)

        for row_batch_start, row_batch_end in [(0, 2), (2, 5), (5, 8)]:
            ymm_wr_list = [(YMMRegister(), YMMRegister()) for _ in range(row_batch_start, row_batch_end)]
            ymm_wi_list = [(YMMRegister(), YMMRegister()) for _ in range(row_batch_start, row_batch_end)]
            for row_offset, (ymm_wr, ymm_wi) in enumerate(zip(ymm_wr_list, ymm_wi_list)):
                row = row_batch_start + row_offset

                VMOVAPS(ymm_wr[0], vfft_columns_0_to_8[row*2+0])
                VMOVAPS(ymm_wr[1], vfft_columns_8_to_16[row*2+0])
                VMOVAPS(ymm_wi[0], vfft_columns_0_to_8[row*2+1])
                VMOVAPS(ymm_wi[1], vfft_columns_8_to_16[row*2+1])

            fft.complex_soa.fft16_within_rows(ymm_wr_list, ymm_wi_list, bit_reversal=False)
            if row_batch_start == 0:
                fft.two_real_to_two_complex_soa_perm_planar.fft16_within_rows_postprocess(ymm_wr_list[0], ymm_wi_list[0], bit_reversal=True)

            VSTOREPS = {"store": VMOVAPS, "stream": VMOVNTPS}[post_operation]
            for row_batch_offset, (ymm_wr, ymm_wi) in enumerate(zip(ymm_wr_list, ymm_wi_list)):
                row = row_batch_start + row_batch_offset

                for column in range(2):
                    VSTOREPS([reg_f], ymm_wr[column])
                    VSTOREPS([reg_f + YMMRegister.size], ymm_wi[column])
                    if row + 1 != 8 or column + 1 != 2:
                        ADD(reg_f, reg_f_stride)

        RETURN()


arg_f_pointer = Argument(ptr(const_float_), name="f_pointer")
arg_t_pointer = Argument(ptr(float_), name="t_pointer")
arg_bias = Argument(ptr(const_float_), name="bias_pointer")
arg_f_stride = Argument(size_t, name="f_stride")
arg_t_stride = Argument(size_t, name="t_stride")
arg_row_count = Argument(uint32_t, name="row_count")
arg_column_count = Argument(uint32_t, name="column_count")
arg_row_offset = Argument(uint32_t, name="row_offset")
arg_column_offset = Argument(uint32_t, name="column_offset")
for with_offset, with_bias, with_relu in [(True, False, False), (False, True, False), (False, True, True)]:
    if with_bias:
        ifft16x16_arguments = (arg_f_pointer, arg_t_pointer, arg_bias, arg_f_stride, arg_t_stride, arg_row_count, arg_column_count)
    else:
        ifft16x16_arguments = (arg_f_pointer, arg_t_pointer, arg_f_stride, arg_t_stride, arg_row_count, arg_column_count)
    if with_offset:
        ifft16x16_arguments += (arg_row_offset, arg_column_offset)
    with Function("nnp_ifft16x16{with_offset}{with_bias}{with_relu}__avx2".format(
            with_offset="_with_offset" if with_offset else "",
            with_bias="_with_bias" if with_bias else "",
            with_relu="_with_relu" if with_relu else ""),
        ifft16x16_arguments, target=uarch.default + isa.fma3 + isa.avx2):

        reg_f = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f, arg_f_pointer)

        reg_t0 = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t0, arg_t_pointer)

        if with_bias:
            reg_bias = GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_bias, arg_bias)

        reg_f_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_f_stride, arg_f_stride)

        reg_t_stride = GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_t_stride, arg_t_stride)

        reg_row_end = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_row_end, arg_row_count)

        reg_column_end = GeneralPurposeRegister32()
        LOAD.ARGUMENT(reg_column_end, arg_column_count)

        if with_offset:
            reg_row_start = GeneralPurposeRegister32()
            LOAD.ARGUMENT(reg_row_start, arg_row_offset)
            ADD(reg_row_end, reg_row_start)

            reg_column_start = GeneralPurposeRegister32()
            LOAD.ARGUMENT(reg_column_start, arg_column_offset)
            ADD(reg_column_end, reg_column_start)
        else:
            reg_row_start = None

        if with_offset:
            ymm_column_start, ymm_column_end = YMMRegister(), YMMRegister()
            VMOVD(ymm_column_start.as_xmm, reg_column_start.as_dword)
            VMOVD(ymm_column_end.as_xmm, reg_column_end.as_dword)
            VPBROADCASTD(ymm_column_start, ymm_column_start.as_xmm)
            VPBROADCASTD(ymm_column_end, ymm_column_end.as_xmm)

            ymm_column_01234567 = YMMRegister()
            VMOVDQA(ymm_column_01234567, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))
            ymm_column_start_gt_01234567, ymm_column_end_gt_01234567 = YMMRegister(), YMMRegister()
            VPCMPGTD(ymm_column_start_gt_01234567, ymm_column_start, ymm_column_01234567)
            VPCMPGTD(ymm_column_end_gt_01234567, ymm_column_end, ymm_column_01234567)

            ymm_column_89ABCDEF = YMMRegister()
            VMOVDQA(ymm_column_89ABCDEF, Constant.uint32x8(8, 9, 10, 11, 12, 13, 14, 15))
            ymm_column_start_gt_89ABCDEF, ymm_column_end_gt_89ABCDEF = YMMRegister(), YMMRegister()
            VPCMPGTD(ymm_column_start_gt_89ABCDEF, ymm_column_start, ymm_column_89ABCDEF)
            VPCMPGTD(ymm_column_end_gt_89ABCDEF, ymm_column_end, ymm_column_89ABCDEF)

            ymm_store_mask_columns_0_to_8 = YMMRegister()
            VPANDN(ymm_store_mask_columns_0_to_8, ymm_column_start_gt_01234567, ymm_column_end_gt_01234567)
            store_mask_columns_0_to_8 = LocalVariable(ymm_store_mask_columns_0_to_8)
            VMOVDQA(store_mask_columns_0_to_8, ymm_store_mask_columns_0_to_8)

            ymm_store_mask_columns_8_to_16 = YMMRegister()
            VPANDN(ymm_store_mask_columns_8_to_16, ymm_column_start_gt_89ABCDEF, ymm_column_end_gt_89ABCDEF)
            store_mask_columns_8_to_16 = LocalVariable(ymm_store_mask_columns_8_to_16)
            VMOVDQA(store_mask_columns_8_to_16, ymm_store_mask_columns_8_to_16)

            SHL(reg_column_start, 2)
            SUB(reg_t0, reg_column_start.as_qword)
        else:
            ymm_column_end = YMMRegister()
            VMOVD(ymm_column_end.as_xmm, reg_column_end.as_dword)
            VPBROADCASTD(ymm_column_end, ymm_column_end.as_xmm)

            ymm_store_mask_columns_0_to_8, ymm_store_mask_columns_8_to_16 = YMMRegister(), YMMRegister()
            VPCMPGTD(ymm_store_mask_columns_0_to_8,  ymm_column_end, Constant.uint32x8(0, 1,  2,  3,  4,  5,  6,  7))
            VPCMPGTD(ymm_store_mask_columns_8_to_16, ymm_column_end, Constant.uint32x8(8, 9, 10, 11, 12, 13, 14, 15))

            store_mask_columns_0_to_8 = LocalVariable(ymm_store_mask_columns_0_to_8)
            VMOVDQA(store_mask_columns_0_to_8, ymm_store_mask_columns_0_to_8)
            store_mask_columns_8_to_16 = LocalVariable(ymm_store_mask_columns_8_to_16)
            VMOVDQA(store_mask_columns_8_to_16, ymm_store_mask_columns_8_to_16)

        # Multiply stride by sizeof(float) to convert from elements to bytes
        SHL(reg_t_stride, 2)

        vfft_columns_0_to_8 = [YMMRegister() if i > 10 else LocalVariable(YMMRegister.size) for i in range(16)]
        vfft_columns_8_to_16 = [LocalVariable(YMMRegister.size) for _ in range(16)]

        for row_batch_start, row_batch_end in [(0, 2), (2, 5), (5, 8)]:
            ymm_wr_list = [(YMMRegister(), YMMRegister()) for _ in range(row_batch_start, row_batch_end)]
            ymm_wi_list = [(YMMRegister(), YMMRegister()) for _ in range(row_batch_start, row_batch_end)]
            for row_offset, (ymm_wr, ymm_wi) in enumerate(zip(ymm_wr_list, ymm_wi_list)):
                row = row_batch_start + row_offset

                VMOVAPS(ymm_wr[0], [reg_f])
                VMOVAPS(ymm_wi[0], [reg_f + YMMRegister.size])
                ADD(reg_f, reg_f_stride)

                if with_bias and row == 0:
                    ymm_bias = YMMRegister()
                    VMOVSS(ymm_bias.as_xmm, [reg_bias])
                    VFMADD231PS(ymm_wr[0], ymm_bias, Constant.float32x8(256.0))

                VMOVAPS(ymm_wr[1], [reg_f])
                VMOVAPS(ymm_wi[1], [reg_f + YMMRegister.size])
                if row + 1 != 8:
                    ADD(reg_f, reg_f_stride)

            if row_batch_start == 0:
                fft.two_complex_soa_perm_to_two_real_planar.ifft16_within_rows_preprocess(ymm_wr_list[0], ymm_wi_list[0], bit_reversal=True)
            fft.complex_soa.ifft16_within_rows(ymm_wr_list, ymm_wi_list, bit_reversal=False)

            for row_offset, (ymm_wr, ymm_wi) in enumerate(zip(ymm_wr_list, ymm_wi_list)):
                row = row_batch_start + row_offset

                VMOVAPS(vfft_columns_0_to_8[row*2+0], ymm_wr[0])
                VMOVAPS(vfft_columns_8_to_16[row*2+0], ymm_wr[1])
                VMOVAPS(vfft_columns_0_to_8[row*2+1], ymm_wi[0])
                VMOVAPS(vfft_columns_8_to_16[row*2+1], ymm_wi[1])


        if reg_row_start is not None:
            # t8_offset = stride * (8 - row_start)
            reg_t8_offset = GeneralPurposeRegister64()
            MOV(reg_t8_offset.as_dword, 8)
            SUB(reg_t8_offset.as_dword, reg_row_start)
            IMUL(reg_t8_offset, reg_t_stride)
            reg_t8 = GeneralPurposeRegister64()
            LEA(reg_t8, [reg_t0 + reg_t8_offset * 1])
            CMP(reg_row_start, 8)
            CMOVAE(reg_t8, reg_t0)
        else:
            reg_t8 = GeneralPurposeRegister64()
            LEA(reg_t8, [reg_t0 + reg_t_stride * 8])

        reg_t0_column_8, reg_t8_column_8 = GeneralPurposeRegister64(), GeneralPurposeRegister64()
        LEA(reg_t0_column_8, [reg_t0 + YMMRegister.size])
        LEA(reg_t8_column_8, [reg_t8 + YMMRegister.size])

        fft16x16.inverse_vfft(reg_t0, reg_t8, reg_t_stride, data_in=vfft_columns_0_to_8,
            reg_row_start=reg_row_start, reg_row_end=reg_row_end, store_mask=store_mask_columns_0_to_8, relu=with_relu)

        with Block() as store_columns_8_to_16:
            CMP(reg_column_end, 8)
            JB(store_columns_8_to_16.end)

            fft16x16.inverse_vfft(reg_t0_column_8, reg_t8_column_8, reg_t_stride, data_in=vfft_columns_8_to_16, \
                reg_row_start=reg_row_start, reg_row_end=reg_row_end, store_mask=store_mask_columns_8_to_16, relu=with_relu)

        RETURN()
