from common import _MM_SHUFFLE


arg_src_pointer = Argument(ptr(const_float_), name="src_pointer")
arg_dst_pointer = Argument(ptr(float_), name="dst_pointer")
arg_src_stride = Argument(size_t, name="src_stride")
arg_src_row_offset = Argument(uint32_t, name="src_row_offset")
arg_src_row_count = Argument(uint32_t, name="src_row_count")
arg_src_column_offset = Argument(uint32_t, name="src_column_offset")
arg_src_column_count = Argument(uint32_t, name="src_column_count")
arg_dst_column_count = Argument(uint32_t, name="dst_column_count")
with Function("nnp_maxpool_2x2_2x2__avx2",
    (arg_src_pointer, arg_dst_pointer, arg_src_stride,
    arg_src_row_offset, arg_src_row_count, arg_src_column_offset, arg_src_column_count,
    arg_dst_column_count),
    target=uarch.default + isa.fma3 + isa.avx2):

    reg_src_ptr = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_src_ptr, arg_src_pointer)

    reg_dst_ptr = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_dst_ptr, arg_dst_pointer)

    reg_src_stride = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_src_stride, arg_src_stride)

    reg_src_row_index = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_src_row_index, arg_src_row_offset)

    reg_src_row_count = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_src_row_count, arg_src_row_count)

    reg_src_column_start = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_src_column_start, arg_src_column_offset)

    reg_src_column_end = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_src_column_end, arg_src_column_count)
    ADD(reg_src_column_end, reg_src_column_start)

    reg_dst_column_count = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_dst_column_count, arg_dst_column_count)

    ymm_src_column_start, ymm_src_column_end, ymm_dst_column_count = YMMRegister(), YMMRegister(), YMMRegister()
    VMOVD(ymm_src_column_start.as_xmm, reg_src_column_start)
    VMOVD(ymm_src_column_end.as_xmm, reg_src_column_end)
    VMOVD(ymm_dst_column_count.as_xmm, reg_dst_column_count)
    VPBROADCASTD(ymm_src_column_start, ymm_src_column_start.as_xmm)
    VPBROADCASTD(ymm_src_column_end, ymm_src_column_end.as_xmm)
    VPBROADCASTD(ymm_dst_column_count, ymm_dst_column_count.as_xmm)

    ymm_column_01234567, ymm_column_89ABCDEF = YMMRegister(), YMMRegister()
    VMOVDQA(ymm_column_01234567, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))
    VMOVDQA(ymm_column_89ABCDEF, Constant.uint32x8(8, 9, 10, 11, 12, 13, 14, 15))

    ymm_src_column_start_gt_01234567, ymm_src_column_end_gt_01234567 = YMMRegister(), YMMRegister()
    VPCMPGTD(ymm_src_column_start_gt_01234567, ymm_src_column_start, ymm_column_01234567)
    VPCMPGTD(ymm_src_column_end_gt_01234567, ymm_src_column_end, ymm_column_01234567)

    ymm_src_column_start_gt_89ABCDEF, ymm_src_column_end_gt_89ABCDEF = YMMRegister(), YMMRegister()
    VPCMPGTD(ymm_src_column_start_gt_89ABCDEF, ymm_src_column_start, ymm_column_89ABCDEF)
    VPCMPGTD(ymm_src_column_end_gt_89ABCDEF, ymm_src_column_end, ymm_column_89ABCDEF)

    ymm_src_mask_columns_0_to_8, ymm_src_mask_columns_8_to_16 = YMMRegister(), YMMRegister()
    VPANDN(ymm_src_mask_columns_0_to_8, ymm_src_column_start_gt_01234567, ymm_src_column_end_gt_01234567)
    VPANDN(ymm_src_mask_columns_8_to_16, ymm_src_column_start_gt_89ABCDEF, ymm_src_column_end_gt_89ABCDEF)

    ymm_dst_mask_columns_0_to_8 = YMMRegister()
    VPCMPGTD(ymm_dst_mask_columns_0_to_8, ymm_dst_column_count, ymm_column_01234567)

    # data points to the first element, which is loaded into lane `reg_column_start`
    # However, VMASKMOVPS expects pointer to the first lane, even if it is not loaded.
    # Adjust the pointer by subtracting column_offset, in bytes
    SHL(reg_src_column_start, 2)
    SUB(reg_src_ptr, reg_src_column_start.as_qword)

    # Multiply stride by sizeof(float) to convert from elements to bytes
    SHL(reg_src_stride, 2)

    ymm_row0 = YMMRegister(), YMMRegister()
    ymm_row1 = YMMRegister(), YMMRegister()

    ymm_minus_inf = YMMRegister()
    VMOVAPS(ymm_minus_inf, Constant.float32x8(-float("inf")))

    VMOVAPS(ymm_row0[0], ymm_minus_inf)
    VMOVAPS(ymm_row0[1], ymm_minus_inf)
    VMOVAPS(ymm_row1[0], ymm_minus_inf)
    VMOVAPS(ymm_row1[1], ymm_minus_inf)

    NEG(reg_src_row_index)

    with Block() as load_row0:
        CMP(reg_src_row_index, reg_src_row_count)
        JAE(load_row0.end)

        VMASKMOVPS(ymm_row0[0], ymm_src_mask_columns_0_to_8, [reg_src_ptr])
        VBLENDVPS(ymm_row0[0], ymm_minus_inf, ymm_row0[0], ymm_src_mask_columns_0_to_8)
        VMASKMOVPS(ymm_row0[1], ymm_src_mask_columns_8_to_16, [reg_src_ptr + YMMRegister.size])
        VBLENDVPS(ymm_row0[1], ymm_minus_inf, ymm_row0[1], ymm_src_mask_columns_8_to_16)

        ADD(reg_src_ptr, reg_src_stride)

    with Block() as load_row1:
        INC(reg_src_row_index)
        CMP(reg_src_row_index, reg_src_row_count)
        JAE(load_row1.end)

        VMASKMOVPS(ymm_row1[0], ymm_src_mask_columns_0_to_8, [reg_src_ptr])
        VBLENDVPS(ymm_row1[0], ymm_minus_inf, ymm_row1[0], ymm_src_mask_columns_0_to_8)
        VMASKMOVPS(ymm_row1[1], ymm_src_mask_columns_8_to_16, [reg_src_ptr + YMMRegister.size])
        VBLENDVPS(ymm_row1[1], ymm_minus_inf, ymm_row1[1], ymm_src_mask_columns_8_to_16)

    # ymm_row[0] = ( x7  x6  x5  x4  x3  x2  x1 x0 )
    # ymm_row[1] = ( x15 x14 x13 x12 x11 x10 x9 x8 )
    ymm_row = YMMRegister(), YMMRegister()
    VMAXPS(ymm_row[0], ymm_row0[0], ymm_row1[0])
    VMAXPS(ymm_row[1], ymm_row0[1], ymm_row1[1])

    # ymm_row[0] = ( x14 x12 x6 x4 x10 x8 x2 x0 )
    # ymm_row[1] = ( x15 x13 x7 x5 x11 x9 x3 x1 )
    ymm_tmp = YMMRegister()
    VSHUFPS(ymm_tmp, ymm_row[0], ymm_row[1], _MM_SHUFFLE(2, 0, 2, 0))
    VSHUFPS(ymm_row[1], ymm_row[0], ymm_row[1], _MM_SHUFFLE(3, 1, 3, 1))
    SWAP.REGISTERS(ymm_row[0], ymm_tmp)

    # ymm_out = ( y7 y6 y3 y2 y5 y4 y1 y0 )
    ymm_out = YMMRegister()
    VMAXPS(ymm_out, ymm_row[0], ymm_row[1])
    VPERMPD(ymm_out, ymm_out, _MM_SHUFFLE(3, 1, 2, 0))

    VMASKMOVPS([reg_dst_ptr], ymm_dst_mask_columns_0_to_8, ymm_out)

    RETURN()
