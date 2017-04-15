from common import _MM_SHUFFLE
from vecmath.exp import simd_exp


arg_n = Argument(size_t, "n")
arg_v = Argument(ptr(const_float_), "v")
with Function("max__avx", (arg_n, arg_v), float_,
    target=uarch.default + isa.avx):

    reg_n = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_n, arg_n)

    reg_v = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_v, arg_v)

    unroll_loop = Loop()
    vector_loop = Loop()
    final_block = Block()

    simd_width = YMMRegister.size // float_.size
    unroll_factor = 4

    # Initialize reduction registers with the first element (v[0])
    ymm_ms = [YMMRegister() for _ in range(unroll_factor)]
    VBROADCASTSS(ymm_ms[0], [reg_v])
    for ymm_m in ymm_ms[1:]:
        VMOVAPS(ymm_m, ymm_ms[0])

    # Unrolled vectorized loop
    SUB(reg_n, simd_width * unroll_factor)
    JB(unroll_loop.end)
    with unroll_loop:
        for i, ymm_m in enumerate(ymm_ms):
            VMAXPS(ymm_m, ymm_m, [reg_v + i * YMMRegister.size])

        SUB(reg_v, -unroll_factor * YMMRegister.size)
        SUB(reg_n, simd_width * unroll_factor)
        JAE(unroll_loop.begin)

    VMAXPS(ymm_ms[0], ymm_ms[0], ymm_ms[1])
    VMAXPS(ymm_ms[2], ymm_ms[2], ymm_ms[3])
    VMAXPS(ymm_ms[0], ymm_ms[0], ymm_ms[2])
    ymm_m = ymm_ms[0]

    ADD(reg_n, simd_width * unroll_factor)
    JZ(final_block.end)

    # Vectorized loop without unrolling
    SUB(reg_n, simd_width)
    JB(vector_loop.end)
    with vector_loop:
        VMAXPS(ymm_m, ymm_m, [reg_v])

        ADD(reg_v, YMMRegister.size)
        SUB(reg_n, simd_width)
        JAE(vector_loop.begin)
    ADD(reg_n, simd_width)
    JZ(final_block.end)

    # Process remainder: 0 < reg_n < simd_width
    with final_block:
        reg_mask = GeneralPurposeRegister64()
        LEA(reg_mask, Constant.uint32x16(*([0xFFFFFFFF] * 8 + [0x00000000] * 8)))
        NEG(reg_n)
        LEA(reg_mask, [reg_mask + reg_n * 4 + 16])

        ymm_mask = YMMRegister()
        VMOVUPS(ymm_mask, [reg_mask])

        ymm_temp = YMMRegister()
        VMASKMOVPS(ymm_temp, ymm_mask, [reg_v])
        VBLENDVPS(ymm_temp, ymm_temp, ymm_m, ymm_mask)
        VMAXPS(ymm_m, ymm_m, ymm_temp)

    ymm_temp = YMMRegister()
    VPERM2F128(ymm_temp, ymm_m, ymm_m, 0x01)
    VMAXPS(ymm_m, ymm_m, ymm_temp)

    VPERMILPS(ymm_temp, ymm_m, _MM_SHUFFLE(1, 0, 3, 2))
    VMAXPS(ymm_m, ymm_m, ymm_temp)

    VPERMILPS(ymm_temp, ymm_m, _MM_SHUFFLE(2, 3, 0, 1))
    VMAXPS(ymm_m, ymm_m, ymm_temp)

    RETURN(ymm_m.as_xmm)


arg_n = Argument(size_t, "n")
arg_v = Argument(ptr(const_float_), "v")
arg_c = Argument(float_, "c")
with Function("sum_exp_minus_c__avx2", (arg_n, arg_v, arg_c), float_,
    target=uarch.default + isa.avx2):

    reg_n = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_n, arg_n)

    reg_v = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_v, arg_v)

    ymm_c = YMMRegister()
    LOAD.ARGUMENT(ymm_c.as_xmm, arg_c)
    VBROADCASTSS(ymm_c, ymm_c.as_xmm)

    unroll_loop = Loop()
    vector_loop = Loop()
    final_block = Block()

    simd_width = YMMRegister.size // float_.size
    unroll_factor = 3

    # Clear reduction registers
    ymm_sums = [YMMRegister() for _ in range(unroll_factor)]
    for ymm_sum in ymm_sums:
        VXORPS(ymm_sum.as_xmm, ymm_sum.as_xmm, ymm_sum.as_xmm)

    # Unrolled vectorized loop
    SUB(reg_n, simd_width * unroll_factor)
    JB(unroll_loop.end)
    with unroll_loop:
        ymm_xs = [YMMRegister() for _ in ymm_sums]
        for i, ymm_x in enumerate(ymm_xs):
            VMOVUPS(ymm_x, [reg_v + i * YMMRegister.size])
            VSUBPS(ymm_x, ymm_x, ymm_c)

        ymm_ys = simd_exp(ymm_xs)

        for ymm_sum, ymm_y in zip(ymm_sums, ymm_ys):
            VADDPS(ymm_sum, ymm_sum, ymm_y)

        SUB(reg_v, -unroll_factor * YMMRegister.size)
        SUB(reg_n, simd_width * unroll_factor)
        JAE(unroll_loop.begin)

    VADDPS(ymm_sums[0], ymm_sums[0], ymm_sums[1])
    VADDPS(ymm_sums[0], ymm_sums[0], ymm_sums[2])
    ymm_sum = ymm_sums[0]

    ADD(reg_n, simd_width * unroll_factor)
    JZ(final_block.end)

    # Vectorized loop without unrolling
    SUB(reg_n, simd_width)
    JB(vector_loop.end)
    with vector_loop:
        ymm_x = YMMRegister()
        VMOVUPS(ymm_x, [reg_v])
        VSUBPS(ymm_x, ymm_x, ymm_c)

        ymm_y = simd_exp([ymm_x])[0]

        VADDPS(ymm_sum, ymm_sum, ymm_y)

        ADD(reg_v, YMMRegister.size)
        SUB(reg_n, simd_width)
        JAE(vector_loop.begin)
    ADD(reg_n, simd_width)
    JZ(final_block.end)

    # Process remainder: 0 < reg_n < simd_width
    with final_block:
        ymm_mask = YMMRegister()
        VMOVD(ymm_mask.as_xmm, reg_n.as_dword)
        VPBROADCASTD(ymm_mask, ymm_mask.as_xmm)
        VPCMPGTD(ymm_mask, ymm_mask, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))

        ymm_x = YMMRegister()
        VMASKMOVPS(ymm_x, ymm_mask, [reg_v])
        VSUBPS(ymm_x, ymm_x, ymm_c)

        ymm_y = simd_exp([ymm_x])[0]

        VANDPS(ymm_y, ymm_y, ymm_mask)
        VADDPS(ymm_sum, ymm_sum, ymm_y)

    ymm_temp = YMMRegister()
    VPERM2F128(ymm_temp, ymm_sum, ymm_sum, 0x01)
    VADDPS(ymm_sum, ymm_sum, ymm_temp)

    VPERMILPS(ymm_temp, ymm_sum, _MM_SHUFFLE(1, 0, 3, 2))
    VADDPS(ymm_sum, ymm_sum, ymm_temp)

    VPERMILPS(ymm_temp, ymm_sum, _MM_SHUFFLE(2, 3, 0, 1))
    VADDPS(ymm_sum, ymm_sum, ymm_temp)

    RETURN(ymm_sum.as_xmm)


def scaled_exp_minus_c(reg_n, reg_x, reg_y, ymm_scale, ymm_c):
    unroll_loop = Loop()
    vector_loop = Loop()
    final_block = Block()

    simd_width = YMMRegister.size // float_.size
    unroll_factor = 3

    # Unrolled vectorized loop
    SUB(reg_n, simd_width * unroll_factor)
    JB(unroll_loop.end)
    with unroll_loop:
        ymm_xs = [YMMRegister() for _ in range(unroll_factor)]
        for i, ymm_x in enumerate(ymm_xs):
            VMOVUPS(ymm_x, [reg_x + i * YMMRegister.size])
            VSUBPS(ymm_x, ymm_x, ymm_c)
        if reg_x != reg_y:
            SUB(reg_x, -unroll_factor * YMMRegister.size)

        ymm_ys = simd_exp(ymm_xs)

        for i, ymm_y in enumerate(ymm_ys):
            VMULPS(ymm_y, ymm_y, ymm_scale)
            VMOVUPS([reg_y + i * YMMRegister.size], ymm_y)
        SUB(reg_y, -unroll_factor * YMMRegister.size)

        SUB(reg_n, simd_width * unroll_factor)
        JAE(unroll_loop.begin)
    ADD(reg_n, simd_width * unroll_factor)
    JZ(final_block.end)

    # Vectorized loop without unrolling
    SUB(reg_n, simd_width)
    JB(vector_loop.end)
    with vector_loop:
        ymm_x = YMMRegister()
        VMOVUPS(ymm_x, [reg_x])
        if reg_x != reg_y:
            ADD(reg_x, YMMRegister.size)
        VSUBPS(ymm_x, ymm_x, ymm_c)

        ymm_y = simd_exp([ymm_x])[0]

        VMULPS(ymm_y, ymm_y, ymm_scale)
        VMOVUPS([reg_y], ymm_y)
        ADD(reg_y, YMMRegister.size)

        SUB(reg_n, simd_width)
        JAE(vector_loop.begin)
    ADD(reg_n, simd_width)
    JZ(final_block.end)

    # Process remainder: 0 < reg_n < simd_width
    with final_block:
        ymm_mask = YMMRegister()
        VMOVD(ymm_mask.as_xmm, reg_n.as_dword)
        VPBROADCASTD(ymm_mask, ymm_mask.as_xmm)
        VPCMPGTD(ymm_mask, ymm_mask, Constant.uint32x8(0, 1, 2, 3, 4, 5, 6, 7))

        ymm_x = YMMRegister()
        VMASKMOVPS(ymm_x, ymm_mask, [reg_x])
        VSUBPS(ymm_x, ymm_x, ymm_c)

        ymm_y = simd_exp([ymm_x])[0]

        VMULPS(ymm_y, ymm_y, ymm_scale)
        VMASKMOVPS([reg_y], ymm_mask, ymm_y)


arg_n = Argument(size_t, "n")
arg_v = Argument(ptr(const_float_), "v")
arg_scale = Argument(float_, "scale")
arg_c = Argument(float_, "c")
with Function("inplace_scaled_exp_minus_c__avx2", (arg_n, arg_v, arg_scale, arg_c),
    target=uarch.default + isa.avx2):

    reg_n = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_n, arg_n)

    reg_v = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_v, arg_v)

    ymm_scale = YMMRegister()
    LOAD.ARGUMENT(ymm_scale.as_xmm, arg_scale)
    VBROADCASTSS(ymm_scale, ymm_scale.as_xmm)

    ymm_c = YMMRegister()
    LOAD.ARGUMENT(ymm_c.as_xmm, arg_c)
    VBROADCASTSS(ymm_c, ymm_c.as_xmm)

    scaled_exp_minus_c(reg_n, reg_v, reg_v, ymm_scale, ymm_c)

    RETURN()

arg_n = Argument(size_t, "n")
arg_x = Argument(ptr(const_float_), "x")
arg_y = Argument(ptr(float_), "y")
arg_scale = Argument(float_, "scale")
arg_c = Argument(float_, "c")
with Function("scaled_exp_minus_c__avx2", (arg_n, arg_x, arg_y, arg_scale, arg_c),
    target=uarch.default + isa.avx2):

    reg_n = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_n, arg_n)

    reg_x = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_x, arg_x)

    reg_y = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_y, arg_y)

    ymm_scale = YMMRegister()
    LOAD.ARGUMENT(ymm_scale.as_xmm, arg_scale)
    VBROADCASTSS(ymm_scale, ymm_scale.as_xmm)

    ymm_c = YMMRegister()
    LOAD.ARGUMENT(ymm_c.as_xmm, arg_c)
    VBROADCASTSS(ymm_c, ymm_c.as_xmm)

    scaled_exp_minus_c(reg_n, reg_x, reg_y, ymm_scale, ymm_c)

    RETURN()
