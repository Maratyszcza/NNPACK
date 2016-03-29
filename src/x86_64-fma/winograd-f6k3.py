import winograd.o6x6k3x3


arg_d_pointer = Argument(ptr(const_float_), name="d")
arg_w_pointer = Argument(ptr(float_), name="w")
with Function("nnp_iwt_f6k3__fma3", (arg_d_pointer, arg_w_pointer),
    target=uarch.default + isa.fma3):

    reg_d = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_d, arg_d_pointer)

    reg_w = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_w, arg_w_pointer)

    ymm_data = [YMMRegister() for _ in range(8)]
    for i, ymm_row in enumerate(ymm_data):
    	VMOVUPS(ymm_row, [reg_d + i * YMMRegister.size])

    ymm_data = winograd.o6x6k3x3.input_transform(ymm_data)

    for i, ymm_row in enumerate(ymm_data):
    	VMOVUPS([reg_w + i * YMMRegister.size], ymm_row)

    RETURN()


arg_g_pointer = Argument(ptr(const_float_), name="g")
arg_w_pointer = Argument(ptr(float_), name="w")
with Function("nnp_kwt_f6k3__fma3", (arg_g_pointer, arg_w_pointer),
    target=uarch.default + isa.fma3):

    reg_g = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_g, arg_g_pointer)

    reg_w = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_w, arg_w_pointer)

    ymm_data = [YMMRegister() for _ in range(3)]
    for i, ymm_row in enumerate(ymm_data):
    	VMOVUPS(ymm_row, [reg_g + i * YMMRegister.size])

    ymm_data = winograd.o6x6k3x3.kernel_transform(ymm_data)

    for i, ymm_row in enumerate(ymm_data):
    	VMOVUPS([reg_w + i * YMMRegister.size], ymm_row)

    RETURN()


arg_m_pointer = Argument(ptr(const_float_), name="m")
arg_s_pointer = Argument(ptr(float_), name="s")
with Function("nnp_owt_f6k3__fma3", (arg_m_pointer, arg_s_pointer),
    target=uarch.default + isa.fma3):

    reg_m = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_m, arg_m_pointer)

    reg_s = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_s, arg_s_pointer)

    ymm_m = [YMMRegister() for _ in range(8)]
    for i, ymm_row in enumerate(ymm_m):
    	VMOVUPS(ymm_row, [reg_m + i * YMMRegister.size])

    ymm_s = winograd.o6x6k3x3.output_transform(ymm_m)

    for i, ymm_row in enumerate(ymm_s):
    	VMOVUPS([reg_s + i * YMMRegister.size], ymm_row)

    RETURN()
