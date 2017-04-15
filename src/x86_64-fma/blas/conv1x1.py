from __future__ import absolute_import
from __future__ import division

mr, nr = 2, 4

arg_input_channels = Argument(size_t, "input_channels")
arg_image_size     = Argument(size_t, "image_size")
arg_input = Argument(ptr(const_float_), "input")
arg_kernel = Argument(ptr(const_float_), "kernel")
arg_output = Argument(ptr(float_), "output")
with Function("nnp_conv1x1_only_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
    (arg_input_channels, arg_image_size, arg_input, arg_kernel, arg_output),
    target=uarch.default + isa.fma3):

    reg_input_channels = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_input_channels, arg_input_channels)
    SHL(reg_input_channels, 2)

    reg_image_size = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_image_size, arg_image_size)
    SHL(reg_image_size, 2)

    reg_inputs = [GeneralPurposeRegister64() for m in range(mr)]
    LOAD.ARGUMENT(reg_inputs[0], arg_input)
    for m in range(1, mr):
        LEA(reg_inputs[m], [reg_inputs[m - 1] + reg_image_size * 1])

    reg_kernel = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_kernel, arg_kernel)

    reg_outputs = [GeneralPurposeRegister64() for n in range(nr)]
    LOAD.ARGUMENT(reg_outputs[0], arg_output)
    for n in range(1, nr):
        LEA(reg_outputs[n], [reg_outputs[n - 1] + reg_image_size * 1])

    ymm_kernel = [[YMMRegister() for n in range(nr)] for m in range(mr)]
    for n in range(nr):
        for m in range(mr):
            VBROADCASTSS(ymm_kernel[m][n], [reg_kernel + m * float_.size])
        if n + 1 != nr:
            ADD(reg_kernel, reg_input_channels)

    main_loop = Loop()
    final_block = Block()

    SUB(reg_image_size, YMMRegister.size)
    JB(main_loop.end)
    with main_loop:
        # Load vectors from different channels of the output image
        ymm_outputs = [YMMRegister() for n in range(nr)]
        for reg_output, ymm_output in zip(reg_outputs, ymm_outputs):
            VMOVUPS(ymm_output, [reg_output])

        for m, reg_input in enumerate(reg_inputs):
            # Load vector for a channel of the input image
            ymm_input = YMMRegister()
            VMOVUPS(ymm_input, [reg_input])
            ADD(reg_input, YMMRegister.size)

            # Update all outputs using the input and corresponding kernel elements
            for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                VFMADD231PS(ymm_output, ymm_kernel[m][n], ymm_input)
                if reg_input is reg_inputs[-1]:
                    VMOVUPS([reg_output], ymm_output)
                    ADD(reg_output, YMMRegister.size)

        SUB(reg_image_size, YMMRegister.size)
        JAE(main_loop.begin)
    ADD(reg_image_size, YMMRegister.size)
    JZ(final_block.end)

    with final_block:
        reg_mask, ymm_mask = GeneralPurposeRegister64(), YMMRegister()
        LEA(reg_mask, Constant.uint32x16(*([0xFFFFFFFF] * 8 + [0x00000000] * 8)))
        SUB(reg_mask, reg_image_size)
        VMOVDQU(ymm_mask, [reg_mask + YMMRegister.size])

        # Load vectors from different channels of the output image
        ymm_outputs = [YMMRegister() for n in range(nr)]
        for reg_output, ymm_output in zip(reg_outputs, ymm_outputs):
            VMASKMOVPS(ymm_output, ymm_mask, [reg_output])

        for m, reg_input in enumerate(reg_inputs):
            # Load vector for a channel of the input image
            ymm_input = YMMRegister()
            VMASKMOVPS(ymm_input, ymm_mask, [reg_input])

            # Update all outputs using the input and corresponding kernel elements
            for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                VFMADD231PS(ymm_output, ymm_kernel[m][n], ymm_input)
                if reg_input is reg_inputs[-1]:
                    VMASKMOVPS([reg_output], ymm_mask, ymm_output)

    RETURN()


arg_mr = Argument(uint32_t, "mr")
arg_nr = Argument(uint32_t, "nr")
arg_input_channels = Argument(size_t, "input_channels")
arg_image_size     = Argument(size_t, "image_size")
arg_input = Argument(ptr(const_float_), "input")
arg_kernel = Argument(ptr(const_float_), "kernel")
arg_output = Argument(ptr(float_), "output")
with Function("nnp_conv1x1_upto_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
    (arg_mr, arg_nr, arg_input_channels, arg_image_size, arg_input, arg_kernel, arg_output),
    target=uarch.default + isa.fma3):

    reg_mr = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_mr, arg_mr)

    reg_nr = GeneralPurposeRegister32()
    LOAD.ARGUMENT(reg_nr, arg_nr)

    reg_input_channels = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_input_channels, arg_input_channels)
    SHL(reg_input_channels, 2)

    reg_image_size = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_image_size, arg_image_size)
    SHL(reg_image_size, 2)

    reg_inputs = [GeneralPurposeRegister64() for m in range(mr)]
    LOAD.ARGUMENT(reg_inputs[0], arg_input)
    for m in range(1, mr):
        LEA(reg_inputs[m], [reg_inputs[m - 1] + reg_image_size * 1])

    reg_kernel = GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_kernel, arg_kernel)

    reg_outputs = [GeneralPurposeRegister64() for n in range(nr)]
    LOAD.ARGUMENT(reg_outputs[0], arg_output)
    for n in range(1, nr):
        LEA(reg_outputs[n], [reg_outputs[n - 1] + reg_image_size * 1])

    VZEROALL()
    ymm_inputs = [YMMRegister() for m in range(mr)]
    ymm_kernel = [[YMMRegister() for n in range(nr)] for m in range(mr)]
    with Block() as load_kernels:
        for n in range(nr):
            with Block() as load_kernels_row:
                for m in range(mr):
                    VBROADCASTSS(ymm_kernel[m][n], [reg_kernel + m * float_.size])
                if m + 1 != mr:
                    CMP(reg_mr, m + 1)
                    JE(load_kernels_row.end)

            if n + 1 != nr:
                CMP(reg_nr, n + 1)
                JE(load_kernels.end)
                ADD(reg_kernel, reg_input_channels)

    main_loop = Loop()
    final_block = Block()

    SUB(reg_image_size, YMMRegister.size)
    JB(main_loop.end)
    with main_loop:
        # Load vectors from different channels of the output image
        ymm_outputs = [YMMRegister() for n in range(nr)]
        with Block() as load_outputs:
            for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                VMOVUPS(ymm_output, [reg_output])
                if n + 1 != nr:
                    CMP(reg_nr, n + 1)
                    JE(load_outputs.end)

        with Block() as load_inputs:
            for m, (ymm_input, reg_input) in enumerate(zip(ymm_inputs, reg_inputs)):
                # Load vector for a channel of the input image
                VMOVUPS(ymm_input, [reg_input])
                ADD(reg_input, YMMRegister.size)

                # Update all outputs using the input and corresponding kernel elements
                for n, ymm_output in enumerate(ymm_outputs):
                    VFMADD231PS(ymm_output, ymm_kernel[m][n], ymm_input)

                if m + 1 != mr:
                    CMP(reg_mr, m + 1)
                    JE(load_inputs.end)

        # Store vectors to different channels of the output image
        with Block() as store_outputs:
            for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                VMOVUPS([reg_output], ymm_output)
                ADD(reg_output, YMMRegister.size)
                if n + 1 != nr:
                    CMP(reg_nr, n + 1)
                    JE(store_outputs.end)

        SUB(reg_image_size, YMMRegister.size)
        JAE(main_loop.begin)
    ADD(reg_image_size, YMMRegister.size)
    JZ(final_block.end)

    with final_block:
        reg_mask, ymm_mask = GeneralPurposeRegister64(), YMMRegister()
        LEA(reg_mask, Constant.uint32x16(*([0xFFFFFFFF] * 8 + [0x00000000] * 8)))
        SUB(reg_mask, reg_image_size)
        VMOVDQU(ymm_mask, [reg_mask + YMMRegister.size])

        # Load vectors from different channels of the output image
        ymm_outputs = [YMMRegister() for n in range(nr)]
        with Block() as load_outputs:
            for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                VMASKMOVPS(ymm_output, ymm_mask, [reg_output])
                if n + 1 != nr:
                    CMP(reg_nr, n + 1)
                    JE(load_outputs.end)

        with Block() as load_inputs:
            for m, (ymm_input, reg_input) in enumerate(zip(ymm_inputs, reg_inputs)):
                # Load vector for a channel of the input image
                VMASKMOVPS(ymm_inputs[m], ymm_mask, [reg_input])

                # Update all outputs using the input and corresponding kernel elements
                for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                    VFMADD231PS(ymm_output, ymm_kernel[m][n], ymm_inputs[m])

                if m + 1 != mr:
                    CMP(reg_mr, m + 1)
                    JE(load_inputs.end)

        # Store vectors to different channels of the output image
        with Block() as store_outputs:
            for n, (reg_output, ymm_output) in enumerate(zip(reg_outputs, ymm_outputs)):
                VMASKMOVPS([reg_output], ymm_mask, ymm_output)
                if n + 1 != nr:
                    CMP(reg_nr, n + 1)
                    JE(store_outputs.end)

    RETURN()
