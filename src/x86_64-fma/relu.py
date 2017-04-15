arg_input = Argument(ptr(const_float_), "input")
arg_output = Argument(ptr(float_), "output")
arg_length = Argument(size_t, "length")
arg_negative_slope = Argument(float_, "negative_slope")
with Function("nnp_relu__avx2",
	(arg_input, arg_output, arg_length, arg_negative_slope),
	target=uarch.default + isa.avx2):

	reg_input = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_input, arg_input)

	reg_output = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_output, arg_output)

	reg_length = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_length, arg_length)

	ymm_negative_slope = YMMRegister()
	LOAD.ARGUMENT(ymm_negative_slope.as_xmm, arg_negative_slope)
	VBROADCASTSS(ymm_negative_slope, ymm_negative_slope.as_xmm)

	loop = Loop()

	TEST(reg_length, reg_length)
	JZ(loop.end)
	with loop:
		# Load (unaligned!) data and update input pointer
		ymm_data = YMMRegister()
		VMOVUPS(ymm_data, [reg_input])
		ADD(reg_input, YMMRegister.size)

		# Scale data with negative slope (for negative inputs)
		ymm_scaled_data = YMMRegister()
		VMULPS(ymm_scaled_data, ymm_data, ymm_negative_slope)

		# Select scaled data if input is negative
		VBLENDVPS(ymm_data, ymm_data, ymm_scaled_data, ymm_data)

		# Stream (aligned!) data to memory and update output pointer
		VMOVNTPS([reg_output], ymm_data)
		ADD(reg_output, YMMRegister.size)

		SUB(reg_length, YMMRegister.size // float_.size)
		JNZ(loop.begin)

	RETURN()


arg_data = Argument(ptr(float_), "data")
arg_length = Argument(size_t, "length")
arg_negative_slope = Argument(float_, "negative_slope")
with Function("nnp_inplace_relu__avx2",
	(arg_data, arg_length, arg_negative_slope),
	target=uarch.default + isa.avx2):

	reg_data = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_data, arg_data)

	reg_length = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_length, arg_length)

	ymm_negative_slope = YMMRegister()
	LOAD.ARGUMENT(ymm_negative_slope.as_xmm, arg_negative_slope)
	VBROADCASTSS(ymm_negative_slope, ymm_negative_slope.as_xmm)

	loop = Loop()

	TEST(reg_length, reg_length)
	JZ(loop.end)
	with loop:
		# Load data
		ymm_data = YMMRegister()
		VMOVAPS(ymm_data, [reg_data])

		# Scale data with negative slope (for negative inputs)
		ymm_scaled_data = YMMRegister()
		VMULPS(ymm_scaled_data, ymm_data, ymm_negative_slope)

		# Select scaled data if input is negative
		VBLENDVPS(ymm_data, ymm_data, ymm_scaled_data, ymm_data)

		# Store data back to the same location and update pointer
		VMOVAPS([reg_data], ymm_data)
		ADD(reg_data, YMMRegister.size)

		SUB(reg_length, YMMRegister.size // float_.size)
		JNZ(loop.begin)

	RETURN()


arg_output_gradient = Argument(ptr(const_float_), "output_gradient")
arg_input = Argument(ptr(const_float_), "input")
arg_input_gradient = Argument(ptr(float_), "input_gradient")
arg_length = Argument(size_t, "length")
arg_negative_slope = Argument(float_, "negative_slope")
with Function("nnp_grad_relu__avx2",
	(arg_output_gradient, arg_input, arg_input_gradient, arg_length, arg_negative_slope),
	target=uarch.default + isa.avx2):

	reg_output_gradient = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_output_gradient, arg_output_gradient)

	reg_input = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_input, arg_input)

	reg_input_gradient = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_input_gradient, arg_input_gradient)

	reg_length = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_length, arg_length)

	ymm_negative_slope = YMMRegister()
	LOAD.ARGUMENT(ymm_negative_slope.as_xmm, arg_negative_slope)
	VBROADCASTSS(ymm_negative_slope, ymm_negative_slope.as_xmm)

	loop = Loop()

	TEST(reg_length, reg_length)
	JZ(loop.end)
	with loop:
		# Load (unaligned!) gradient and update output gradient pointer
		ymm_gradient = YMMRegister()
		VMOVUPS(ymm_gradient, [reg_output_gradient])
		ADD(reg_output_gradient, YMMRegister.size)

		# Load (unaligned!) data and update input pointer
		ymm_data = YMMRegister()
		VMOVUPS(ymm_data, [reg_input])
		ADD(reg_input, YMMRegister.size)

		# Scale gradient with negative slope (for negative inputs)
		ymm_scaled_gradient = YMMRegister()
		VMULPS(ymm_scaled_gradient, ymm_gradient, ymm_negative_slope)

		# Select scaled gradient if input is negative
		VBLENDVPS(ymm_gradient, ymm_gradient, ymm_scaled_gradient, ymm_data)

		# Stream (aligned!) gradient to memory and update input gradient pointer
		VMOVAPS([reg_input_gradient], ymm_gradient)
		ADD(reg_input_gradient, YMMRegister.size)

		SUB(reg_length, YMMRegister.size // float_.size)
		JNZ(loop.begin)

	RETURN()
