arg_mem = Argument(ptr(), "mem")
arg_len = Argument(size_t, "n")
with Function("read_memory", (arg_mem, arg_len)):
	reg_mem = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_mem, arg_mem)

	reg_len = GeneralPurposeRegister64()
	LOAD.ARGUMENT(reg_len, arg_len)

	main_loop = Loop()
	SUB(reg_len, 64)
	JB(main_loop.end)
	with main_loop:
		MOVAPS(xmm0, [reg_mem])
		ADD(reg_mem, 64)
		SUB(reg_len, 64)
		JAE(main_loop.begin)

	RETURN()
