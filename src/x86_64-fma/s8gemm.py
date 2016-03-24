from common import interleave


def sgemm_loop(ymm_c, reg_a, reg_b, reg_k, step_k, loop):
	assert isinstance(reg_k, GeneralPurposeRegister64)
	assert isinstance(step_k, int) and step_k >= 1
	assert isinstance(loop, Loop)

	assert isinstance(ymm_c, list)
	mr = len(ymm_c)
	assert isinstance(ymm_c[0], list)
	nr = len(ymm_c[0])
	assert all(isinstance(ymm_c_m, list) and len(ymm_c_m) == nr for ymm_c_m in ymm_c)

	step_a, step_b = mr * step_k * YMMRegister.size, nr * step_k * YMMRegister.size
	disp_shift_a = 0 if step_a <= 128 else -128
	disp_shift_b = 0 if step_b <= 128 else -128

	ymm_a = [YMMRegister() for m in range(mr)]
	ymm_b_n = YMMRegister()

	if step_k > 1:
		if disp_shift_a != 0:
			SUB(reg_a, disp_shift_a)
		if disp_shift_b != 0:
			SUB(reg_b, disp_shift_b)
		SUB(reg_k, step_k)
		JB(loop.end)

	with loop:
		for k in range(step_k):
			for m, ymm_a_m in enumerate(ymm_a):
				VMOVAPS(ymm_a_m, [reg_a + (m + mr*k) * YMMRegister.size + disp_shift_a])

			for n in range(nr):
				# offset_a = (m + mr*k) * YMMRegister.size + disp_shift_a
				# if offset_a % 64 == 0 and False:
				# 	PREFETCHNTA([reg_a + 640 + offset_a])

				VMOVAPS(ymm_b_n, [reg_b + (n + nr*k) * YMMRegister.size + disp_shift_b])

				for m in range(mr):
					VFMADD231PS(ymm_c[m][n], ymm_a[m], ymm_b_n)

		SUB(reg_a, -step_a)
		SUB(reg_b, -step_b)
		if step_k > 1:
			SUB(reg_k, step_k)
			JAE(loop.begin)
		else:
			DEC(reg_k)
			JNE(loop.begin)

	if step_k > 1:
		if disp_shift_a:
			ADD(reg_a, disp_shift_a)
		if disp_shift_b:
			ADD(reg_b, disp_shift_b)
		ADD(reg_k, step_k)


for mr in [1, 2, 3]:
	for nr in [1, 2, 3, 4]:
		arg_k = Argument(size_t, "k")
		arg_k_tile = Argument(size_t, "k_tile")
		arg_a = Argument(ptr(const_float_), "a")
		arg_b = Argument(ptr(const_float_), "b")
		arg_c = Argument(ptr(float_), "c")
		arg_row_stride = Argument(size_t, "row_stride_c")
		arg_col_stride = Argument(size_t, "column_stride_c")
		arguments = (arg_k, arg_k_tile, arg_a, arg_b, arg_c, arg_row_stride, arg_col_stride)
		with Function("nnp_s8gemm%dx%d__fma3" % (mr, nr),
			(arg_k, arg_k_tile, arg_a, arg_b, arg_c, arg_row_stride, arg_col_stride),
			target=uarch.default + isa.fma3):

			reg_k = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_k, arg_k)

			reg_k_tile = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_k_tile, arg_k_tile)

			reg_a = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_a, arg_a)

			reg_b = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_b, arg_b)

			reg_c = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_c, arg_c)

			if mr > 1:
				reg_row_stride = GeneralPurposeRegister64()
				LOAD.ARGUMENT(reg_row_stride, arg_row_stride)
				SHL(reg_row_stride, 2)

			if nr > 1:
				reg_col_stride = GeneralPurposeRegister64()
				LOAD.ARGUMENT(reg_col_stride, arg_col_stride)
				SHL(reg_col_stride, 2)

			ymm_c = [[YMMRegister() for n in range(nr)] for m in range(mr)]
			VZEROALL()

			process_by_2 = Loop()
			process_by_1 = Loop()

			if mr > 1 and nr > 1:
				sgemm_loop(ymm_c, reg_a, reg_b, reg_k, 2, process_by_2)
				JZ(process_by_1.end)
			sgemm_loop(ymm_c, reg_a, reg_b, reg_k, 1, process_by_1)

			load_and_store_c = Block()
			store_c = Block()

			TEST(reg_k_tile, reg_k_tile)
			JZ(store_c.begin)

			with load_and_store_c:
				reg_c_m0, reg_c_mn = GeneralPurposeRegister64(), GeneralPurposeRegister64()
				for m in range(mr):
					if m == 0:
						reg_c_m0 = reg_c
					else:
						ADD(reg_c_m0, reg_row_stride)
					for n in range(nr):
						if n == 0:
							if m + 1 == mr:
								reg_c_mn = reg_c_m0
							else:
								MOV(reg_c_mn, reg_c_m0)
						else:
							ADD(reg_c_mn, reg_col_stride)

						VADDPS(ymm_c[m][n], ymm_c[m][n], [reg_c_mn])
						VMOVAPS([reg_c_mn], ymm_c[m][n])

				RETURN()

			with store_c:
				reg_c_m0, reg_c_mn = GeneralPurposeRegister64(), GeneralPurposeRegister64()
				for m in range(mr):
					if m == 0:
						reg_c_m0 = reg_c
					else:
						ADD(reg_c_m0, reg_row_stride)
					for n in range(nr):
						if n == 0:
							if m + 1 == mr:
								reg_c_mn = reg_c_m0
							else:
								MOV(reg_c_mn, reg_c_m0)
						else:
							ADD(reg_c_mn, reg_col_stride)

						VMOVAPS([reg_c_mn], ymm_c[m][n])

				RETURN()
