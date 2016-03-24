from common import interleave


def cgemm_loop(ymm_c, reg_a, reg_b, reg_k, step_k, loop, conjugate_b, mixed_columns):
	ymm_c_real, ymm_c_imag = ymm_c
	assert isinstance(reg_k, GeneralPurposeRegister64)
	assert isinstance(step_k, int) and step_k >= 1
	assert isinstance(mixed_columns, bool)
	assert isinstance(loop, Loop)

	assert isinstance(ymm_c_real, list) and isinstance(ymm_c_imag, list) and len(ymm_c_real) == len(ymm_c_imag)
	mr = len(ymm_c_real)
	assert isinstance(ymm_c_real[0], list)
	nr = len(ymm_c_real[0])
	assert all(isinstance(ymm_c_real_m, list) and len(ymm_c_real_m) == nr for ymm_c_real_m in ymm_c_real)
	assert all(isinstance(ymm_c_imag_m, list) and len(ymm_c_imag_m) == nr for ymm_c_imag_m in ymm_c_imag)

	step_a, step_b = mr * step_k * YMMRegister.size * 2, nr * step_k * YMMRegister.size * 2
	disp_shift_a = 0 if step_a <= 128 else -128
	disp_shift_b = 0 if step_b <= 128 else -128

	ymm_a_real, ymm_a_imag = tuple([YMMRegister() for m in range(mr)] for c in range(2))
	ymm_b_real, ymm_b_imag = tuple([YMMRegister() for n in range(nr)] for c in range(2))

	use_disp_shift = False

	if step_k > 1:
		if disp_shift_a != 0:
			SUB(reg_a, disp_shift_a)
		if disp_shift_b != 0:
			SUB(reg_b, disp_shift_b)
		SUB(reg_k, step_k)
		JB(loop.end)

	with loop:
		for k in range(step_k):
			for i, ymm_a in enumerate(interleave(ymm_a_real, ymm_a_imag)):
				VMOVAPS(ymm_a, [reg_a + (i + 2*mr*k) * YMMRegister.size + disp_shift_a])

			for i, ymm_b in enumerate(interleave(ymm_b_real, ymm_b_imag)):
				VMOVAPS(ymm_b, [reg_b + (i + 2*nr*k) * YMMRegister.size + disp_shift_b])

			for n in range(nr):
				for m in range(mr):
					VFMADD231PS(ymm_c_real[m][n], ymm_a_real[m], ymm_b_real[n])

				if mixed_columns:
					VBLENDPS(ymm_b_real[n], ymm_b_real[n], ymm_b_imag[n], 0b00000011)
				for m in range(mr):
					VFMADD231PS(ymm_c_imag[m][n], ymm_a_imag[m], ymm_b_real[n])

			if nr > 1 and mixed_columns:
				zero_columns01_mask = YMMRegister()
				VMOVAPS(zero_columns01_mask, Constant.uint32x8(0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))
			else:
				zero_columns01_mask = Constant.uint32x8(0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

			# if step_k > 1:
			# 	PREFETCHNTA([reg_a + 640])

			for n in range(nr):
				if mixed_columns:
					VANDPS(ymm_b_imag[n], ymm_b_imag[n], zero_columns01_mask)
				for m in range(mr):
					if conjugate_b:
						VFMADD231PS(ymm_c_real[m][n], ymm_a_imag[m], ymm_b_imag[n])
						VFNMADD231PS(ymm_c_imag[m][n], ymm_a_real[m], ymm_b_imag[n])
					else:
						VFNMADD231PS(ymm_c_real[m][n], ymm_a_imag[m], ymm_b_imag[n])
						VFMADD231PS(ymm_c_imag[m][n], ymm_a_real[m], ymm_b_imag[n])

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


for conjugate in [None, "b", "a"]:
	for mr in [1, 2]:
		for nr in [1, 2]:
			for mixed_columns in [True, False]:
				arg_k = Argument(size_t, "k")
				arg_k_tile = Argument(size_t, "k_tile")
				arg_a = Argument(ptr(const_float_), "a")
				arg_b = Argument(ptr(const_float_), "b")
				arg_c = Argument(ptr(float_), "c")
				arg_row_stride = Argument(size_t, "row_stride_c")
				arg_col_stride = Argument(size_t, "column_stride_c")

				with Function("nnp_{type}gemm{conjugate}{mr}x{nr}__fma3".format(
						type="s4c6" if mixed_columns else "c8",
						conjugate={None: "", "a": "ca", "b": "cb"}[conjugate],
						mr=mr, nr=nr),
					(arg_k, arg_k_tile, arg_a, arg_b, arg_c, arg_row_stride, arg_col_stride),
					target=uarch.default + isa.fma3):

					load_data = True

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

					ymm_c_real, ymm_c_imag = tuple([[YMMRegister() for n in range(nr)] for m in range(mr)] for c in range(2))
					VZEROALL()

					process_by_2 = Loop()
					process_by_1 = Loop()

					if conjugate != "a":
						if mr > 1 and nr > 1:
							cgemm_loop((ymm_c_real, ymm_c_imag), reg_a, reg_b, reg_k, 2, process_by_2,
								conjugate_b=conjugate == "b", mixed_columns=mixed_columns)
							JZ(process_by_1.end)
						cgemm_loop((ymm_c_real, ymm_c_imag), reg_a, reg_b, reg_k, 1, process_by_1,
							conjugate_b=conjugate == "b", mixed_columns=mixed_columns)
					else:
						ymm_ct_real, ymm_ct_imag = map(list, zip(*ymm_c_real)), map(list, zip(*ymm_c_imag))
						if mr > 1 and nr > 1:
							cgemm_loop((ymm_ct_real, ymm_ct_imag), reg_b, reg_a, reg_k, 2, process_by_2,
								conjugate_b=True, mixed_columns=mixed_columns)
							JZ(process_by_1.end)
						cgemm_loop((ymm_ct_real, ymm_ct_imag), reg_b, reg_a, reg_k, 1, process_by_1,
							conjugate_b=True, mixed_columns=mixed_columns)

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

								VADDPS(ymm_c_real[m][n], ymm_c_real[m][n], [reg_c_mn])
								VADDPS(ymm_c_imag[m][n], ymm_c_imag[m][n], [reg_c_mn + YMMRegister.size])

								VMOVAPS([reg_c_mn], ymm_c_real[m][n])
								VMOVAPS([reg_c_mn + YMMRegister.size], ymm_c_imag[m][n])

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

								VMOVAPS([reg_c_mn], ymm_c_real[m][n])
								VMOVAPS([reg_c_mn + YMMRegister.size], ymm_c_imag[m][n])

						RETURN()
