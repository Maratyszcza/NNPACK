from __future__ import absolute_import
from __future__ import division

mr, nr = 2, 2

for conjugate_b, transpose_c in [(False, False), (True, False), (True, True)]:
	arg_k = Argument(size_t, "k")
	arg_update = Argument(size_t, "update")
	arg_a = Argument(ptr(const_float_), "a")
	arg_b = Argument(ptr(const_float_), "b")
	arg_c = Argument(ptr(float_), "c")
	arg_row_stride = Argument(size_t, "row_stride_c")
	with Function("nnp_s4c6gemm{conjb}{transc}_only_{mr}x{nr}__fma3".format(mr=mr, nr=nr,
			conjb="_conjb" if conjugate_b else "",
			transc="_transc" if transpose_c else ""),
		(arg_k, arg_update, arg_a, arg_b, arg_c, arg_row_stride),
		target=uarch.default + isa.fma3):

		reg_k = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_k, arg_k)

		reg_update = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_update, arg_update)

		reg_a = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_a, arg_a)

		reg_b = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_b, arg_b)

		reg_c = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_c, arg_c)

		reg_row_stride = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_row_stride, arg_row_stride)
		SHL(reg_row_stride, 2)

		with Block() as prefetch_c:
			if not transpose_c:
				for m in range(mr):
					PREFETCHT0([reg_c])
					if m + 1 != mr:
						ADD(reg_c, reg_row_stride)
			else:
				for n in range(nr):
					PREFETCHT0([reg_c])
					if n + 1 != nr:
						ADD(reg_c, reg_row_stride)

		ymm_c_re, ymm_c_im = tuple([[YMMRegister() for n in range(nr)] for m in range(mr)] for c in range(2))
		VZEROALL()

		ymm_a_re, ymm_a_im = tuple([YMMRegister() for m in range(2*mr)] for c in range(2))
		ymm_b_re, ymm_b_im = tuple([YMMRegister() for n in range(2*nr)] for c in range(2))
		with Loop() as loop:
			for m in range(mr):
				VMOVAPS(ymm_a_re[m], [reg_a + (2*m+0) * YMMRegister.size])
				VMOVAPS(ymm_a_im[m], [reg_a + (2*m+1) * YMMRegister.size])
			SUB(reg_a, -YMMRegister.size * 2 * mr)

			for n in range(nr):
				VMOVAPS(ymm_b_re[n], [reg_b + (2*n+0) * YMMRegister.size])
				for m in range(mr):
					VFMADD231PS(ymm_c_re[m][n], ymm_a_re[m], ymm_b_re[n])

				VMOVAPS(ymm_b_im[n], [reg_b + (2*n+1) * YMMRegister.size])
				VBLENDPS(ymm_b_re[n], ymm_b_re[n], ymm_b_im[n], 0b00000011)
				for m in range(mr):
					VFMADD231PS(ymm_c_im[m][n], ymm_a_im[m], ymm_b_re[n])
			SUB(reg_b, -YMMRegister.size * 2 * nr)

			ymm_zero_columns01_mask = YMMRegister()
			VMOVAPS(ymm_zero_columns01_mask, Constant.uint32x8(0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))

			for n in range(nr):
				VANDPS(ymm_b_im[n], ymm_b_im[n], ymm_zero_columns01_mask)
				for m in range(mr):
					if conjugate_b:
						VFMADD231PS(ymm_c_re[m][n], ymm_a_im[m], ymm_b_im[n])
						VFNMADD231PS(ymm_c_im[m][n], ymm_a_re[m], ymm_b_im[n])
					else:
						VFNMADD231PS(ymm_c_re[m][n], ymm_a_im[m], ymm_b_im[n])
						VFMADD231PS(ymm_c_im[m][n], ymm_a_re[m], ymm_b_im[n])

			DEC(reg_k)
			JNZ(loop.begin)

		store_c = Block()

		# Check if we need to update C or overwrite it
		TEST(reg_update, reg_update)
		JZ(store_c.begin)

		if transpose_c:
			mr, nr = nr, mr
			ymm_c_re = [list(ymm_column) for ymm_column in zip(*ymm_c_re)]
			ymm_c_im = [list(ymm_column) for ymm_column in zip(*ymm_c_im)]

		with Block() as update_c:
			for m in reversed(range(mr)):
				for n in range(nr):
					VADDPS(ymm_c_re[m][n], ymm_c_re[m][n], [reg_c + (2*n+0) * YMMRegister.size])
					VADDPS(ymm_c_im[m][n], ymm_c_im[m][n], [reg_c + (2*n+1) * YMMRegister.size])
					VMOVAPS([reg_c + (2*n+0) * YMMRegister.size], ymm_c_re[m][n])
					VMOVAPS([reg_c + (2*n+1) * YMMRegister.size], ymm_c_im[m][n])
				if m != 0:
					SUB(reg_c, reg_row_stride)

		RETURN()

		with store_c:
			for m in reversed(range(mr)):
				for n in range(nr):
					VMOVAPS([reg_c + (2*n+0) * YMMRegister.size], ymm_c_re[m][n])
					VMOVAPS([reg_c + (2*n+1) * YMMRegister.size], ymm_c_im[m][n])
				if m != 0:
					SUB(reg_c, reg_row_stride)

		RETURN()


	arg_mr = Argument(uint32_t, "mr")
	arg_nr = Argument(uint32_t, "nr")
	arg_k = Argument(size_t, "k")
	arg_update = Argument(size_t, "update")
	arg_a = Argument(ptr(const_float_), "a")
	arg_b = Argument(ptr(const_float_), "b")
	arg_c = Argument(ptr(float_), "c")
	arg_row_stride = Argument(size_t, "row_stride_c")
	with Function("nnp_s4c6gemm{conjb}{transc}_upto_{mr}x{nr}__fma3".format(mr=mr, nr=nr,
			conjb="_conjb" if conjugate_b else "",
			transc="_transc" if transpose_c else ""),
		(arg_mr, arg_nr, arg_k, arg_update, arg_a, arg_b, arg_c, arg_row_stride),
		target=uarch.default + isa.fma3):

		reg_mr = GeneralPurposeRegister32()
		LOAD.ARGUMENT(reg_mr, arg_mr)

		reg_nr = GeneralPurposeRegister32()
		LOAD.ARGUMENT(reg_nr, arg_nr)

		reg_k = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_k, arg_k)

		reg_update = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_update, arg_update)

		reg_a = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_a, arg_a)

		reg_b = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_b, arg_b)

		reg_c = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_c, arg_c)

		reg_row_stride = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_row_stride, arg_row_stride)
		SHL(reg_row_stride, 2)

		ymm_c_re, ymm_c_im = tuple([[YMMRegister() for n in range(nr)] for m in range(mr)] for c in range(2))
		VZEROALL()

		ymm_a_re, ymm_a_im = tuple([YMMRegister() for m in range(mr)] for c in range(2))
		ymm_b_re, ymm_b_im = tuple([YMMRegister() for n in range(nr)] for c in range(2))
		with Loop() as loop:
			with Block() as load_a:
				for m, (ymm_re, ymm_im) in enumerate(zip(ymm_a_re, ymm_a_im)):
					VMOVAPS(ymm_re, [reg_a])
					VMOVAPS(ymm_im, [reg_a + YMMRegister.size])
					ADD(reg_a, 2 * YMMRegister.size)
					if m + 1 != mr:
						CMP(reg_mr, m + 1)
						JE(load_a.end)

			with Block() as load_b:
				for n in range(nr):
					VMOVAPS(ymm_b_re[n], [reg_b])
					for m in range(mr):
						VFMADD231PS(ymm_c_re[m][n], ymm_a_re[m], ymm_b_re[n])

					VMOVAPS(ymm_b_im[n], [reg_b + YMMRegister.size])
					VBLENDPS(ymm_b_re[n], ymm_b_re[n], ymm_b_im[n], 0b00000011)
					for m in range(mr):
						VFMADD231PS(ymm_c_im[m][n], ymm_a_im[m], ymm_b_re[n])
					ADD(reg_b, YMMRegister.size * 2)

					if n + 1 != nr:
						CMP(reg_nr, n + 1)
						JE(load_b.end)

			ymm_zero_columns01_mask = YMMRegister()
			VMOVAPS(ymm_zero_columns01_mask, Constant.uint32x8(0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))

			for n in range(nr):
				VANDPS(ymm_b_im[n], ymm_b_im[n], ymm_zero_columns01_mask)
				for m in range(mr):
					if conjugate_b:
						VFMADD231PS(ymm_c_re[m][n], ymm_a_im[m], ymm_b_im[n])
						VFNMADD231PS(ymm_c_im[m][n], ymm_a_re[m], ymm_b_im[n])
					else:
						VFNMADD231PS(ymm_c_re[m][n], ymm_a_im[m], ymm_b_im[n])
						VFMADD231PS(ymm_c_im[m][n], ymm_a_re[m], ymm_b_im[n])

			DEC(reg_k)
			JNZ(loop.begin)

		store_c = Block()

		# Check if we need to update C or overwrite it
		TEST(reg_update, reg_update)
		JZ(store_c.begin)

		if transpose_c:
			mr, nr = nr, mr
			reg_mr, reg_nr = reg_nr, reg_mr
			ymm_c_re = [list(ymm_column) for ymm_column in zip(*ymm_c_re)]
			ymm_c_im = [list(ymm_column) for ymm_column in zip(*ymm_c_im)]

		with Block() as update_c:
			for m in range(mr):
				with Block() as update_c_row:
					for n in range(nr):
						VADDPS(ymm_c_re[m][n], ymm_c_re[m][n], [reg_c + (2*n+0) * YMMRegister.size])
						VADDPS(ymm_c_im[m][n], ymm_c_im[m][n], [reg_c + (2*n+1) * YMMRegister.size])
						VMOVAPS([reg_c + (2*n+0) * YMMRegister.size], ymm_c_re[m][n])
						VMOVAPS([reg_c + (2*n+1) * YMMRegister.size], ymm_c_im[m][n])

						if n + 1 != nr:
							CMP(reg_nr, n + 1)
							JE(update_c_row.end)

				if m + 1 != mr:
					CMP(reg_mr, m + 1)
					JE(update_c.end)

					ADD(reg_c, reg_row_stride)

		RETURN()

		with store_c:
			for m in range(mr):
				with Block() as store_c_row:
					for n in range(nr):
						VMOVAPS([reg_c + (2*n+0) * YMMRegister.size], ymm_c_re[m][n])
						VMOVAPS([reg_c + (2*n+1) * YMMRegister.size], ymm_c_im[m][n])

						if n + 1 != nr:
							CMP(reg_nr, n + 1)
							JE(store_c_row.end)

				if m + 1 != mr:
					CMP(reg_mr, m + 1)
					JE(store_c.end)

					ADD(reg_c, reg_row_stride)

		RETURN()
