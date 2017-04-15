from __future__ import absolute_import
from __future__ import division

mr, nr = 3, 4

arg_k = Argument(size_t, "k")
arg_update = Argument(size_t, "update")
arg_a = Argument(ptr(const_float_), "a")
arg_b = Argument(ptr(const_float_), "b")
arg_c = Argument(ptr(float_), "c")
arg_row_stride = Argument(size_t, "row_stride_c")
with Function("nnp_s8gemm_only_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
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
		for m in range(mr):
			PREFETCHT0([reg_c])
			if m + 1 != mr:
				ADD(reg_c, reg_row_stride)

	ymm_c = [[YMMRegister() for n in range(nr)] for m in range(mr)]
	VZEROALL()

	ymm_a = [YMMRegister() for m in range(mr)]
	ymm_b_n = YMMRegister()
	with Loop() as loop:
		for m in range(mr):
			VMOVAPS(ymm_a[m], [reg_a + m * YMMRegister.size])
		SUB(reg_a, -mr * YMMRegister.size)

		for n in range(nr):
			VMOVAPS(ymm_b_n, [reg_b + n * YMMRegister.size])
			for m in range(mr):
				VFMADD231PS(ymm_c[m][n], ymm_a[m], ymm_b_n)
		SUB(reg_b, -nr * YMMRegister.size)

		DEC(reg_k)
		JNZ(loop.begin)

	store_c = Block()

	# Check if we need to update C or overwrite it
	TEST(reg_update, reg_update)
	JZ(store_c.begin)

	with Block() as update_c:
		for m in reversed(range(mr)):
			for n in range(nr):
				VADDPS(ymm_c[m][n], ymm_c[m][n], [reg_c + n * YMMRegister.size])
				VMOVAPS([reg_c + n * YMMRegister.size], ymm_c[m][n])
			if m != 0:
				SUB(reg_c, reg_row_stride)

	RETURN()

	with store_c:
		for m in reversed(range(mr)):
			for n in range(nr):
				VMOVAPS([reg_c + n * YMMRegister.size], ymm_c[m][n])
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
with Function("nnp_s8gemm_upto_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
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

	ymm_c = [[YMMRegister() for n in range(nr)] for m in range(mr)]
	VZEROALL()

	ymm_a = [YMMRegister() for m in range(mr)]
	ymm_b_n = YMMRegister()
	with Loop() as loop:
		with Block() as load_a:
			for m in range(mr):
				VMOVAPS(ymm_a[m], [reg_a])
				ADD(reg_a, YMMRegister.size)
				if m + 1 != mr:
					CMP(reg_mr, m + 1)
					JE(load_a.end)

		with Block() as load_b:
			for n in range(nr):
				VMOVAPS(ymm_b_n, [reg_b])
				ADD(reg_b, YMMRegister.size)
				for m in range(mr):
					VFMADD231PS(ymm_c[m][n], ymm_a[m], ymm_b_n)

				if n + 1 != nr:
					CMP(reg_nr, n + 1)
					JE(load_b.end)

		DEC(reg_k)
		JNE(loop.begin)

	store_c = Block()

	# Check if we need to update C or overwrite it
	TEST(reg_update, reg_update)
	JZ(store_c.begin)

	with Block() as update_c:
		for m in range(mr):
			with Block() as update_c_row:
				for n in range(nr):
					VADDPS(ymm_c[m][n], ymm_c[m][n], [reg_c + n * YMMRegister.size])
					VMOVAPS([reg_c + n * YMMRegister.size], ymm_c[m][n])

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
					VMOVAPS([reg_c + n * YMMRegister.size], ymm_c[m][n])

					if n + 1 != nr:
						CMP(reg_nr, n + 1)
						JE(store_c_row.end)

			if m + 1 != mr:
				CMP(reg_mr, m + 1)
				JE(store_c.end)

				ADD(reg_c, reg_row_stride)

	RETURN()
