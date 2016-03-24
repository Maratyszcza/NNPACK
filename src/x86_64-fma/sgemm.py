simd_width = YMMRegister.size / float_.size
mr_max = 4
nr_max = 3 * simd_width

arg_k = Argument(size_t, "k")
arg_k_block_number = Argument(size_t, "k_block_number")
arg_a = Argument(ptr(const_float_), "a")
arg_b = Argument(ptr(const_float_), "b")
arg_c = Argument(ptr(float_), "c")
arg_row_stride = Argument(size_t, "row_stride_c")
arg_col_mask = Argument(ptr(const_float_), "col_mask")
for mr in range (1, mr_max + 1):
	for nr in range(simd_width, nr_max + simd_width, simd_width):
		with Function("nnp_sgemm_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
			(arg_k, arg_k_block_number, arg_a, arg_b, arg_c, arg_row_stride, arg_col_mask),
			target=uarch.default + isa.fma3):

			reg_k = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_k, arg_k)

			reg_k_block_number = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_k_block_number, arg_k_block_number)

			reg_a = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_a, arg_a)

			reg_b = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_b, arg_b)

			reg_c = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_c, arg_c)

			reg_row_stride = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_row_stride, arg_row_stride)
			SHL(reg_row_stride, 2)

			reg_col_mask = GeneralPurposeRegister64()
			LOAD.ARGUMENT(reg_col_mask, arg_col_mask)

			assert nr % simd_width == 0

			ymm_c = [[YMMRegister() for n in range(nr // simd_width)] for m in range(mr)]
			ymm_b = [YMMRegister() for n in range(nr // simd_width)]
			ymm_a_m = YMMRegister()

			with Block() as prefetch_c:
				for m in range(mr):
					PREFETCHT0([reg_c])
					if m + 1 != mr:
						ADD(reg_c, reg_row_stride)

			VZEROALL()

			with Loop() as loop:
				for n in range(nr // simd_width):
					VMOVAPS(ymm_b[n], [reg_b + n * YMMRegister.size])

				for m in range(mr):
					VBROADCASTSS(ymm_a_m, [reg_a + m * float_.size])
					for n in range(nr // simd_width):
						VFMADD231PS(ymm_c[m][n], ymm_a_m, ymm_b[n])

				ADD(reg_a, mr_max * float_.size)
				ADD(reg_b, nr_max * float_.size)

				DEC(reg_k)
				JNE(loop.begin)

			store_c = Block()

			ymm_col_mask = YMMRegister()
			VMOVDQU(ymm_col_mask, [reg_col_mask])

			TEST(reg_k_block_number, reg_k_block_number)
			JZ(store_c.begin)

			with Block() as load_and_store_c:
				for m in reversed(range(mr)):
					for n in range(nr // simd_width):
						if n + 1 != nr // simd_width:
							VADDPS(ymm_c[m][n], ymm_c[m][n], [reg_c + n * YMMRegister.size])
							VMOVUPS([reg_c + n * YMMRegister.size], ymm_c[m][n])
						else:
							ymm_c_mn_old = YMMRegister()
							VMASKMOVPS(ymm_c_mn_old, ymm_col_mask, [reg_c + n * YMMRegister.size])
							VADDPS(ymm_c[m][n], ymm_c[m][n], ymm_c_mn_old)
							VMASKMOVPS([reg_c + n * YMMRegister.size], ymm_col_mask, ymm_c[m][n])
					if m != 0:
						SUB(reg_c, reg_row_stride)

				RETURN()

			with store_c:
				for m in reversed(range(mr)):
					for n in range(nr // simd_width):
						if n + 1 != nr // simd_width:
							VMOVUPS([reg_c + n * YMMRegister.size], ymm_c[m][n])
						else:
							VMASKMOVPS([reg_c + n * YMMRegister.size], ymm_col_mask, ymm_c[m][n])
					if m != 0:
						SUB(reg_c, reg_row_stride)

				RETURN()
