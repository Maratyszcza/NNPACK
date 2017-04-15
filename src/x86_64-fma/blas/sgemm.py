from __future__ import absolute_import
from __future__ import division

from common import _MM_SHUFFLE

simd_width = YMMRegister.size // float_.size
mr = 4
nr = 3 * simd_width

arg_k = Argument(size_t, "k")
arg_update = Argument(size_t, "update")
arg_a = Argument(ptr(const_float_), "a")
arg_b = Argument(ptr(const_float_), "b")
arg_c = Argument(ptr(float_), "c")
arg_row_stride = Argument(size_t, "row_stride_c")
with Function("nnp_sgemm_only_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
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

	ymm_c = [[YMMRegister() for n in range(0, nr, simd_width)] for m in range(mr)]
	VZEROALL()

	ymm_b = [YMMRegister() for n in range(0, nr, simd_width)]
	ymm_a_m = YMMRegister()
	with Loop() as loop:
		for n in range(nr // simd_width):
			VMOVAPS(ymm_b[n], [reg_b + n * YMMRegister.size])
		ADD(reg_b, nr * float_.size)

		for m in range(mr):
			VBROADCASTSS(ymm_a_m, [reg_a + m * float_.size])
			for n in range(nr // simd_width):
				VFMADD231PS(ymm_c[m][n], ymm_a_m, ymm_b[n])
		ADD(reg_a, mr * float_.size)

		DEC(reg_k)
		JNE(loop.begin)

	store_c = Block()

	# Check if we need to update C or overwrite it
	TEST(reg_update, reg_update)
	JZ(store_c.begin)

	with Block() as load_and_store_c:
		for m in reversed(range(mr)):
			for n in range(nr // simd_width):
				VADDPS(ymm_c[m][n], ymm_c[m][n], [reg_c + n * YMMRegister.size])
				VMOVUPS([reg_c + n * YMMRegister.size], ymm_c[m][n])
			if m != 0:
				SUB(reg_c, reg_row_stride)

		RETURN()

	with store_c:
		for m in reversed(range(mr)):
			for n in range(nr // simd_width):
				VMOVUPS([reg_c + n * YMMRegister.size], ymm_c[m][n])
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
with Function("nnp_sgemm_upto_{mr}x{nr}__fma3".format(mr=mr, nr=nr),
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

	ymm_c = [[YMMRegister() for n in range(0, nr, simd_width)] for m in range(mr)]
	VZEROALL()

	ymm_b = [YMMRegister() for n in range(0, nr, simd_width)]
	ymm_a_m = YMMRegister()
	with Loop() as loop:
		with Block() as load_b:
			for n in range(nr // simd_width):
				VMOVAPS(ymm_b[n], [reg_b])
				ADD(reg_b, YMMRegister.size)
				if n + 1 != nr // simd_width:
					CMP(reg_nr, (n + 1) * simd_width)
					JBE(load_b.end)

		with Block() as multiply_by_a:
			for m in range(mr):
				VBROADCASTSS(ymm_a_m, [reg_a])
				ADD(reg_a, float_.size)
				for n in range(nr // simd_width):
					VFMADD231PS(ymm_c[m][n], ymm_a_m, ymm_b[n])

				if m + 1 != mr:
					CMP(reg_mr, m + 1)
					JE(multiply_by_a.end)

		DEC(reg_k)
		JNE(loop.begin)

	store_c = Block()

	# Load mask
	reg_mask_index = GeneralPurposeRegister32()
	LEA(reg_mask_index, [reg_nr.as_qword - 1])
	AND(reg_mask_index, simd_width - 1)
	NEG(reg_mask_index.as_qword)
	const_mask_table = Constant.uint32x16(*([0xFFFFFFFF] * 8 + [0x00000000] * 8))
	reg_mask = GeneralPurposeRegister64()
	LEA(reg_mask, const_mask_table)
	LEA(reg_mask, [reg_mask + reg_mask_index.as_qword * 4 + 32 - 4])
	ymm_mask = YMMRegister()
	VMOVDQU(ymm_mask, [reg_mask])

	# Check if we need to update C or overwrite it
	TEST(reg_update, reg_update)
	JZ(store_c.begin)

	with Block() as update_c:
		for m in range(mr):
			reg_c_mn = GeneralPurposeRegister64()
			MOV(reg_c_mn, reg_c)
			ymm_c_mn = YMMRegister()
			with Block() as update_c_full_registers:
				for n in range(nr // simd_width):
					# Copy the current accumulator register into a fixed register ymm_c_mn.
					# If a partial register is to be stored, the storing code would expect it there.
					VMOVAPS(ymm_c_mn, ymm_c[m][n])
					if n + 1 != nr // simd_width:
						CMP(reg_nr, (n + 1) * simd_width)
						JBE(update_c_full_registers.end)

						VADDPS(ymm_c[m][n], ymm_c[m][n], [reg_c_mn])
						VMOVUPS([reg_c_mn], ymm_c[m][n])
						ADD(reg_c_mn, YMMRegister.size)

			# Update (potentially) partial register
			# Important: ymm_c_mn is the content of the register and [reg_c_mn] is the address of the tuple of C
			ymm_temp = YMMRegister()
			VMASKMOVPS(ymm_temp, ymm_mask, [reg_c_mn])
			VADDPS(ymm_c_mn, ymm_c_mn, ymm_temp)
			VMASKMOVPS([reg_c_mn], ymm_mask, ymm_c_mn)

			if m + 1 != mr:
				CMP(reg_mr, m + 1)
				JE(update_c.end)

				ADD(reg_c, reg_row_stride)

	RETURN()

	with store_c:
		for m in range(mr):
			reg_c_mn = GeneralPurposeRegister64()
			MOV(reg_c_mn, reg_c)
			ymm_c_mn = YMMRegister()
			with Block() as store_c_full_registers:
				for n in range(nr // simd_width):
					# Copy the current accumulator register into a fixed register ymm_c_mn.
					# If a partial register is to be stored, the storing code would expect it there.
					VMOVAPS(ymm_c_mn, ymm_c[m][n])
					if n + 1 != nr // simd_width:
						CMP(reg_nr, (n + 1) * simd_width)
						JBE(store_c_full_registers.end)

						VMOVUPS([reg_c_mn], ymm_c[m][n])
						ADD(reg_c_mn, YMMRegister.size)

			# Store (potentially) partial register
			# Important: ymm_c_mn is the content of the register and [reg_c_mn] is the address of the tuple of C
			VMASKMOVPS([reg_c_mn], ymm_mask, ymm_c_mn)

			if m + 1 != mr:
				CMP(reg_mr, m + 1)
				JE(store_c.end)

				ADD(reg_c, reg_row_stride)

	RETURN()
