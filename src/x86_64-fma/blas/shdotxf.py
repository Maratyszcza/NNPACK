from __future__ import absolute_import
from __future__ import division

from fp16.avx import fp16_alt_xmm_to_fp32_xmm
from fp16.avx2 import fp16_alt_xmm_to_fp32_ymm

simd_width = YMMRegister.size // float_.size

for fusion_factor in range(1, 8 + 1):
	arg_x = Argument(ptr(const_float_), "x")
	arg_y = Argument(ptr(const_float_), "y")
	arg_stride_y = Argument(size_t, "stride_y")
	arg_sum = Argument(ptr(float_), "sum")
	arg_n = Argument(size_t, "n")
	with Function("nnp_shdotxf{fusion_factor}__avx2".format(fusion_factor=fusion_factor),
		(arg_x, arg_y, arg_stride_y, arg_sum, arg_n),
		target=uarch.default + isa.fma3 + isa.avx2):

		reg_x = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_x, arg_x)

		reg_ys = [GeneralPurposeRegister64() for m in range(fusion_factor)]
		LOAD.ARGUMENT(reg_ys[0], arg_y)

		reg_stride_y = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_stride_y, arg_stride_y)
		ADD(reg_stride_y, reg_stride_y)

		reg_sum = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_sum, arg_sum)

		reg_n = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_n, arg_n)

		ymm_accs = [YMMRegister() for m in range(fusion_factor)]
		VZEROALL()

		for m in range(1, fusion_factor):
			LEA(reg_ys[m], [reg_ys[m - 1] + reg_stride_y * 1])

		main_loop = Loop()
		edge_loop = Loop()

		SUB(reg_n, XMMRegister.size // uint16_t.size)
		JB(main_loop.end)

		with main_loop:
			ymm_x = YMMRegister()
			VMOVUPS(ymm_x, [reg_x])
			ADD(reg_x, YMMRegister.size)

			for reg_y, ymm_acc in zip(reg_ys, ymm_accs):
				xmm_half = XMMRegister()
				VMOVUPS(xmm_half, [reg_y])
				ADD(reg_y, XMMRegister.size)

				ymm_y = fp16_alt_xmm_to_fp32_ymm(xmm_half)
				VFMADD231PS(ymm_acc, ymm_x, ymm_y)

			SUB(reg_n, YMMRegister.size // float_.size)
			JAE(main_loop.begin)

		ADD(reg_n, XMMRegister.size // uint16_t.size)
		JE(edge_loop.end)

		with edge_loop:
			xmm_x = XMMRegister()
			VMOVSS(xmm_x, [reg_x])
			ADD(reg_x, YMMRegister.size)

			for reg_y, ymm_acc in zip(reg_ys, ymm_accs):
				reg_half = GeneralPurposeRegister32()
				MOVZX(reg_half, word[reg_y])

				xmm_half = XMMRegister()
				VMOVD(xmm_half, reg_half)
				ADD(reg_y, uint16_t.size)

				ymm_y = fp16_alt_xmm_to_fp32_ymm(xmm_half)
				VFMADD231PS(ymm_acc, xmm_x.as_ymm, ymm_y)

			SUB(reg_n, 1)
			JAE(edge_loop.begin)

		# Reduce the SIMD registers into a single elements
		xmm_tmp = XMMRegister()
		for i, ymm_acc in enumerate(ymm_accs):
			VEXTRACTF128(xmm_tmp, ymm_acc, 1)
			VADDPS(ymm_acc.as_xmm, ymm_acc.as_xmm, xmm_tmp)
			VHADDPS(ymm_acc, ymm_acc, ymm_acc)
			VHADDPS(ymm_acc, ymm_acc, ymm_acc)
			VMOVSS([reg_sum + i * float_.size], ymm_acc.as_xmm)

		RETURN()

