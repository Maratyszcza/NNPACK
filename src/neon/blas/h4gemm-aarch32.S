#include <nnpack/assembly.h>

# void nnp_h4gemm_only_3x3__aarch32_neonhp(
#        size_t k,
#        size_t update,
#        const __fp16* a,
#        const __fp16* b,
#        __fp16* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_h4gemm_only_3x3__aarch32_neonhp
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon-vfpv4
#endif
	VPUSH {d8-d15}

	# d7 := acc00
	VMOV.I16  q7, #0
	# d8 := acc01
	VMOV.I16  q8, #0
	# d9 := acc02
	VMOV.I16  q9, #0

	# d10 := acc10
	VMOV.I16 q10, #0
	# d11 := acc11
	VMOV.I16 q11, #0
	# d12 := acc12
	VMOV.I16 q12, #0

	# d13 := acc20
	VMOV.I16 q13, #0
	# d14 := acc21
	VMOV.I16 q14, #0
	# d15 := acc22
	VMOV.I16 q15, #0

	.align 4
0:
	# Load a0, a1, a2
	# - d0 = a0
	# - d1 = a1
	# - d2 = a2
	VLD1.16 {d0-d2}, [r2:64]!

	VCVT.F32.F16 q5, d0
	VCVT.F32.F16 q0, d1
	VCVT.F32.F16 q1, d2

	# Load b0, b1, b2
	# - d4 = b0
	# - d5 = b1
	# - d6 = b2
	VLD1.16 {d4-d6}, [r3:64]!

	VCVT.F32.F16 q4, d4
	VCVT.F32.F16 q2, d5
	VCVT.F32.F16 q3, d6

	VMLA.F32  q7, q5, q4
	VMLA.F32 q10, q0, q4
	VMLA.F32 q13, q1, q4

	VMLA.F32  q8, q5, q2
	VMLA.F32 q11, q0, q2
	VMLA.F32 q14, q1, q2

	VMLA.F32  q9, q5, q3
	VMLA.F32 q12, q0, q3
	VMLA.F32 q15, q1, q3

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes
	ADD r3, r3, r3
	# Skip to label 1 to update c
	BNE 1f

	##### Overwrite c matrix with results in acc[0:3][0:16]

	VCVT.F16.F32  d7,  q7
	VCVT.F16.F32  d8,  q8
	VCVT.F16.F32  d9,  q9
	VCVT.F16.F32 d10, q10
	VCVT.F16.F32 d11, q11
	VCVT.F16.F32 d12, q12
	VCVT.F16.F32 d13, q13
	VCVT.F16.F32 d14, q14
	VCVT.F16.F32 d15, q15

	# Overwrite c[0][0:12] = acc[0][0:12]
	VST1.16   {d7-d9}, [r2:64], r3

	# Overwrite c[1][0:12] = acc[1][0:12]
	VST1.16 {d10-d12}, [r2:64], r3

	# Overwrite c[2][0:12] = acc[2][0:12]
	VST1.16 {d13-d15}, [r2:64]

	VPOP {d8-d15}
	BX lr

1:
	##### Accumulate c matrix with results in acc[0:3][0:12]

	# Accumulate c[0][0:12] += acc[0][0:12]
	VLD1.16 {d0-d2}, [r2:64]
	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q3, d1
	VCVT.F32.F16 q4, d2
	VADD.F32 q2, q2, q7
	VADD.F32 q3, q3, q8
	VADD.F32 q4, q4, q9
	VCVT.F16.F32 d0, q2
	VCVT.F16.F32 d1, q3
	VCVT.F16.F32 d2, q4
	VST1.16 {d0-d2}, [r2:64], r3

	# Accumulate c[1][0:12] += acc[1][0:12]
	VLD1.32 {d0-d2}, [r2:64]
	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q3, d1
	VCVT.F32.F16 q4, d2
	VADD.F32 q2, q2, q10
	VADD.F32 q3, q3, q11
	VADD.F32 q4, q4, q12
	VCVT.F16.F32 d0, q2
	VCVT.F16.F32 d1, q3
	VCVT.F16.F32 d2, q4
	VST1.32 {d0-d2}, [r2:64], r3

	# Accumulate c[2][0:12] += acc[2][0:12]
	VLD1.32 {d0-d2}, [r2:64]
	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q3, d1
	VCVT.F32.F16 q4, d2
	VADD.F32 q2, q2, q13
	VADD.F32 q3, q3, q14
	VADD.F32 q4, q4, q15
	VCVT.F16.F32 d0, q2
	VCVT.F16.F32 d1, q3
	VCVT.F16.F32 d2, q4
	VST1.32 {d0-d2}, [r2:64]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_h4gemm_only_3x3__aarch32_neonhp

# void nnp_h4gemm_only_3x3__aarch32_neon2(
#        size_t k,
#        size_t update,
#        const __fp16* a,
#        const __fp16* b,
#        __fp16* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_h4gemm_only_3x3__aarch32_neon2
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon-vfpv4
#endif
	VPUSH {d8-d15}

	# d7 := acc00
	VMOV.I16  q7, #0
	# d8 := acc01
	VMOV.I16  q8, #0
	# d9 := acc02
	VMOV.I16  q9, #0

	# d10 := acc10
	VMOV.I16 q10, #0
	# d11 := acc11
	VMOV.I16 q11, #0
	# d12 := acc12
	VMOV.I16 q12, #0

	# d13 := acc20
	VMOV.I16 q13, #0
	# d14 := acc21
	VMOV.I16 q14, #0
	# d15 := acc22
	VMOV.I16 q15, #0

	.align 4
0:
	# Load a0, a1, a2
	# - d0 = a0
	# - d1 = a1
	# - d2 = a2
	VLDM r2!, {d0-d2}

	VCVT.F32.F16 q5, d0
	VCVT.F32.F16 q0, d1
	VCVT.F32.F16 q1, d2

	# Load b0, b1, b2
	# - d4 = b0
	# - d5 = b1
	# - d6 = b2
	VLDM r3!, {d4-d6}

	VCVT.F32.F16 q4, d4
	VCVT.F32.F16 q2, d5
	VCVT.F32.F16 q3, d6

	VFMA.F32  q7, q5, q4
	VFMA.F32 q10, q0, q4
	VFMA.F32 q13, q1, q4

	VFMA.F32  q8, q5, q2
	VFMA.F32 q11, q0, q2
	VFMA.F32 q14, q1, q2

	VFMA.F32  q9, q5, q3
	VFMA.F32 q12, q0, q3
	VFMA.F32 q15, q1, q3

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes
	ADD r3, r3, r3
	# Skip to label 1 to update c
	BNE 1f

	##### Overwrite c matrix with results in acc[0:3][0:12]

	VCVT.F16.F32  d7,  q7
	VCVT.F16.F32  d8,  q8
	VCVT.F16.F32  d9,  q9
	VCVT.F16.F32 d10, q10
	VCVT.F16.F32 d11, q11
	VCVT.F16.F32 d12, q12
	VCVT.F16.F32 d13, q13
	VCVT.F16.F32 d14, q14
	VCVT.F16.F32 d15, q15

	# Overwrite c[0][0:12] = acc[0][0:12]
	VST1.16   {d7-d9}, [r2:64], r3

	# Overwrite c[1][0:12] = acc[1][0:12]
	VST1.16 {d10-d12}, [r2:64], r3

	# Overwrite c[2][0:12] = acc[2][0:12]
	VST1.16 {d13-d15}, [r2:64]

	VPOP {d8-d15}
	BX lr

1:
	##### Accumulate c matrix with results in acc[0:3][0:12]

	# Accumulate c[0][0:12] += acc[0][0:12]
	VLDM r2, {d0-d2}
	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q3, d1
	VCVT.F32.F16 q4, d2
	VADD.F32 q2, q2, q7
	VADD.F32 q3, q3, q8
	VADD.F32 q4, q4, q9
	VCVT.F16.F32 d0, q2
	VCVT.F16.F32 d1, q3
	VCVT.F16.F32 d2, q4
	VST1.16 {d0-d2}, [r2:64], r3

	# Accumulate c[1][0:12] += acc[1][0:12]
	VLDM r2, {d0-d2}
	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q3, d1
	VCVT.F32.F16 q4, d2
	VADD.F32 q2, q2, q10
	VADD.F32 q3, q3, q11
	VADD.F32 q4, q4, q12
	VCVT.F16.F32 d0, q2
	VCVT.F16.F32 d1, q3
	VCVT.F16.F32 d2, q4
	VST1.32 {d0-d2}, [r2:64], r3

	# Accumulate c[2][0:12] += acc[2][0:12]
	VLDM r2, {d0-d2}
	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q3, d1
	VCVT.F32.F16 q4, d2
	VADD.F32 q2, q2, q13
	VADD.F32 q3, q3, q14
	VADD.F32 q4, q4, q15
	VCVT.F16.F32 d0, q2
	VCVT.F16.F32 d1, q3
	VCVT.F16.F32 d2, q4
	VST1.32 {d0-d2}, [r2:64]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_h4gemm_only_3x3__aarch32_neon2

# void nnp_h4gemm_upto_3x3__aarch32_neon2(
#        uint32_t mr,
#        uint32_t nr,
#        size_t k,
#        size_t update,
#        const __fp16* a,
#        const __fp16* b,
#        __fp16* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_h4gemm_upto_3x3__aarch32_neon2
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon-vfpv4
#endif
	PUSH {r4-r7}
	VPUSH {d8-d15}

	# Load a, b
	# - r4 = a
	# - r5 = b
	LDRD r4, r5, [sp, #80]

	ADD r6, r5, #8
	CMP r1, #2
	MOVLO r6, r5

	ADD r7, r6, #8
	MOVLS r7, r6

	LSL r1, r1, #3

	#  q7 := acc00
	VMOV.I16  q7, #0
	# q10 := acc01
	VMOV.I16 q10, #0
	# q13 := acc02
	VMOV.I16 q13, #0

	# mr <=> 2
	CMP r0, #2
	BHS 4f

	.align 4
0:
	##### Main loop (mr == 1)

	# Load a0
	# - d0 = a0
	VLD1.16 {d0}, [r4]!

	# Load b0
	# - d4 = b0
	VLD1.16 {d2}, [r5], r1

	# Load b1
	# - d5 = b1
	VLD1.16 {d4}, [r6], r1

	# Load b2
	# - d6 = b2
	VLD1.16 {d6}, [r7], r1

	VCVT.F32.F16 q0, d0

	VCVT.F32.F16 q1, d2
	VCVT.F32.F16 q2, d4
	VCVT.F32.F16 q3, d6

	# acc00 = vfmaq_f32(acc00, a0, b0);
	VFMA.F32  q7, q1, q0
	# acc01 = vfmaq_f32(acc01, a0, b1);
	VFMA.F32 q10, q2, q0
	# acc02 = vfmaq_f32(acc02, a0, b2);
	VFMA.F32 q13, q3, q0

	SUBS r2, r2, #1
	BNE 0b

	# Load argument c:
	# - r2 = c
	LDR r2, [sp, #88]
	# Check if c is updated (r3 != 0) or overwritten (r3 == 0)
	TEQ r3, #0
	# Skip to label 1 to update c
	BNE 2f

	##### Overwrite c matrix (mr == 1)

	VCVT.F16.F32 d14, q7
	VST1.16 {d14}, [r2]!

	# nr >= 2?
	CMP r1, #16
	BLO 1f

	VCVT.F16.F32 d20, q10
	VST1.16 {d20}, [r2]!

	BLS 1f

	VCVT.F16.F32 d26, q13
	VST1.16 {d26}, [r2]

1:
	VPOP {d8-d15}
	POP {r4-r7}
	BX lr

	##### Accumulate to c matrix (mr == 1)
2:

	VLD1.16 {d0}, [r2]
	VCVT.F32.F16 q0, d0
	VADD.F32 q7, q7, q0
	VCVT.F16.F32 d14, q7
	VST1.16 {d14}, [r2]!

	# nr >= 2?
	CMP r1, #16
	BLO 3f

	VLD1.16 {d0}, [r2]
	VCVT.F32.F16 q0, d0
	VADD.F32 q10, q10, q0
	VCVT.F16.F32 d20, q10
	VST1.16 {d20}, [r2]!

	BLS 3f

	VLD1.16 {d0}, [r2]
	VCVT.F32.F16 q0, d0
	VADD.F32 q13, q13, q0
	VCVT.F16.F32 d26, q13
	VST1.16 {d26}, [r2]

	VPOP {d8-d15}
	POP {r4-r7}
	BX lr

3:
	VPOP {d8-d15}
	POP {r4-r7}
	BX lr

	.align 3
4:
	##### Initialization (mr == 2)

	#  q8 := acc10
	VMOV.I16  q8, #0
	# q11 := acc11
	VMOV.I16 q11, #0
	# q14 := acc12
	VMOV.I16 q14, #0

	BHI 9f

5:
	##### Main loop (mr == 2)

	# Load a0, a1
	# - d0 = a0
	# - d1 = a1
	VLDM r4!, {d0-d1}

	# Load b0
	# - d4 = b0
	VLD1.16 {d4}, [r5], r1

	# Load b1
	# - d6 = b1
	VLD1.16 {d6}, [r6], r1

	# Load b2
	# - d8 = b2
	VLD1.16 {d8}, [r7], r1

	VCVT.F32.F16 q1, d0
	VCVT.F32.F16 q0, d1

	VCVT.F32.F16 q2, d4
	VCVT.F32.F16 q3, d6
	VCVT.F32.F16 q4, d8

	# acc00 = vfmaq_f32(acc00, a0, b0);
	VFMA.F32  q7, q2, q1
	# acc01 = vfmaq_f32(acc01, a0, b1);
	VFMA.F32 q10, q3, q1
	# acc02 = vfmaq_f32(acc02, a0, b2);
	VFMA.F32 q13, q4, q1

	# acc10 = vfmaq_f32(acc10, a1, b0);
	VFMA.F32  q8, q2, q0
	# acc11 = vfmaq_f32(acc11, a1, b1);
	VFMA.F32 q11, q3, q0
	# acc12 = vfmaq_f32(acc12, a1, b2);
	VFMA.F32 q14, q4, q0

	SUBS r2, r2, #1
	BNE 5b

	# Load argument c, row_stride_c:
	# - r4 = c
	# - r5 = row_stride_c
	LDRD r4, r5, [sp, #88]
	# Check if c is updated (r3 != 0) or overwritten (r3 == 0)
	TEQ r3, #0
	# Set crow0, crow1
	# - r4 = crow0
	# - r5 = crow1
	ADD r5, r4, r5, LSL #1
	# Skip to label 1 to update c
	BNE 7f

	##### Overwrite c matrix (mr == 2)

	VCVT.F16.F32 d14, q7
	VCVT.F16.F32 d16, q8
	VST1.16 {d14}, [r4]!
	VST1.16 {d16}, [r5]!

	# nr >= 2?
	CMP r1, #16
	BLO 6f

	VCVT.F16.F32 d20, q10
	VCVT.F16.F32 d22, q11
	VST1.16 {d20}, [r4]!
	VST1.16 {d22}, [r5]!

	BLS 6f

	VCVT.F16.F32 d26, q13
	VCVT.F16.F32 d28, q14
	VST1.16 {d26}, [r4]
	VST1.16 {d28}, [r5]

6:
	VPOP {d8-d15}
	POP {r4-r7}
	BX lr

	##### Accumulate to c matrix (mr == 2)
7:
	VLD1.16 {d0}, [r4]
	VLD1.16 {d2}, [r5]
	VCVT.F32.F16 q0, d0
	VCVT.F32.F16 q1, d2
	VADD.F32 q7, q7, q0
	VADD.F32 q8, q8, q1
	VCVT.F16.F32 d14, q7
	VCVT.F16.F32 d16, q8
	VST1.16 {d14}, [r4]!
	VST1.16 {d16}, [r5]!

	# nr >= 2?
	CMP r1, #16
	BLO 8f

	VLD1.16 {d0}, [r4]
	VLD1.16 {d2}, [r5]
	VCVT.F32.F16 q0, d0
	VCVT.F32.F16 q1, d2
	VADD.F32 q10, q10, q0
	VADD.F32 q11, q11, q1
	VCVT.F16.F32 d20, q10
	VCVT.F16.F32 d22, q11
	VST1.16 {d20}, [r4]!
	VST1.16 {d22}, [r5]!

	BLS 8f

	VLD1.16 {d0}, [r4]
	VLD1.16 {d2}, [r5]
	VCVT.F32.F16 q0, d0
	VCVT.F32.F16 q1, d2
	VADD.F32 q13, q13, q0
	VADD.F32 q14, q14, q1
	VCVT.F16.F32 d26, q13
	VCVT.F16.F32 d28, q14
	VST1.16 {d26}, [r4]
	VST1.16 {d28}, [r5]

8:
	VPOP {d8-d15}
	POP {r4-r7}
	BX lr

	##### Initialization (mr == 3)
9:

	#  q9 := acc20
	VMOV.I16  q9, #0
	# q12 := acc21
	VMOV.I16 q12, #0
	# q15 := acc22
	VMOV.I16 q15, #0

	.align 4
10:
	# Load a0, a1, a2
	# - d0 = a0
	# - d1 = a1
	# - d2 = a2
	VLDM r4!, {d0-d2}

	# Load b0
	# - d6 = b0
	VLD1.16 {d6}, [r5], r1

	# Load b1
	# - d8 = b1
	VLD1.16 {d8}, [r6], r1

	# Load b2
	# - d10 = b2
	VLD1.16 {d10}, [r7], r1

	VCVT.F32.F16 q2, d0
	VCVT.F32.F16 q0, d1
	VCVT.F32.F16 q1, d2

	VCVT.F32.F16 q3, d6
	VCVT.F32.F16 q4, d8
	VCVT.F32.F16 q5, d10

	# acc00 = vfmaq_f32(acc00, a0, b0);
	VFMA.F32  q7, q3, q2
	# acc01 = vfmaq_f32(acc01, a0, b1);
	VFMA.F32 q10, q4, q2
	# acc02 = vfmaq_f32(acc02, a0, b2);
	VFMA.F32 q13, q5, q2

	# acc10 = vfmaq_f32(acc10, a1, b0);
	VFMA.F32  q8, q3, q0
	# acc11 = vfmaq_f32(acc11, a1, b1);
	VFMA.F32 q11, q4, q0
	# acc12 = vfmaq_f32(acc12, a1, b2);
	VFMA.F32 q14, q5, q0

	# acc20 = vfmaq_f32(acc20, a2, b0);
	VFMA.F32  q9, q3, q1
	# acc21 = vfmaq_f32(acc21, a2, b1);
	VFMA.F32 q12, q4, q1
	# acc22 = vfmaq_f32(acc22, a2, b2);
	VFMA.F32 q15, q5, q1

	SUBS r2, r2, #1
	BNE 10b

	# Load argument c, row_stride_c:
	# - r4 = c
	# - r5 = row_stride_c
	LDRD r4, r5, [sp, #88]
	# Check if c is updated (r3 != 0) or overwritten (r3 == 0)
	TEQ r3, #0
	# Set crow0, crow1, crow2
	# - r4 = crow0
	# - r5 = crow1
	# - r6 = crow1
	ADD r6, r4, r5, LSL #2
	ADD r5, r4, r5, LSL #1
	# Skip to label 1 to update c
	BNE 12f

	##### Overwrite c matrix (mr == 3)

	VCVT.F16.F32 d14, q7
	VCVT.F16.F32 d16, q8
	VCVT.F16.F32 d18, q9
	VST1.16 {d14}, [r4]!
	VST1.16 {d16}, [r5]!
	VST1.16 {d18}, [r6]!

	# nr >= 2?
	CMP r1, #16
	BLO 11f

	VCVT.F16.F32 d20, q10
	VCVT.F16.F32 d22, q11
	VCVT.F16.F32 d24, q12
	VST1.16 {d20}, [r4]!
	VST1.16 {d22}, [r5]!
	VST1.16 {d24}, [r6]!

	BLS 11f

	VCVT.F16.F32 d26, q13
	VCVT.F16.F32 d28, q14
	VCVT.F16.F32 d30, q15
	VST1.16 {d26}, [r4]
	VST1.16 {d28}, [r5]
	VST1.16 {d30}, [r6]

11:
	VPOP {d8-d15}
	POP {r4-r7}
	BX lr

	##### Accumulate to c matrix (mr == 3)
12:

	VLD1.16 {d0}, [r4]
	VLD1.16 {d2}, [r5]
	VLD1.16 {d4}, [r6]
	VCVT.F32.F16 q0, d0
	VCVT.F32.F16 q1, d2
	VCVT.F32.F16 q2, d4
	VADD.F32 q7, q7, q0
	VADD.F32 q8, q8, q1
	VADD.F32 q9, q9, q2
	VCVT.F16.F32 d14, q7
	VCVT.F16.F32 d16, q8
	VCVT.F16.F32 d18, q9
	VST1.16 {d14}, [r4]!
	VST1.16 {d16}, [r5]!
	VST1.16 {d18}, [r6]!

	# nr >= 2?
	CMP r1, #16
	BLO 13f

	VLD1.16 {d0}, [r4]
	VLD1.16 {d2}, [r5]
	VLD1.16 {d4}, [r6]
	VCVT.F32.F16 q0, d0
	VCVT.F32.F16 q1, d2
	VCVT.F32.F16 q2, d4
	VADD.F32 q10, q10, q0
	VADD.F32 q11, q11, q1
	VADD.F32 q12, q12, q2
	VCVT.F16.F32 d20, q10
	VCVT.F16.F32 d22, q11
	VCVT.F16.F32 d24, q12
	VST1.16 {d20}, [r4]!
	VST1.16 {d22}, [r5]!
	VST1.16 {d24}, [r6]!

	BLS 13f

	VLD1.16 {d0}, [r4]
	VLD1.16 {d2}, [r5]
	VLD1.16 {d4}, [r6]
	VCVT.F32.F16 q0, d0
	VCVT.F32.F16 q1, d2
	VCVT.F32.F16 q2, d4
	VADD.F32 q13, q13, q0
	VADD.F32 q14, q14, q1
	VADD.F32 q15, q15, q2
	VCVT.F16.F32 d26, q13
	VCVT.F16.F32 d28, q14
	VCVT.F16.F32 d30, q15
	VST1.16 {d26}, [r4]
	VST1.16 {d28}, [r5]
	VST1.16 {d30}, [r6]

13:
	VPOP {d8-d15}
	POP {r4-r7}
	BX lr
END_FUNCTION nnp_h4gemm_upto_3x3__aarch32_neon2

# void nnp_h4gemm_only_3x3__aarch32_neonhparith(
#        size_t k,
#        size_t update,
#        const __fp16* a,
#        const __fp16* b,
#        __fp16* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_h4gemm_only_3x3__aarch32_neonhparith
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon
#endif
	VPUSH {d8-d15}

	# d7 := acc00
	VMOV.I16  d7, #0
	# d8 := acc01
	VMOV.I16  d8, #0
	# d9 := acc02
	VMOV.I16  d9, #0

	# d10 := acc10
	VMOV.I16 d10, #0
	# d11 := acc11
	VMOV.I16 d11, #0
	# d12 := acc12
	VMOV.I16 d12, #0

	# d13 := acc20
	VMOV.I16 d13, #0
	# d14 := acc21
	VMOV.I16 d14, #0
	# d15 := acc22
	VMOV.I16 d15, #0

	.align 4
0:
	# Load a0, a1, a2
	# - d0 = a0
	# - d1 = a1
	# - d2 = a2
	VLD1.16 {d0-d2}, [r2:64]!

	# Load b0, b1, b2
	# - d4 = b0
	# - d5 = b1
	# - d6 = b2
	VLD1.16 {d4-d6}, [r3:64]!

	# VFMA.F16  d7, d0, d4
	.word 0xF2107C14
	# VFMA.F16 d10, d1, d4
	.word 0xF211AC14
	# VFMA.F16 d13, d2, d4
	.word 0xF212DC14

	# VFMA.F16  d8, d0, d5
	.word 0xF2108C15
	# VFMA.F16 d11, d1, d5
	.word 0xF211BC15
	# VFMA.F16 d14, d2, d5
	.word 0xF212EC15

	# VFMA.F16  d9, d0, d6
	.word 0xF2109C16
	# VFMA.F16 d12, d1, d6
	.word 0xF211CC16
	# VFMA.F16 d15, d2, d6
	.word 0xF212FC16

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes
	ADD r3, r3, r3
	# Skip to label 1 to update c
	BNE 1f

	##### Overwrite c matrix with results in acc[0:3][0:16]

	# Overwrite c[0][0:12] = acc[0][0:12]
	VST1.16   {d7-d9}, [r2:64], r3

	# Overwrite c[1][0:12] = acc[1][0:12]
	VST1.16 {d10-d12}, [r2:64], r3

	# Overwrite c[2][0:12] = acc[2][0:12]
	VST1.16 {d13-d15}, [r2:64]

	VPOP {d8-d15}
	BX lr

1:
	##### Accumulate c matrix with results in acc[0:3][0:12]

	# Accumulate c[0][0:12] += acc[0][0:12]
	VLD1.16 {d0-d2}, [r2:64]
	# VADD.F16 d0, d0, d7
	.word 0xF2100D07
	# VADD.F16 d1, d1, d8
	.word 0xF2111D08
	# VADD.F16 d2, d2, d9
	.word 0xF2122D09
	VST1.16 {d0-d2}, [r2:64], r3

	# Accumulate c[1][0:12] += acc[1][0:12]
	VLD1.32 {d0-d2}, [r2:64]
	# VADD.F16 d0, d0, d10
	.word 0xF2100D0A
	# VADD.F16 d1, d1, d11
	.word 0xF2111D0B
	# VADD.F16 d2, d2, d12
	.word 0xF2122D0C
	VST1.32 {d0-d2}, [r2:64], r3

	# Accumulate c[2][0:12] += acc[2][0:12]
	VLD1.32 {d0-d2}, [r2:64]
	# VADD.F16 d0, d0, d13
	.word 0xF2100D0D
	# VADD.F16 d1, d1, d14
	.word 0xF2111D0E
	# VADD.F16 d2, d2, d15
	.word 0xF2122D0F
	VST1.32 {d0-d2}, [r2:64]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_h4gemm_only_3x3__aarch32_neonhparith

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
