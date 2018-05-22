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
END_FUNCTION nnp_h4gemm_only_3x3__aarch32_neon2

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
