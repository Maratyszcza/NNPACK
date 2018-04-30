#include <nnpack/assembly.h>

# void nnp_s4gemm_only_3x3__aarch32_neon(
#        size_t k,
#        size_t update,
#        const float* a,
#        const float* b,
#        float* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_s4gemm_only_3x3__aarch32_neon
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon
#endif

	VPUSH {d8-d15}

	# q4 := acc00
	VMOV.I32  q7, #0
	# q5 := acc01
	VMOV.I32  q8, #0
	# q6 := acc02
	VMOV.I32  q9, #0

	#  q8 := acc10
	VMOV.I32 q10, #0
	#  q9 := acc11
	VMOV.I32 q11, #0
	# q10 := acc12
	VMOV.I32 q12, #0

	# q12 := acc20
	VMOV.I32 q13, #0
	# q13 := acc21
	VMOV.I32 q14, #0
	# q14 := acc22
	VMOV.I32 q15, #0

	.align 4
0:
	# Load a0, a1
	# - q0 = a0
	# - q1 = a1
	VLD1.32 {d0-d3}, [r2:128]!

	# Load b0
	# - q3 = b0
	VLD1.32 {d6-d7}, [r3:128]!

	# Load a2
	# - q2 = a2
	VLD1.32 {d4-d5}, [r2:128]!

	VMLA.F32  q7, q0, q3

	# Load b1, b2
	# - q4 = b1
	# - q5 = b2
	VLD1.32 {d8-d11}, [r3:128]!

	VMLA.F32 q10, q1, q3
	VMLA.F32 q13, q2, q3

	VMLA.F32  q8, q0, q4
	VMLA.F32 q11, q1, q4
	VMLA.F32 q14, q2, q4

	VMLA.F32  q9, q0, q5
	VMLA.F32 q12, q1, q5
	VMLA.F32 q15, q2, q5

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	MOV ip, #-32
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes - 32
	ADD r3, ip, r3, LSL #2
	# Skip to label 1 to update c
	BNE 1f

	##### Overwrite c matrix with results in acc[0:3][0:16]

	# Overwrite c[0][0:16] = acc[0][0:16]
	VST1.32 {d14-d17}, [r2:128]!
	VST1.32 {d18-d19}, [r2:128], r3

	# Overwrite c[1][0:8] = acc[1][0:8]
	VST1.32 {d20-d23}, [r2:128]!
	VST1.32 {d24-d25}, [r2:128], r3

	# Overwrite c[2][0:8] = acc[2][0:8]
	VST1.32 {d26-d29}, [r2:128]!
	VST1.32 {d30-d31}, [r2:128]

	VPOP {d8-d15}
	BX lr

1:
	##### Accumulate c matrix with results in acc[0:6][0:8]

	# Accumulate c[0][0:16] += acc[0][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 q0, q0, q7
	VADD.F32 q1, q1, q8
	VADD.F32 q2, q2, q9
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128], r3

	# Accumulate c[1][0:16] += acc[1][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 q0, q0, q10
	VADD.F32 q1, q1, q11
	VADD.F32 q2, q2, q12
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128], r3

	# Accumulate c[1][0:16] += acc[1][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 q0, q0, q13
	VADD.F32 q1, q1, q14
	VADD.F32 q2, q2, q15
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_s4gemm_only_3x3__aarch32_neon

# void nnp_s4gemm_only_3x3__aarch32_neon2(
#        size_t k,
#        size_t update,
#        const float* a,
#        const float* b,
#        float* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_s4gemm_only_3x3__aarch32_neon2
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon-vfpv4
#endif

	VPUSH {d8-d15}

	# q4 := acc00
	VMOV.I32  q7, #0
	# q5 := acc01
	VMOV.I32  q8, #0
	# q6 := acc02
	VMOV.I32  q9, #0

	#  q8 := acc10
	VMOV.I32 q10, #0
	#  q9 := acc11
	VMOV.I32 q11, #0
	# q10 := acc12
	VMOV.I32 q12, #0

	# q12 := acc20
	VMOV.I32 q13, #0
	# q13 := acc21
	VMOV.I32 q14, #0
	# q14 := acc22
	VMOV.I32 q15, #0

	.align 4
0:
	# Load a0, a1
	# - q0 = a0
	# - q1 = a1
	VLD1.32 {d0-d3}, [r2:128]!

	# Load b0
	# - q3 = b0
	VLD1.32 {d6-d7}, [r3:128]!

	# Load a2
	# - q2 = a2
	VLD1.32 {d4-d5}, [r2:128]!

	VFMA.F32  q7, q0, q3

	# Load b1, b2
	# - q4 = b1
	# - q5 = b2
	VLD1.32 {d8-d11}, [r3:128]!

	VFMA.F32 q10, q1, q3
	VFMA.F32 q13, q2, q3

	VFMA.F32  q8, q0, q4
	VFMA.F32 q11, q1, q4
	VFMA.F32 q14, q2, q4

	VFMA.F32  q9, q0, q5
	VFMA.F32 q12, q1, q5
	VFMA.F32 q15, q2, q5

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	MOV ip, #-32
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes - 32
	ADD r3, ip, r3, LSL #2
	# Skip to label 1 to update c
	BNE 1f

	##### Overwrite c matrix with results in acc[0:3][0:16]

	# Overwrite c[0][0:16] = acc[0][0:16]
	VST1.32 {d14-d17}, [r2:128]!
	VST1.32 {d18-d19}, [r2:128], r3

	# Overwrite c[1][0:8] = acc[1][0:8]
	VST1.32 {d20-d23}, [r2:128]!
	VST1.32 {d24-d25}, [r2:128], r3

	# Overwrite c[2][0:8] = acc[2][0:8]
	VST1.32 {d26-d29}, [r2:128]!
	VST1.32 {d30-d31}, [r2:128]

	VPOP {d8-d15}
	BX lr

1:
	##### Accumulate c matrix with results in acc[0:6][0:8]

	# Accumulate c[0][0:16] += acc[0][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 q0, q0, q7
	VADD.F32 q1, q1, q8
	VADD.F32 q2, q2, q9
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128], r3

	# Accumulate c[1][0:16] += acc[1][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 q0, q0, q10
	VADD.F32 q1, q1, q11
	VADD.F32 q2, q2, q12
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128], r3

	# Accumulate c[1][0:16] += acc[1][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 q0, q0, q13
	VADD.F32 q1, q1, q14
	VADD.F32 q2, q2, q15
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_s4gemm_only_3x3__aarch32_neon2

# void nnp_s4gemm_only_3x3__aarch32_meerkat(
#        size_t k,
#        size_t update,
#        const float* a,
#        const float* b,
#        float* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_s4gemm_only_3x3__aarch32_meerkat
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon-vfpv4
#endif

	VPUSH {d8-d15}

	# q7 := acc00
	VMOV.I32  q7, #0
	# q8 := acc01
	VMOV.I32  q8, #0
	# q9 := acc02
	VMOV.I32  q9, #0

	# q10 := acc10
	VMOV.I32 q10, #0
	# q11 := acc11
	VMOV.I32 q11, #0
	# q12 := acc12
	VMOV.I32 q12, #0

	# q13 := acc20
	VMOV.I32 q13, #0
	# q14 := acc21
	VMOV.I32 q14, #0
	# q15 := acc22
	VMOV.I32 q15, #0

	.align 4
0:
	# Load a0, a1
	# - q0 = a0
	# - q1 = a1
	VLD1.32 {d0-d3}, [r2:128]!

	# Load b0
	# - q3 = b0
	VLD1.32 {d6-d7}, [r3:128]!

	# Load a2
	# - q2 = a2
	VLD1.32 {d4-d5}, [r2:128]!

	VFMA.F32 d14, d0, d6
	VFMA.F32 d15, d1, d7

	# Load b1, b2
	# - q4 = b1
	# - q5 = b2
	VLD1.32 {d8-d11}, [r3:128]!

	VFMA.F32 d20, d2, d6
	VFMA.F32 d21, d3, d7
	VFMA.F32 d26, d4, d6
	VFMA.F32 d27, d5, d7

	VFMA.F32 d16, d0, d8
	VFMA.F32 d17, d1, d9
	VFMA.F32 d22, d2, d8
	VFMA.F32 d23, d3, d9
	VFMA.F32 d28, d4, d8
	VFMA.F32 d29, d5, d9

	VFMA.F32 d18, d0, d10
	VFMA.F32 d19, d1, d11
	VFMA.F32 d24, d2, d10
	VFMA.F32 d25, d3, d11
	VFMA.F32 d30, d4, d10
	VFMA.F32 d31, d5, d11

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	MOV ip, #-32
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes - 32
	ADD r3, ip, r3, LSL #2
	# Skip to label 1 to update c
	BNE 1f

	##### Overwrite c matrix with results in acc[0:3][0:16]

	# Overwrite c[0][0:16] = acc[0][0:16]
	VST1.32 {d14-d17}, [r2:128]!
	VST1.32 {d18-d19}, [r2:128], r3

	# Overwrite c[1][0:8] = acc[1][0:8]
	VST1.32 {d20-d23}, [r2:128]!
	VST1.32 {d24-d25}, [r2:128], r3

	# Overwrite c[2][0:8] = acc[2][0:8]
	VST1.32 {d26-d29}, [r2:128]!
	VST1.32 {d30-d31}, [r2:128]

	VPOP {d8-d15}
	BX lr

1:
	##### Accumulate c matrix with results in acc[0:6][0:8]

	# Accumulate c[0][0:16] += acc[0][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 d0, d0, d14
	VADD.F32 d1, d1, d15
	VADD.F32 d2, d2, d16
	VADD.F32 d3, d3, d17
	VADD.F32 d4, d4, d18
	VADD.F32 d5, d5, d19
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128], r3

	# Accumulate c[1][0:16] += acc[1][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 d0, d0, d20
	VADD.F32 d1, d1, d21
	VADD.F32 d2, d2, d22
	VADD.F32 d3, d3, d23
	VADD.F32 d4, d4, d24
	VADD.F32 d5, d5, d25
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128], r3

	# Accumulate c[1][0:16] += acc[1][0:16]
	VLD1.32 {d0-d3}, [r2:128]!
	VLD1.32 {d4-d5}, [r2:128], ip
	VADD.F32 d0, d0, d26
	VADD.F32 d1, d1, d27
	VADD.F32 d2, d2, d28
	VADD.F32 d3, d3, d29
	VADD.F32 d4, d4, d30
	VADD.F32 d5, d5, d31
	VST1.32 {d0-d3}, [r2:128]!
	VST1.32 {d4-d5}, [r2:128]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_s4gemm_only_3x3__aarch32_meerkat
