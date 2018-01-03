#include <nnpack/assembly.h>

# void nnp_sgemm_only_6x8__neon(
#        size_t k,
#        size_t update,
#        const float* a,
#        const float* b,
#        float* c,
#        size_t row_stride_c)
BEGIN_FUNCTION nnp_sgemm_only_6x8__aarch32_neon
	.arm
#ifndef __APPLE__
	.arch armv7-a
	.fpu neon
#endif

	VPUSH {d8-d15}

	# q4 := acc[0][0:4]
	VMOV.I32  q4, #0
	# q5 := acc[0][4:8]
	VMOV.I32  q5, #0
	# q6 := acc[1][0:4]
	VMOV.I32  q6, #0
	# q7 := acc[1][4:8]
	VMOV.I32  q7, #0
	# q8 := acc[2][0:4]
	VMOV.I32  q8, #0
	# q9 := acc[2][4:8]
	VMOV.I32  q9, #0
	# q10 := acc[3][0:4]
	VMOV.I32 q10, #0
	# q11 := acc[3][4:8]
	VMOV.I32 q11, #0
	# q12 := acc[4][0:4]
	VMOV.I32 q12, #0
	# q13 := acc[4][4:8]
	VMOV.I32 q13, #0
	# q14 := acc[5][0:4]
	VMOV.I32 q14, #0
	# q15 := acc[5][4:8]
	VMOV.I32 q15, #0

0:
	# Load b[0:8]
	# - d4 = b[0:2]
	# - d5 = b[2:4]
	# - d6 = b[4:6]
	# - d7 = b[6:8]
	VLD1.32 {d4-d7}, [r3:128]!

	# Load a[0:6]
	# - d0 = a[0:2]
	# - d1 = a[2:4]
	# - d2 = a[4:6]
	VLD1.32 {d0-d2}, [r2:64]!

	# Update acc[0][0:4] += a[0] * b[0:4]
	VMLA.F32 q4, q2, d0[0]
	# Update acc[0][4:8] += a[0] * b[4:8]
	VMLA.F32 q5, q3, d0[0]
	# Update acc[1][0:4] += a[1] * b[0:4]
	VMLA.F32 q6, q2, d0[1]
	# Update acc[1][4:8] += a[1] * b[4:8]
	VMLA.F32 q7, q3, d0[1]

	# Update acc[2][0:4] += a[2] * b[0:4]
	VMLA.F32  q8, q2, d1[0]
	# Update acc[2][4:8] += a[2] * b[4:8]
	VMLA.F32  q9, q3, d1[0]
	# Update acc[3][0:4] += a[3] * b[0:4]
	VMLA.F32 q10, q2, d1[1]
	# Update acc[3][4:8] += a[3] * b[4:8]
	VMLA.F32 q11, q3, d1[1]

	# Update acc[4][0:4] += a[4] * b[0:4]
	VMLA.F32 q12, q2, d2[0]
	# Update acc[4][4:8] += a[4] * b[4:8]
	VMLA.F32 q13, q3, d2[0]
	# Update acc[5][0:4] += a[5] * b[0:4]
	VMLA.F32 q14, q2, d2[1]
	# Update acc[5][4:8] += a[5] * b[4:8]
	VMLA.F32 q15, q3, d2[1]

	SUBS r0, r0, #1
	BNE 0b

	# Load arguments:
	# - r2 = c
	# - r3 = row_stride_c
	LDRD r2, r3, [sp, #64]
	# Check if c is updated (r1 != 0) or overwritten (r1 == 0)
	CMP r1, #0
	# Convert row_stride_c (stride in elements) to stride in bytes
	LSL r3, r3, #2
	# Skip to label 1 to overwrite c
	BEQ 1f

	##### Accumulate c matrix with results in acc[0:6][0:8]

	# Accumulate c[0][0:8] += acc[0][0:8]
	VLD1.32 {d0-d3}, [r2]
	VADD.F32 q0, q0, q4
	VADD.F32 q1, q1, q5
	VST1.32 {d0-d3}, [r2], r3

	# Accumulate c[1][0:8] += acc[1][0:8]
	VLD1.32 {d4-d7}, [r2]
	VADD.F32 q2, q2, q6
	VADD.F32 q3, q3, q7
	VST1.32 {d4-d7}, [r2], r3

	# Accumulate c[2][0:8] += acc[2][0:8]
	VLD1.32 {d0-d3}, [r2]
	VADD.F32 q0, q0, q8
	VADD.F32 q1, q1, q9
	VST1.32 {d0-d3}, [r2], r3

	# Accumulate c[3][0:8] += acc[3][0:8]
	VLD1.32 {d4-d7}, [r2]
	VADD.F32 q2, q2, q10
	VADD.F32 q3, q3, q11
	VST1.32 {d4-d7}, [r2], r3

	# Accumulate c[4][0:8] += acc[4][0:8]
	VLD1.32 {d0-d3}, [r2]
	VADD.F32 q0, q0, q12
	VADD.F32 q1, q1, q13
	VST1.32 {d0-d3}, [r2], r3

	# Accumulate c[5][0:8] += acc[5][0:8]
	VLD1.32 {d4-d7}, [r2]
	VADD.F32 q2, q2, q14
	VADD.F32 q3, q3, q15
	VST1.32 {d4-d7}, [r2]

	VPOP {d8-d15}
	BX lr

1:
	##### Overwrite c matrix with results in acc[0:6][0:8]

	# Overwrite c[0][0:8] += acc[0][0:8]
	VST1.32 {d8-d11}, [r2], r3

	# Overwrite c[1][0:8] += acc[1][0:8]
	VST1.32 {d12-d15}, [r2], r3

	# Overwrite c[2][0:8] += acc[2][0:8]
	VST1.32 {d16-d19}, [r2], r3

	# Overwrite c[3][0:8] += acc[3][0:8]
	VST1.32 {d20-d23}, [r2], r3

	# Overwrite c[4][0:8] += acc[4][0:8]
	VST1.32 {d24-d27}, [r2], r3

	# Overwrite c[5][0:8] += acc[5][0:8]
	VST1.32 {d28-d31}, [r2]

	VPOP {d8-d15}
	BX lr
END_FUNCTION nnp_sgemm_only_6x8__aarch32_neon
