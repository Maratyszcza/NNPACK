#include <stddef.h>
#include <stdint.h>

#include <nnpack/arm_neon.h>
#include <nnpack/macros.h>


void nnp_h4gemm_only_3x3__neonhp(
	size_t k, size_t update,
	const void *restrict a_ptr,
	const void *restrict b_ptr,
	void *restrict c_ptr,
	size_t row_stride_c)
{
	const uint16_t *restrict a = a_ptr;
	const uint16_t *restrict b = b_ptr;
	uint16_t *restrict c = c_ptr;

	float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f);
	float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f);
	float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f);
	do {
		const float32x4_t a0 = vld1q_f32_f16(a + 0);
		const float32x4_t a1 = vld1q_f32_f16(a + 4);
		const float32x4_t a2 = vld1q_f32_f16(a + 8);

		const float32x4_t b0 = vld1q_f32_f16(b +  0);
		acc00 = vmuladdq_f32(acc00, a0, b0);
		acc10 = vmuladdq_f32(acc10, a1, b0);
		acc20 = vmuladdq_f32(acc20, a2, b0);
		const float32x4_t b1 = vld1q_f32_f16(b +  4);
		acc01 = vmuladdq_f32(acc01, a0, b1);
		acc11 = vmuladdq_f32(acc11, a1, b1);
		acc21 = vmuladdq_f32(acc21, a2, b1);
		const float32x4_t b2 = vld1q_f32_f16(b +  8);
		acc02 = vmuladdq_f32(acc02, a0, b2);
		acc12 = vmuladdq_f32(acc12, a1, b2);
		acc22 = vmuladdq_f32(acc22, a2, b2);

		a += 12;
		b += 12;
	} while (--k);

	if (update != 0) {
		vst1q_f16_f32(c +  0, vaddq_f32(vld1q_f32_f16(c +  0), acc00));
		vst1q_f16_f32(c +  4, vaddq_f32(vld1q_f32_f16(c +  4), acc01));
		vst1q_f16_f32(c +  8, vaddq_f32(vld1q_f32_f16(c +  8), acc02));
		c += row_stride_c;
		vst1q_f16_f32(c +  0, vaddq_f32(vld1q_f32_f16(c +  0), acc10));
		vst1q_f16_f32(c +  4, vaddq_f32(vld1q_f32_f16(c +  4), acc11));
		vst1q_f16_f32(c +  8, vaddq_f32(vld1q_f32_f16(c +  8), acc12));
		c += row_stride_c;
		vst1q_f16_f32(c +  0, vaddq_f32(vld1q_f32_f16(c +  0), acc20));
		vst1q_f16_f32(c +  4, vaddq_f32(vld1q_f32_f16(c +  4), acc21));
		vst1q_f16_f32(c +  8, vaddq_f32(vld1q_f32_f16(c +  8), acc22));
	} else {
		vst1q_f16_f32(c +  0, acc00);
		vst1q_f16_f32(c +  4, acc01);
		vst1q_f16_f32(c +  8, acc02);
		c += row_stride_c;
		vst1q_f16_f32(c +  0, acc10);
		vst1q_f16_f32(c +  4, acc11);
		vst1q_f16_f32(c +  8, acc12);
		c += row_stride_c;
		vst1q_f16_f32(c +  0, acc20);
		vst1q_f16_f32(c +  4, acc21);
		vst1q_f16_f32(c +  8, acc22);
	}
}

void nnp_h4gemm_upto_3x3__neonhp(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const void *restrict a_ptr,
	const void *restrict b_ptr,
	void *restrict c_ptr,
	size_t row_stride_c)
{
	const uint16_t *restrict a = a_ptr;

	float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f);
	const uint16_t* b0_ptr = b_ptr;
	const uint16_t* b1_ptr = b0_ptr + 4;
	if (nr < 2) {
		b1_ptr = b0_ptr;
	}
	const uint16_t* b2_ptr = b1_ptr + 4;
	if (nr <= 2) {
		b2_ptr = b1_ptr;
	}
	const size_t b_increment = nr * 4;
	if (mr < 2) {
		/* mr == 1 */
		do {
			const float32x4_t a0 = vld1q_f32_f16(a); a += 4;

			const float32x4_t b0 = vld1q_f32_f16(b0_ptr); b0_ptr += b_increment;
			acc00 = vmuladdq_f32(acc00, a0, b0);

			const float32x4_t b1 = vld1q_f32_f16(b1_ptr); b1_ptr += b_increment;
			acc01 = vmuladdq_f32(acc01, a0, b1);

			const float32x4_t b2 = vld1q_f32_f16(b2_ptr); b2_ptr += b_increment;
			acc02 = vmuladdq_f32(acc02, a0, b2);
		} while (--k);

		uint16_t* restrict c = c_ptr;
		if (update != 0) {
			vst1q_f16_f32(c, vaddq_f32(vld1q_f32_f16(c), acc00));
			if (nr > 1) {
				vst1q_f16_f32(c + 4, vaddq_f32(vld1q_f32_f16(c + 4), acc01));
				if (nr > 2) {
					vst1q_f16_f32(c + 8, vaddq_f32(vld1q_f32_f16(c + 8), acc02));
				}
			}
		} else {
			vst1q_f16_f32(c, acc00);
			if (nr > 1) {
				vst1q_f16_f32(c + 4, acc01);
				if (nr > 2) {
					vst1q_f16_f32(c + 8, acc02);
				}
			}
		}
	} else {
		float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f);
		if (mr <= 2) {
			/* mr == 2 */
			do {
				const float32x4_t a0 = vld1q_f32_f16(a); a += 4;
				const float32x4_t a1 = vld1q_f32_f16(a); a += 4;

				const float32x4_t b0 = vld1q_f32_f16(b0_ptr); b0_ptr += b_increment;
				acc00 = vmuladdq_f32(acc00, a0, b0);
				acc10 = vmuladdq_f32(acc10, a1, b0);

				const float32x4_t b1 = vld1q_f32_f16(b1_ptr); b1_ptr += b_increment;
				acc01 = vmuladdq_f32(acc01, a0, b1);
				acc11 = vmuladdq_f32(acc11, a1, b1);

				const float32x4_t b2 = vld1q_f32_f16(b2_ptr); b2_ptr += b_increment;
				acc02 = vmuladdq_f32(acc02, a0, b2);
				acc12 = vmuladdq_f32(acc12, a1, b2);
			} while (--k);

			uint16_t* restrict crow0 = c_ptr;
			uint16_t* restrict crow1 = crow0 + row_stride_c;
			if (update != 0) {
				vst1q_f16_f32(crow0, vaddq_f32(vld1q_f32_f16(crow0), acc00)); crow0 += 4;
				vst1q_f16_f32(crow1, vaddq_f32(vld1q_f32_f16(crow1), acc10)); crow1 += 4;
				if (nr > 1) {
					vst1q_f16_f32(crow0, vaddq_f32(vld1q_f32_f16(crow0), acc01)); crow0 += 4;
					vst1q_f16_f32(crow1, vaddq_f32(vld1q_f32_f16(crow1), acc11)); crow1 += 4;
					if (nr > 2) {
						vst1q_f16_f32(crow0, vaddq_f32(vld1q_f32_f16(crow0), acc02));
						vst1q_f16_f32(crow1, vaddq_f32(vld1q_f32_f16(crow1), acc12));
					}
				}
			} else {
				vst1q_f16_f32(crow0, acc00); crow0 += 4;
				vst1q_f16_f32(crow1, acc10); crow1 += 4;
				if (nr > 1) {
					vst1q_f16_f32(crow0, acc01); crow0 += 4;
					vst1q_f16_f32(crow1, acc11); crow1 += 4;
					if (nr > 2) {
						vst1q_f16_f32(crow0, acc02);
						vst1q_f16_f32(crow1, acc12);
					}
				}
			}
		} else {
			/* mr == 3 */
			float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f);
			do {
				const float32x4_t a0 = vld1q_f32_f16(a); a += 4;
				const float32x4_t a1 = vld1q_f32_f16(a); a += 4;
				const float32x4_t a2 = vld1q_f32_f16(a); a += 4;

				const float32x4_t b0 = vld1q_f32_f16(b0_ptr); b0_ptr += b_increment;
				acc00 = vmuladdq_f32(acc00, a0, b0);
				acc10 = vmuladdq_f32(acc10, a1, b0);
				acc20 = vmuladdq_f32(acc20, a2, b0);

				const float32x4_t b1 = vld1q_f32_f16(b1_ptr); b1_ptr += b_increment;
				acc01 = vmuladdq_f32(acc01, a0, b1);
				acc11 = vmuladdq_f32(acc11, a1, b1);
				acc21 = vmuladdq_f32(acc21, a2, b1);

				const float32x4_t b2 = vld1q_f32_f16(b2_ptr); b2_ptr += b_increment;
				acc02 = vmuladdq_f32(acc02, a0, b2);
				acc12 = vmuladdq_f32(acc12, a1, b2);
				acc22 = vmuladdq_f32(acc22, a2, b2);
			} while (--k);

			uint16_t* restrict crow0 = c_ptr;
			uint16_t* restrict crow1 = crow0 + row_stride_c;
			uint16_t* restrict crow2 = crow1 + row_stride_c;
			if (update != 0) {
				vst1q_f16_f32(crow0, vaddq_f32(vld1q_f32_f16(crow0), acc00)); crow0 += 4;
				vst1q_f16_f32(crow1, vaddq_f32(vld1q_f32_f16(crow1), acc10)); crow1 += 4;
				vst1q_f16_f32(crow2, vaddq_f32(vld1q_f32_f16(crow2), acc20)); crow2 += 4;
				if (nr > 1) {
					vst1q_f16_f32(crow0, vaddq_f32(vld1q_f32_f16(crow0), acc01)); crow0 += 4;
					vst1q_f16_f32(crow1, vaddq_f32(vld1q_f32_f16(crow1), acc11)); crow1 += 4;
					vst1q_f16_f32(crow2, vaddq_f32(vld1q_f32_f16(crow2), acc21)); crow2 += 4;
					if (nr > 2) {
						vst1q_f16_f32(crow0, vaddq_f32(vld1q_f32_f16(crow0), acc02));
						vst1q_f16_f32(crow1, vaddq_f32(vld1q_f32_f16(crow1), acc12));
						vst1q_f16_f32(crow2, vaddq_f32(vld1q_f32_f16(crow2), acc22));
					}
				}
			} else {
				vst1q_f16_f32(crow0, acc00); crow0 += 4;
				vst1q_f16_f32(crow1, acc10); crow1 += 4;
				vst1q_f16_f32(crow2, acc20); crow2 += 4;
				if (nr > 1) {
					vst1q_f16_f32(crow0, acc01); crow0 += 4;
					vst1q_f16_f32(crow1, acc11); crow1 += 4;
					vst1q_f16_f32(crow2, acc21); crow2 += 4;
					if (nr > 2) {
						vst1q_f16_f32(crow0, acc02);
						vst1q_f16_f32(crow1, acc12);
						vst1q_f16_f32(crow2, acc22);
					}
				}
			}
		}
	}
}
