#include <stddef.h>
#include <stdint.h>

#include <nnpack/arm_neon.h>


void nnp_h4gemm_only_3x4__neonhp(
	size_t k, size_t update,
	const void *restrict a_ptr,
	const void *restrict b_ptr,
	void *restrict c_ptr,
	size_t row_stride, size_t column_stride)
{
	const uint16_t *restrict a = a_ptr;
	const uint16_t *restrict b = b_ptr;
	uint16_t *restrict c = c_ptr;

	float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f);
	float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);
	float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f), acc23 = vdupq_n_f32(0.0f);
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
		const float32x4_t b3 = vld1q_f32_f16(b + 12);
		acc03 = vmuladdq_f32(acc03, a0, b3);
		acc13 = vmuladdq_f32(acc13, a1, b3);
		acc23 = vmuladdq_f32(acc23, a2, b3);

		a += 12;
		b += 16;
	} while (--k);

	if (update != 0) {
		vst1q_f16_f32(c +  0, vaddq_f32(vld1q_f32_f16(c +  0), acc00));
		vst1q_f16_f32(c +  4, vaddq_f32(vld1q_f32_f16(c +  4), acc01));
		vst1q_f16_f32(c +  8, vaddq_f32(vld1q_f32_f16(c +  8), acc02));
		vst1q_f16_f32(c + 12, vaddq_f32(vld1q_f32_f16(c + 12), acc03));
		c += row_stride;
		vst1q_f16_f32(c +  0, vaddq_f32(vld1q_f32_f16(c +  0), acc10));
		vst1q_f16_f32(c +  4, vaddq_f32(vld1q_f32_f16(c +  4), acc11));
		vst1q_f16_f32(c +  8, vaddq_f32(vld1q_f32_f16(c +  8), acc12));
		vst1q_f16_f32(c + 12, vaddq_f32(vld1q_f32_f16(c + 12), acc13));
		c += row_stride;
		vst1q_f16_f32(c +  0, vaddq_f32(vld1q_f32_f16(c +  0), acc20));
		vst1q_f16_f32(c +  4, vaddq_f32(vld1q_f32_f16(c +  4), acc21));
		vst1q_f16_f32(c +  8, vaddq_f32(vld1q_f32_f16(c +  8), acc22));
		vst1q_f16_f32(c + 12, vaddq_f32(vld1q_f32_f16(c + 12), acc23));
	} else {
		vst1q_f16_f32(c +  0, acc00);
		vst1q_f16_f32(c +  4, acc01);
		vst1q_f16_f32(c +  8, acc02);
		vst1q_f16_f32(c + 12, acc03);
		c += row_stride;
		vst1q_f16_f32(c +  0, acc10);
		vst1q_f16_f32(c +  4, acc11);
		vst1q_f16_f32(c +  8, acc12);
		vst1q_f16_f32(c + 12, acc13);
		c += row_stride;
		vst1q_f16_f32(c +  0, acc20);
		vst1q_f16_f32(c +  4, acc21);
		vst1q_f16_f32(c +  8, acc22);
		vst1q_f16_f32(c + 12, acc23);
	}
}

void nnp_h4gemm_upto_3x4__neonhp(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const void *restrict a_ptr,
	const void *restrict b_ptr,
	void *restrict c_ptr,
	size_t row_stride, size_t column_stride)
{
	const uint16_t *restrict a = a_ptr;
	const uint16_t *restrict b = b_ptr;
	uint16_t *restrict c = c_ptr;

	float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f);
	float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);
	float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f), acc23 = vdupq_n_f32(0.0f);
	do {
		float32x4_t a0, a1, a2;

		a0 = vld1q_f32_f16(a);
		a += 4;
		if (mr > 1) {
			a1 = vld1q_f32_f16(a);
			a += 4;
			if (mr > 2) {
				a2 = vld1q_f32_f16(a);
				a += 4;
			}
		}

		const float32x4_t b0 = vld1q_f32_f16(b);
		b += 4;
		acc00 = vmuladdq_f32(acc00, a0, b0);
		acc10 = vmuladdq_f32(acc10, a1, b0);
		acc20 = vmuladdq_f32(acc20, a2, b0);
		if (nr > 1) {
			const float32x4_t b1 = vld1q_f32_f16(b);
			b += 4;
			acc01 = vmuladdq_f32(acc01, a0, b1);
			acc11 = vmuladdq_f32(acc11, a1, b1);
			acc21 = vmuladdq_f32(acc21, a2, b1);
			if (nr > 2) {
				const float32x4_t b2 = vld1q_f32_f16(b);
				b += 4;
				acc02 = vmuladdq_f32(acc02, a0, b2);
				acc12 = vmuladdq_f32(acc12, a1, b2);
				acc22 = vmuladdq_f32(acc22, a2, b2);
				if (nr > 3) {
					const float32x4_t b3 = vld1q_f32_f16(b);
					b += 4;
					acc03 = vmuladdq_f32(acc03, a0, b3);
					acc13 = vmuladdq_f32(acc13, a1, b3);
					acc23 = vmuladdq_f32(acc23, a2, b3);
				}
			}
		}
	} while (--k);

	if (update != 0) {
		vst1q_f16_f32(c, vaddq_f32(vld1q_f32_f16(c), acc00));
		if (nr > 1) {
			vst1q_f16_f32(c + 4, vaddq_f32(vld1q_f32_f16(c + 4), acc01));
			if (nr > 2) {
				vst1q_f16_f32(c + 8, vaddq_f32(vld1q_f32_f16(c + 8), acc02));
				if (nr > 3) {
					vst1q_f16_f32(c + 12, vaddq_f32(vld1q_f32_f16(c + 12), acc03));
				}
			}
		}
		if (mr > 1) {
			c += row_stride;
			vst1q_f16_f32(c, vaddq_f32(vld1q_f32_f16(c), acc10));
			if (nr > 1) {
				vst1q_f16_f32(c + 4, vaddq_f32(vld1q_f32_f16(c + 4), acc11));
				if (nr > 2) {
					vst1q_f16_f32(c + 8, vaddq_f32(vld1q_f32_f16(c + 8), acc12));
					if (nr > 3) {
						vst1q_f16_f32(c + 12, vaddq_f32(vld1q_f32_f16(c + 12), acc13));
					}
				}
			}
			if (mr > 2) {
				c += row_stride;
				vst1q_f16_f32(c, vaddq_f32(vld1q_f32_f16(c), acc20));
				if (nr > 1) {
					vst1q_f16_f32(c + 4, vaddq_f32(vld1q_f32_f16(c + 4), acc21));
					if (nr > 2) {
						vst1q_f16_f32(c + 8, vaddq_f32(vld1q_f32_f16(c + 8), acc22));
						if (nr > 3) {
							vst1q_f16_f32(c + 12, vaddq_f32(vld1q_f32_f16(c + 12), acc23));
						}
					}
				}
			}
		}
	} else {
		vst1q_f16_f32(c, acc00);
		if (nr > 1) {
			vst1q_f16_f32(c + 4, acc01);
			if (nr > 2) {
				vst1q_f16_f32(c + 8, acc02);
				if (nr > 3) {
					vst1q_f16_f32(c + 12, acc03);
				}
			}
		}
		if (mr > 1) {
			c += row_stride;
			vst1q_f16_f32(c, acc10);
			if (nr > 1) {
				vst1q_f16_f32(c + 4, acc11);
				if (nr > 2) {
					vst1q_f16_f32(c + 8, acc12);
					if (nr > 3) {
						vst1q_f16_f32(c + 12, acc13);
					}
				}
			}
			if (mr > 2) {
				c += row_stride;
				vst1q_f16_f32(c, acc20);
				if (nr > 1) {
					vst1q_f16_f32(c + 4, acc21);
					if (nr > 2) {
						vst1q_f16_f32(c + 8, acc22);
						if (nr > 3) {
							vst1q_f16_f32(c + 12, acc23);
						}
					}
				}
			}
		}
	}
}
