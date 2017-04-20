#include <stddef.h>
#include <stdint.h>

#include <nnpack/arm_neon.h>


void nnp_s4c2gemm_conjb_transc_only_2x2__neon(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float32x4_t acc00r = vdupq_n_f32(0.0f), acc00i = vdupq_n_f32(0.0f);
	float32x4_t acc01r = vdupq_n_f32(0.0f), acc01i = vdupq_n_f32(0.0f);
	float32x4_t acc10r = vdupq_n_f32(0.0f), acc10i = vdupq_n_f32(0.0f);
	float32x4_t acc11r = vdupq_n_f32(0.0f), acc11i = vdupq_n_f32(0.0f);
	do {
		const float32x4_t a0r = vld1q_f32_aligned(a +  0);
		const float32x4_t a0i = vld1q_f32_aligned(a +  4);
		const float32x4_t a1r = vld1q_f32_aligned(a +  8);
		const float32x4_t a1i = vld1q_f32_aligned(a + 12);

		float32x4_t b0r = vld1q_f32_aligned(b +  0);
		float32x4_t b0i = vld1q_f32_aligned(b +  4);
		float32x4_t b1r = vld1q_f32_aligned(b +  8);
		float32x4_t b1i = vld1q_f32_aligned(b + 12);
		acc00r = vmuladdq_f32(acc00r, a0r, b0r);
		acc10r = vmuladdq_f32(acc10r, a1r, b0r);
		acc01r = vmuladdq_f32(acc01r, a0r, b1r);
		acc11r = vmuladdq_f32(acc11r, a1r, b1r);

		b0r = vcombine_f32(vget_low_f32(b0i), vget_high_f32(b0r));
		b1r = vcombine_f32(vget_low_f32(b1i), vget_high_f32(b1r));
		acc00i = vmuladdq_f32(acc00i, a0i, b0r);
		acc10i = vmuladdq_f32(acc10i, a1i, b0r);
		acc01i = vmuladdq_f32(acc01i, a0i, b1r);
		acc11i = vmuladdq_f32(acc11i, a1i, b1r);

		acc00r = vcombine_f32(vget_low_f32(acc00r), vmuladd_f32(vget_high_f32(acc00r), vget_high_f32(a0i), vget_high_f32(b0i)));
		acc00i = vcombine_f32(vget_low_f32(acc00i), vmulsub_f32(vget_high_f32(acc00i), vget_high_f32(a0r), vget_high_f32(b0i)));
		acc10r = vcombine_f32(vget_low_f32(acc10r), vmuladd_f32(vget_high_f32(acc10r), vget_high_f32(a1i), vget_high_f32(b0i)));
		acc10i = vcombine_f32(vget_low_f32(acc10i), vmulsub_f32(vget_high_f32(acc10i), vget_high_f32(a1r), vget_high_f32(b0i)));

		acc01r = vcombine_f32(vget_low_f32(acc01r), vmuladd_f32(vget_high_f32(acc01r), vget_high_f32(a0i), vget_high_f32(b1i)));
		acc01i = vcombine_f32(vget_low_f32(acc01i), vmulsub_f32(vget_high_f32(acc01i), vget_high_f32(a0r), vget_high_f32(b1i)));
		acc11r = vcombine_f32(vget_low_f32(acc11r), vmuladd_f32(vget_high_f32(acc11r), vget_high_f32(a1i), vget_high_f32(b1i)));
		acc11i = vcombine_f32(vget_low_f32(acc11i), vmulsub_f32(vget_high_f32(acc11i), vget_high_f32(a1r), vget_high_f32(b1i)));

		a += 16;
		b += 16;
	} while (--k);

	if (update != 0) {
		vst1q_f32_aligned(c +  0, vaddq_f32(vld1q_f32_aligned(c +  0), acc00r));
		vst1q_f32_aligned(c +  4, vaddq_f32(vld1q_f32_aligned(c +  4), acc00i));
		vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc10r));
		vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc10i));
		c += row_stride_c;
		vst1q_f32_aligned(c +  0, vaddq_f32(vld1q_f32_aligned(c +  0), acc01r));
		vst1q_f32_aligned(c +  4, vaddq_f32(vld1q_f32_aligned(c +  4), acc01i));
		vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc11r));
		vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc11i));
	} else {
		vst1q_f32_aligned(c +  0, acc00r);
		vst1q_f32_aligned(c +  4, acc00i);
		vst1q_f32_aligned(c +  8, acc10r);
		vst1q_f32_aligned(c + 12, acc10i);
		c += row_stride_c;
		vst1q_f32_aligned(c +  0, acc01r);
		vst1q_f32_aligned(c +  4, acc01i);
		vst1q_f32_aligned(c +  8, acc11r);
		vst1q_f32_aligned(c + 12, acc11i);
	}
}

void nnp_s4c2gemm_conjb_transc_upto_2x2__neon(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float32x4_t acc00r = vdupq_n_f32(0.0f), acc00i = vdupq_n_f32(0.0f);
	float32x4_t acc01r = vdupq_n_f32(0.0f), acc01i = vdupq_n_f32(0.0f);
	float32x4_t acc10r = vdupq_n_f32(0.0f), acc10i = vdupq_n_f32(0.0f);
	float32x4_t acc11r = vdupq_n_f32(0.0f), acc11i = vdupq_n_f32(0.0f);
	do {
		float32x4_t a0r, a0i, a1r, a1i;
		a0r = vld1q_f32_aligned(a + 0);
		a0i = vld1q_f32_aligned(a + 4);
		a += 8;
		if (mr > 1) {
			a1r = vld1q_f32_aligned(a + 0);
			a1i = vld1q_f32_aligned(a + 4);
			a += 8;
		}

		float32x4_t b0r = vld1q_f32_aligned(b + 0);
		float32x4_t b0i = vld1q_f32_aligned(b + 4);
		b += 8;

		acc00r = vmuladdq_f32(acc00r, a0r, b0r);
		acc10r = vmuladdq_f32(acc10r, a1r, b0r);
		b0r = vcombine_f32(vget_low_f32(b0i), vget_high_f32(b0r));
		acc00i = vmuladdq_f32(acc00i, a0i, b0r);
		acc10i = vmuladdq_f32(acc10i, a1i, b0r);

		acc00r = vcombine_f32(vget_low_f32(acc00r), vmuladd_f32(vget_high_f32(acc00r), vget_high_f32(a0i), vget_high_f32(b0i)));
		acc00i = vcombine_f32(vget_low_f32(acc00i), vmulsub_f32(vget_high_f32(acc00i), vget_high_f32(a0r), vget_high_f32(b0i)));
		acc10r = vcombine_f32(vget_low_f32(acc10r), vmuladd_f32(vget_high_f32(acc10r), vget_high_f32(a1i), vget_high_f32(b0i)));
		acc10i = vcombine_f32(vget_low_f32(acc10i), vmulsub_f32(vget_high_f32(acc10i), vget_high_f32(a1r), vget_high_f32(b0i)));

		if (nr > 1) {
			float32x4_t b1r = vld1q_f32_aligned(b + 0);
			float32x4_t b1i = vld1q_f32_aligned(b + 4);
			b += 8;

			acc01r = vmuladdq_f32(acc01r, a0r, b1r);
			acc11r = vmuladdq_f32(acc11r, a1r, b1r);
			b1r = vcombine_f32(vget_low_f32(b1i), vget_high_f32(b1r));
			acc01i = vmuladdq_f32(acc01i, a0i, b1r);
			acc11i = vmuladdq_f32(acc11i, a1i, b1r);

			acc01r = vcombine_f32(vget_low_f32(acc01r), vmuladd_f32(vget_high_f32(acc01r), vget_high_f32(a0i), vget_high_f32(b1i)));
			acc01i = vcombine_f32(vget_low_f32(acc01i), vmulsub_f32(vget_high_f32(acc01i), vget_high_f32(a0r), vget_high_f32(b1i)));
			acc11r = vcombine_f32(vget_low_f32(acc11r), vmuladd_f32(vget_high_f32(acc11r), vget_high_f32(a1i), vget_high_f32(b1i)));
			acc11i = vcombine_f32(vget_low_f32(acc11i), vmulsub_f32(vget_high_f32(acc11i), vget_high_f32(a1r), vget_high_f32(b1i)));
		}
	} while (--k);

	if (update != 0) {
		vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc00r));
		vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc00i));
		if (mr > 1) {
			vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc10r));
			vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc10i));
		}
		if (nr > 1) {
			c += row_stride_c;
			vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc01r));
			vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc01i));
			if (mr > 1) {
				vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc11r));
				vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc11i));
			}
		}
	} else {
		vst1q_f32_aligned(c + 0, acc00r);
		vst1q_f32_aligned(c + 4, acc00i);
		if (mr > 1) {
			vst1q_f32_aligned(c +  8, acc10r);
			vst1q_f32_aligned(c + 12, acc10i);
		}
		if (nr > 1) {
			c += row_stride_c;
			vst1q_f32_aligned(c + 0, acc01r);
			vst1q_f32_aligned(c + 4, acc01i);
			if (mr > 1) {
				vst1q_f32_aligned(c +  8, acc11r);
				vst1q_f32_aligned(c + 12, acc11i);
			}
		}
	}
}
