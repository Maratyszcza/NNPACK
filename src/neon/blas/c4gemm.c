#include <stddef.h>
#include <stdint.h>

#include <nnpack/arm_neon.h>


void nnp_c4gemm_only_2x2__neon(
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

		const float32x4_t b0r = vld1q_f32_aligned(b +  0);
		const float32x4_t b0i = vld1q_f32_aligned(b +  4);
		const float32x4_t b1r = vld1q_f32_aligned(b +  8);
		const float32x4_t b1i = vld1q_f32_aligned(b + 12);
		acc00r = vmuladdq_f32(acc00r, a0r, b0r);
		acc00i = vmuladdq_f32(acc00i, a0i, b0r);
		acc10r = vmuladdq_f32(acc10r, a1r, b0r);
		acc10i = vmuladdq_f32(acc10i, a1i, b0r);
		acc01r = vmuladdq_f32(acc01r, a0r, b1r);
		acc01i = vmuladdq_f32(acc01i, a0i, b1r);
		acc11r = vmuladdq_f32(acc11r, a1r, b1r);
		acc11i = vmuladdq_f32(acc11i, a1i, b1r);

		acc00r = vmulsubq_f32(acc00r, a0i, b0i);
		acc00i = vmuladdq_f32(acc00i, a0r, b0i);
		acc10r = vmulsubq_f32(acc10r, a1i, b0i);
		acc10i = vmuladdq_f32(acc10i, a1r, b0i);
		acc01r = vmulsubq_f32(acc01r, a0i, b1i);
		acc01i = vmuladdq_f32(acc01i, a0r, b1i);
		acc11r = vmulsubq_f32(acc11r, a1i, b1i);
		acc11i = vmuladdq_f32(acc11i, a1r, b1i);

		a += 16;
		b += 16;
	} while (--k);

	if (update != 0) {
		vst1q_f32_aligned(c +  0, vaddq_f32(vld1q_f32_aligned(c +  0), acc00r));
		vst1q_f32_aligned(c +  4, vaddq_f32(vld1q_f32_aligned(c +  4), acc00i));
		vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc01r));
		vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc01i));
		c += row_stride_c;
		vst1q_f32_aligned(c +  0, vaddq_f32(vld1q_f32_aligned(c +  0), acc10r));
		vst1q_f32_aligned(c +  4, vaddq_f32(vld1q_f32_aligned(c +  4), acc10i));
		vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc11r));
		vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc11i));
	} else {
		vst1q_f32_aligned(c +  0, acc00r);
		vst1q_f32_aligned(c +  4, acc00i);
		vst1q_f32_aligned(c +  8, acc01r);
		vst1q_f32_aligned(c + 12, acc01i);
		c += row_stride_c;
		vst1q_f32_aligned(c +  0, acc10r);
		vst1q_f32_aligned(c +  4, acc10i);
		vst1q_f32_aligned(c +  8, acc11r);
		vst1q_f32_aligned(c + 12, acc11i);
	}
}

void nnp_c4gemm_upto_2x2__neon(
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

		const float32x4_t b0r = vld1q_f32_aligned(b + 0);
		const float32x4_t b0i = vld1q_f32_aligned(b + 4);
		b += 8;

		acc00r = vmuladdq_f32(acc00r, a0r, b0r);
		acc00i = vmuladdq_f32(acc00i, a0i, b0r);
		acc10r = vmuladdq_f32(acc10r, a1r, b0r);
		acc10i = vmuladdq_f32(acc10i, a1i, b0r);

		acc00r = vmulsubq_f32(acc00r, a0i, b0i);
		acc00i = vmuladdq_f32(acc00i, a0r, b0i);
		acc10r = vmulsubq_f32(acc10r, a1i, b0i);
		acc10i = vmuladdq_f32(acc10i, a1r, b0i);

		if (nr > 1) {
			const float32x4_t b1r = vld1q_f32_aligned(b + 0);
			const float32x4_t b1i = vld1q_f32_aligned(b + 4);
			b += 8;

			acc01r = vmuladdq_f32(acc01r, a0r, b1r);
			acc01i = vmuladdq_f32(acc01i, a0i, b1r);
			acc11r = vmuladdq_f32(acc11r, a1r, b1r);
			acc11i = vmuladdq_f32(acc11i, a1i, b1r);

			acc01r = vmulsubq_f32(acc01r, a0i, b1i);
			acc01i = vmuladdq_f32(acc01i, a0r, b1i);
			acc11r = vmulsubq_f32(acc11r, a1i, b1i);
			acc11i = vmuladdq_f32(acc11i, a1r, b1i);
		}
	} while (--k);

	if (update != 0) {
		vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc00r));
		vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc00i));
		if (nr > 1) {
			vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc01r));
			vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc01i));
		}
		if (mr > 1) {
			c += row_stride_c;
			vst1q_f32_aligned(c +  0, vaddq_f32(vld1q_f32_aligned(c +  0), acc10r));
			vst1q_f32_aligned(c +  4, vaddq_f32(vld1q_f32_aligned(c +  4), acc10i));
			if (nr > 1) {
				vst1q_f32_aligned(c +  8, vaddq_f32(vld1q_f32_aligned(c +  8), acc11r));
				vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc11i));
			}
		}
	} else {
		vst1q_f32_aligned(c + 0, acc00r);
		vst1q_f32_aligned(c + 4, acc00i);
		if (nr > 1) {
			vst1q_f32_aligned(c +  8, acc01r);
			vst1q_f32_aligned(c + 12, acc01i);
		}
		if (mr > 1) {
			c += row_stride_c;
			vst1q_f32_aligned(c + 0, acc10r);
			vst1q_f32_aligned(c + 4, acc10i);
			if (nr > 1) {
				vst1q_f32_aligned(c +  8, acc11r);
				vst1q_f32_aligned(c + 12, acc11i);
			}
		}
	}
}
