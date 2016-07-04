#include <stddef.h>
#include <stdint.h>

#include <nnpack/simd.h>


void nnp_c4gemm_only_2x2__psimd(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	v4f acc00r = v4f_zero(), acc00i = v4f_zero();
	v4f acc01r = v4f_zero(), acc01i = v4f_zero();
	v4f acc10r = v4f_zero(), acc10i = v4f_zero();
	v4f acc11r = v4f_zero(), acc11i = v4f_zero();
	do {
		const v4f a0r = v4f_ld(a +  0);
		const v4f a0i = v4f_ld(a +  4);
		const v4f a1r = v4f_ld(a +  8);
		const v4f a1i = v4f_ld(a + 12);

		const v4f b0r = v4f_ld(b + 0);
		const v4f b1r = v4f_ld(b + 8);
		acc00r += a0r * b0r;
		acc00i += a0i * b0r;
		acc01r += a0r * b1r;
		acc01i += a0i * b1r;
		acc10r += a1r * b0r;
		acc10i += a1i * b0r;
		acc11r += a1r * b1r;
		acc11i += a1i * b1r;
		const v4f b0i = v4f_ld(b +  4);
		const v4f b1i = v4f_ld(b + 12);
		acc00r -= a0i * b0i;
		acc00i += a0r * b0i;
		acc01r -= a0i * b1i;
		acc01i += a0r * b1i;
		acc10r -= a1i * b0i;
		acc10i += a1r * b0i;
		acc11r -= a1i * b1i;
		acc11i += a1r * b1i;

		a += 16;
		b += 16;
	} while (--k);

	if (update != 0) {
		v4f_st(c +  0, v4f_ld(c +  0) + acc00r);
		v4f_st(c +  4, v4f_ld(c +  4) + acc00i);
		v4f_st(c +  8, v4f_ld(c +  8) + acc01r);
		v4f_st(c + 12, v4f_ld(c + 12) + acc01i);
		c += row_stride_c;
		v4f_st(c +  0, v4f_ld(c +  0) + acc10r);
		v4f_st(c +  4, v4f_ld(c +  4) + acc10i);
		v4f_st(c +  8, v4f_ld(c +  8) + acc11r);
		v4f_st(c + 12, v4f_ld(c + 12) + acc11i);
	} else {
		v4f_st(c +  0, acc00r);
		v4f_st(c +  4, acc00i);
		v4f_st(c +  8, acc01r);
		v4f_st(c + 12, acc01i);
		c += row_stride_c;
		v4f_st(c +  0, acc10r);
		v4f_st(c +  4, acc10i);
		v4f_st(c +  8, acc11r);
		v4f_st(c + 12, acc11i);
	}
}

void nnp_c4gemm_upto_2x2__psimd(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	v4f acc00r = v4f_zero(), acc00i = v4f_zero();
	v4f acc01r = v4f_zero(), acc01i = v4f_zero();
	v4f acc10r = v4f_zero(), acc10i = v4f_zero();
	v4f acc11r = v4f_zero(), acc11i = v4f_zero();
	do {
		v4f a0r, a0i, a1r, a1i;

		a0r = v4f_ld(a + 0);
		a0i = v4f_ld(a + 4);
		a += 8;
		if (mr > 1) {
			a1r = v4f_ld(a + 0);
			a1i = v4f_ld(a + 4);
			a += 8;
		}

		const v4f b0r = v4f_ld(b + 0);
		const v4f b0i = v4f_ld(b + 4);
		b += 8;

		acc00r += a0r * b0r;
		acc00i += a0i * b0r;
		acc10r += a1r * b0r;
		acc10i += a1i * b0r;

		acc00r -= a0i * b0i;
		acc00i += a0r * b0i;
		acc10r -= a1i * b0i;
		acc10i += a1r * b0i;

		if (nr > 1) {
			const v4f b1r = v4f_ld(b + 0);
			const v4f b1i = v4f_ld(b + 4);
			b += 8;

			acc01r += a0r * b1r;
			acc01i += a0i * b1r;
			acc11r += a1r * b1r;
			acc11i += a1i * b1r;

			acc01r -= a0i * b1i;
			acc01i += a0r * b1i;
			acc11r -= a1i * b1i;
			acc11i += a1r * b1i;
		}
	} while (--k);

	if (update != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00r);
		v4f_st(c + 4, v4f_ld(c + 4) + acc00i);
		if (nr > 1) {
			v4f_st(c +  8, v4f_ld(c +  8) + acc01r);
			v4f_st(c + 12, v4f_ld(c + 12) + acc01i);
		}
		if (mr > 1) {
			c += row_stride_c;
			v4f_st(c +  0, v4f_ld(c +  0) + acc10r);
			v4f_st(c +  4, v4f_ld(c +  4) + acc10i);
			if (nr > 1) {
				v4f_st(c +  8, v4f_ld(c +  8) + acc11r);
				v4f_st(c + 12, v4f_ld(c + 12) + acc11i);
			}
		}
	} else {
		v4f_st(c + 0, acc00r);
		v4f_st(c + 4, acc00i);
		if (nr > 1) {
			v4f_st(c +  8, acc01r);
			v4f_st(c + 12, acc01i);
		}
		if (mr > 1) {
			c += row_stride_c;
			v4f_st(c + 0, acc10r);
			v4f_st(c + 4, acc10i);
			if (nr > 1) {
				v4f_st(c +  8, acc11r);
				v4f_st(c + 12, acc11i);
			}
		}
	}
}
