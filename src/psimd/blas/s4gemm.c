#include <stddef.h>
#include <stdint.h>

#include <nnpack/simd.h>


void nnp_s4gemm_only_3x4__psimd(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero(), acc03 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero(), acc12 = v4f_zero(), acc13 = v4f_zero();
	v4f acc20 = v4f_zero(), acc21 = v4f_zero(), acc22 = v4f_zero(), acc23 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);
		const v4f a2 = v4f_ld(a + 8);

		const v4f b0 = v4f_ld(b +  0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;
		const v4f b1 = v4f_ld(b +  4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;
		acc21 += a2 * b1;
		const v4f b2 = v4f_ld(b +  8);
		acc02 += a0 * b2;
		acc12 += a1 * b2;
		acc22 += a2 * b2;
		const v4f b3 = v4f_ld(b + 12);
		acc03 += a0 * b3;
		acc13 += a1 * b3;
		acc23 += a2 * b3;

		a += 12;
		b += 16;
	} while (--k);

	if (update != 0) {
		v4f_st(c +  0, v4f_ld(c +  0) + acc00);
		v4f_st(c +  4, v4f_ld(c +  4) + acc01);
		v4f_st(c +  8, v4f_ld(c +  8) + acc02);
		v4f_st(c + 12, v4f_ld(c + 12) + acc03);
		c += row_stride;
		v4f_st(c +  0, v4f_ld(c +  0) + acc10);
		v4f_st(c +  4, v4f_ld(c +  4) + acc11);
		v4f_st(c +  8, v4f_ld(c +  8) + acc12);
		v4f_st(c + 12, v4f_ld(c + 12) + acc13);
		c += row_stride;
		v4f_st(c +  0, v4f_ld(c +  0) + acc20);
		v4f_st(c +  4, v4f_ld(c +  4) + acc21);
		v4f_st(c +  8, v4f_ld(c +  8) + acc22);
		v4f_st(c + 12, v4f_ld(c + 12) + acc23);
	} else {
		v4f_st(c +  0, acc00);
		v4f_st(c +  4, acc01);
		v4f_st(c +  8, acc02);
		v4f_st(c + 12, acc03);
		c += row_stride;
		v4f_st(c +  0, acc10);
		v4f_st(c +  4, acc11);
		v4f_st(c +  8, acc12);
		v4f_st(c + 12, acc13);
		c += row_stride;
		v4f_st(c +  0, acc20);
		v4f_st(c +  4, acc21);
		v4f_st(c +  8, acc22);
		v4f_st(c + 12, acc23);
	}
}

void nnp_s4gemm_upto_3x4__psimd(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero(), acc03 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero(), acc12 = v4f_zero(), acc13 = v4f_zero();
	v4f acc20 = v4f_zero(), acc21 = v4f_zero(), acc22 = v4f_zero(), acc23 = v4f_zero();
	do {
		v4f a0, a1, a2;

		a0 = v4f_ld(a);
		a += 4;
		if (mr > 1) {
			a1 = v4f_ld(a);
			a += 4;
			if (mr > 2) {
				a2 = v4f_ld(a);
				a += 4;
			}
		}

		const v4f b0 = v4f_ld(b);
		b += 4;
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;
		if (nr > 1) {
			const v4f b1 = v4f_ld(b);
			b += 4;
			acc01 += a0 * b1;
			acc11 += a1 * b1;
			acc21 += a2 * b1;
			if (nr > 2) {
				const v4f b2 = v4f_ld(b);
				b += 4;
				acc02 += a0 * b2;
				acc12 += a1 * b2;
				acc22 += a2 * b2;
				if (nr > 3) {
					const v4f b3 = v4f_ld(b);
					b += 4;
					acc03 += a0 * b3;
					acc13 += a1 * b3;
					acc23 += a2 * b3;
				}
			}
		}
	} while (--k);

	if (update != 0) {
		v4f_st(c, v4f_ld(c) + acc00);
		if (nr > 1) {
			v4f_st(c + 4, v4f_ld(c + 4) + acc01);
			if (nr > 2) {
				v4f_st(c + 8, v4f_ld(c + 8) + acc02);
				if (nr > 3) {
					v4f_st(c + 12, v4f_ld(c + 12) + acc03);
				}
			}
		}
		if (mr > 1) {
			c += row_stride;
			v4f_st(c, v4f_ld(c) + acc10);
			if (nr > 1) {
				v4f_st(c + 4, v4f_ld(c + 4) + acc11);
				if (nr > 2) {
					v4f_st(c + 8, v4f_ld(c + 8) + acc12);
					if (nr > 3) {
						v4f_st(c + 12, v4f_ld(c + 12) + acc13);
					}
				}
			}
			if (mr > 2) {
				c += row_stride;
				v4f_st(c, v4f_ld(c) + acc20);
				if (nr > 1) {
					v4f_st(c + 4, v4f_ld(c + 4) + acc21);
					if (nr > 2) {
						v4f_st(c + 8, v4f_ld(c + 8) + acc22);
						if (nr > 3) {
							v4f_st(c + 12, v4f_ld(c + 12) + acc23);
						}
					}
				}
			}
		}
	} else {
		v4f_st(c, acc00);
		if (nr > 1) {
			v4f_st(c + 4, acc01);
			if (nr > 2) {
				v4f_st(c + 8, acc02);
				if (nr > 3) {
					v4f_st(c + 12, acc03);
				}
			}
		}
		if (mr > 1) {
			c += row_stride;
			v4f_st(c, acc10);
			if (nr > 1) {
				v4f_st(c + 4, acc11);
				if (nr > 2) {
					v4f_st(c + 8, acc12);
					if (nr > 3) {
						v4f_st(c + 12, acc13);
					}
				}
			}
			if (mr > 2) {
				c += row_stride;
				v4f_st(c, acc20);
				if (nr > 1) {
					v4f_st(c + 4, acc21);
					if (nr > 2) {
						v4f_st(c + 8, acc22);
						if (nr > 3) {
							v4f_st(c + 12, acc23);
						}
					}
				}
			}
		}
	}
}
