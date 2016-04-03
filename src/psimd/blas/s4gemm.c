#include <stddef.h>

#include <nnpack/simd.h>


void nnp_s4gemm1x1__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a);

		const v4f b0 = v4f_ld(b);
		acc00 += a0 * b0;

		a += 4;
		b += 4;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c, v4f_ld(c) + acc00);
	} else {
		v4f_st(c, acc00);
	}
}

void nnp_s4gemm1x2__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a);

		const v4f b0 = v4f_ld(b + 0);
		acc00 += a0 * b0;
		const v4f b1 = v4f_ld(b + 4);
		acc01 += a0 * b1;

		a += 4;
		b += 8;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00);
		v4f_st(c + 4, v4f_ld(c + 4) + acc01);
	} else {
		v4f_st(c + 0, acc00);
		v4f_st(c + 4, acc01);
	}
}

void nnp_s4gemm1x3__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a);

		const v4f b0 = v4f_ld(b + 0);
		acc00 += a0 * b0;
		const v4f b1 = v4f_ld(b + 4);
		acc01 += a0 * b1;
		const v4f b2 = v4f_ld(b + 8);
		acc02 += a0 * b2;

		a +=  4;
		b += 12;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00);
		v4f_st(c + 4, v4f_ld(c + 4) + acc01);
		v4f_st(c + 8, v4f_ld(c + 8) + acc02);
	} else {
		v4f_st(c + 0, acc00);
		v4f_st(c + 4, acc01);
		v4f_st(c + 8, acc02);
	}
}

void nnp_s4gemm1x4__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero(), acc03 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a);

		const v4f b0 = v4f_ld(b +  0);
		acc00 += a0 * b0;
		const v4f b1 = v4f_ld(b +  4);
		acc01 += a0 * b1;
		const v4f b2 = v4f_ld(b +  8);
		acc02 += a0 * b2;
		const v4f b3 = v4f_ld(b + 12);
		acc03 += a0 * b3;

		a +=  4;
		b += 16;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c +  0, v4f_ld(c +  0) + acc00);
		v4f_st(c +  4, v4f_ld(c +  4) + acc01);
		v4f_st(c +  8, v4f_ld(c +  8) + acc02);
		v4f_st(c + 12, v4f_ld(c + 12) + acc03);
	} else {
		v4f_st(c +  0, acc00);
		v4f_st(c +  4, acc01);
		v4f_st(c +  8, acc02);
		v4f_st(c + 12, acc03);
	}
}

void nnp_s4gemm2x1__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero();
	v4f acc10 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);

		const v4f b0 = v4f_ld(b);
		acc00 += a0 * b0;
		acc10 += a1 * b0;

		a += 8;
		b += 4;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c, v4f_ld(c) + acc00);
		c += row_stride;
		v4f_st(c, v4f_ld(c) + acc10);
	} else {
		v4f_st(c, acc00);
		c += row_stride;
		v4f_st(c, acc10);
	}
}

void nnp_s4gemm2x2__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);

		const v4f b0 = v4f_ld(b + 0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		const v4f b1 = v4f_ld(b + 4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;

		a += 8;
		b += 8;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00);
		v4f_st(c + 4, v4f_ld(c + 4) + acc01);
		c += row_stride;
		v4f_st(c + 0, v4f_ld(c + 0) + acc10);
		v4f_st(c + 4, v4f_ld(c + 4) + acc11);
	} else {
		v4f_st(c + 0, acc00);
		v4f_st(c + 4, acc01);
		c += row_stride;
		v4f_st(c + 0, acc10);
		v4f_st(c + 4, acc11);
	}
}

void nnp_s4gemm2x3__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero(), acc12 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);

		const v4f b0 = v4f_ld(b + 0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		const v4f b1 = v4f_ld(b + 4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;
		const v4f b2 = v4f_ld(b + 8);
		acc02 += a0 * b2;
		acc12 += a1 * b2;

		a +=  8;
		b += 12;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00);
		v4f_st(c + 4, v4f_ld(c + 4) + acc01);
		v4f_st(c + 8, v4f_ld(c + 8) + acc02);
		c += row_stride;
		v4f_st(c + 0, v4f_ld(c + 0) + acc10);
		v4f_st(c + 4, v4f_ld(c + 4) + acc11);
		v4f_st(c + 8, v4f_ld(c + 8) + acc12);
	} else {
		v4f_st(c + 0, acc00);
		v4f_st(c + 4, acc01);
		v4f_st(c + 8, acc02);
		c += row_stride;
		v4f_st(c + 0, acc10);
		v4f_st(c + 4, acc11);
		v4f_st(c + 8, acc12);
	}
}

void nnp_s4gemm2x4__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero(), acc03 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero(), acc12 = v4f_zero(), acc13 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);

		const v4f b0 = v4f_ld(b +  0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		const v4f b1 = v4f_ld(b +  4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;
		const v4f b2 = v4f_ld(b +  8);
		acc02 += a0 * b2;
		acc12 += a1 * b2;
		const v4f b3 = v4f_ld(b + 12);
		acc03 += a0 * b3;
		acc13 += a1 * b3;

		a +=  8;
		b += 16;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c +  0, v4f_ld(c +  0) + acc00);
		v4f_st(c +  4, v4f_ld(c +  4) + acc01);
		v4f_st(c +  8, v4f_ld(c +  8) + acc02);
		v4f_st(c + 12, v4f_ld(c + 12) + acc03);
		c += row_stride;
		v4f_st(c +  0, v4f_ld(c +  0) + acc10);
		v4f_st(c +  4, v4f_ld(c +  4) + acc11);
		v4f_st(c +  8, v4f_ld(c +  8) + acc12);
		v4f_st(c + 12, v4f_ld(c + 12) + acc13);
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
	}
}

void nnp_s4gemm3x1__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero();
	v4f acc10 = v4f_zero();
	v4f acc20 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);
		const v4f a2 = v4f_ld(a + 8);

		const v4f b0 = v4f_ld(b);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;

		a += 12;
		b +=  4;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c, v4f_ld(c) + acc00);
		c += row_stride;
		v4f_st(c, v4f_ld(c) + acc10);
		c += row_stride;
		v4f_st(c, v4f_ld(c) + acc20);
	} else {
		v4f_st(c, acc00);
		c += row_stride;
		v4f_st(c, acc10);
		c += row_stride;
		v4f_st(c, acc20);
	}
}

void nnp_s4gemm3x2__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero();
	v4f acc20 = v4f_zero(), acc21 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);
		const v4f a2 = v4f_ld(a + 8);

		const v4f b0 = v4f_ld(b + 0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;
		const v4f b1 = v4f_ld(b + 4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;
		acc21 += a2 * b1;

		a += 12;
		b +=  8;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00);
		v4f_st(c + 4, v4f_ld(c + 4) + acc01);
		c += row_stride;
		v4f_st(c + 0, v4f_ld(c + 0) + acc10);
		v4f_st(c + 4, v4f_ld(c + 4) + acc11);
		c += row_stride;
		v4f_st(c + 0, v4f_ld(c + 0) + acc20);
		v4f_st(c + 4, v4f_ld(c + 4) + acc21);
	} else {
		v4f_st(c + 0, acc00);
		v4f_st(c + 4, acc01);
		c += row_stride;
		v4f_st(c + 0, acc10);
		v4f_st(c + 4, acc11);
		c += row_stride;
		v4f_st(c + 0, acc20);
		v4f_st(c + 4, acc21);
	}
}

void nnp_s4gemm3x3__psimd(
	size_t k, size_t k_tile,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride, size_t column_stride)
{
	v4f acc00 = v4f_zero(), acc01 = v4f_zero(), acc02 = v4f_zero();
	v4f acc10 = v4f_zero(), acc11 = v4f_zero(), acc12 = v4f_zero();
	v4f acc20 = v4f_zero(), acc21 = v4f_zero(), acc22 = v4f_zero();
	do {
		const v4f a0 = v4f_ld(a + 0);
		const v4f a1 = v4f_ld(a + 4);
		const v4f a2 = v4f_ld(a + 8);

		const v4f b0 = v4f_ld(b + 0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;
		const v4f b1 = v4f_ld(b + 4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;
		acc21 += a2 * b1;
		const v4f b2 = v4f_ld(b + 8);
		acc02 += a0 * b2;
		acc12 += a1 * b2;
		acc22 += a2 * b2;

		a += 12;
		b += 12;
	} while (--k);

	if (k_tile != 0) {
		v4f_st(c + 0, v4f_ld(c + 0) + acc00);
		v4f_st(c + 4, v4f_ld(c + 4) + acc01);
		v4f_st(c + 8, v4f_ld(c + 8) + acc02);
		c += row_stride;
		v4f_st(c + 0, v4f_ld(c + 0) + acc10);
		v4f_st(c + 4, v4f_ld(c + 4) + acc11);
		v4f_st(c + 8, v4f_ld(c + 8) + acc12);
		c += row_stride;
		v4f_st(c + 0, v4f_ld(c + 0) + acc20);
		v4f_st(c + 4, v4f_ld(c + 4) + acc21);
		v4f_st(c + 8, v4f_ld(c + 8) + acc22);
	} else {
		v4f_st(c + 0, acc00);
		v4f_st(c + 4, acc01);
		v4f_st(c + 8, acc02);
		c += row_stride;
		v4f_st(c + 0, acc10);
		v4f_st(c + 4, acc11);
		v4f_st(c + 8, acc12);
		c += row_stride;
		v4f_st(c + 0, acc20);
		v4f_st(c + 4, acc21);
		v4f_st(c + 8, acc22);
	}
}

void nnp_s4gemm3x4__psimd(
	size_t k, size_t k_tile,
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

	if (k_tile != 0) {
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
