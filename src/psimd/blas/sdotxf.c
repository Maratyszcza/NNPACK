#include <stddef.h>

#include <nnpack/simd.h>


void nnp_sdotxf1__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0 = v4f_zero();
	const float *restrict y0 = y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
	}
	sum[0] = acc0;
}

void nnp_sdotxf2__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1;
	vacc0 = vacc1 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
}

void nnp_sdotxf3__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1, vacc2;
	vacc0 = vacc1 = vacc2 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
		vacc2 += vx * v4f_ld(y2);
		y2 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	float acc2 = v4f_reduce_sum(vacc2);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
		acc2 += sx * (*y2++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
}

void nnp_sdotxf4__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1, vacc2, vacc3;
	vacc0 = vacc1 = vacc2 = vacc3 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
		vacc2 += vx * v4f_ld(y2);
		y2 += 4;
		vacc3 += vx * v4f_ld(y3);
		y3 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	float acc2 = v4f_reduce_sum(vacc2);
	float acc3 = v4f_reduce_sum(vacc3);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
		acc2 += sx * (*y2++);
		acc3 += sx * (*y3++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
}

void nnp_sdotxf5__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1, vacc2, vacc3, vacc4;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
		vacc2 += vx * v4f_ld(y2);
		y2 += 4;
		vacc3 += vx * v4f_ld(y3);
		y3 += 4;
		vacc4 += vx * v4f_ld(y4);
		y4 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	float acc2 = v4f_reduce_sum(vacc2);
	float acc3 = v4f_reduce_sum(vacc3);
	float acc4 = v4f_reduce_sum(vacc4);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
		acc2 += sx * (*y2++);
		acc3 += sx * (*y3++);
		acc4 += sx * (*y4++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
}

void nnp_sdotxf6__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1, vacc2, vacc3, vacc4, vacc5;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = vacc5 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
		vacc2 += vx * v4f_ld(y2);
		y2 += 4;
		vacc3 += vx * v4f_ld(y3);
		y3 += 4;
		vacc4 += vx * v4f_ld(y4);
		y4 += 4;
		vacc5 += vx * v4f_ld(y5);
		y5 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	float acc2 = v4f_reduce_sum(vacc2);
	float acc3 = v4f_reduce_sum(vacc3);
	float acc4 = v4f_reduce_sum(vacc4);
	float acc5 = v4f_reduce_sum(vacc5);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
		acc2 += sx * (*y2++);
		acc3 += sx * (*y3++);
		acc4 += sx * (*y4++);
		acc5 += sx * (*y5++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
}

void nnp_sdotxf7__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = vacc5 = vacc6 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	const float *restrict y6 = y5 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
		vacc2 += vx * v4f_ld(y2);
		y2 += 4;
		vacc3 += vx * v4f_ld(y3);
		y3 += 4;
		vacc4 += vx * v4f_ld(y4);
		y4 += 4;
		vacc5 += vx * v4f_ld(y5);
		y5 += 4;
		vacc6 += vx * v4f_ld(y6);
		y6 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	float acc2 = v4f_reduce_sum(vacc2);
	float acc3 = v4f_reduce_sum(vacc3);
	float acc4 = v4f_reduce_sum(vacc4);
	float acc5 = v4f_reduce_sum(vacc5);
	float acc6 = v4f_reduce_sum(vacc6);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
		acc2 += sx * (*y2++);
		acc3 += sx * (*y3++);
		acc4 += sx * (*y4++);
		acc5 += sx * (*y5++);
		acc6 += sx * (*y6++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
}

void nnp_sdotxf8__psimd(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	v4f vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = vacc5 = vacc6 = vacc7 = v4f_zero();
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	const float *restrict y6 = y5 + stride_y;
	const float *restrict y7 = y6 + stride_y;
	for (; n >= 4; n -= 4) {
		const v4f vx = v4f_ld(x);
		x += 4;

		vacc0 += vx * v4f_ld(y0);
		y0 += 4;
		vacc1 += vx * v4f_ld(y1);
		y1 += 4;
		vacc2 += vx * v4f_ld(y2);
		y2 += 4;
		vacc3 += vx * v4f_ld(y3);
		y3 += 4;
		vacc4 += vx * v4f_ld(y4);
		y4 += 4;
		vacc5 += vx * v4f_ld(y5);
		y5 += 4;
		vacc6 += vx * v4f_ld(y6);
		y6 += 4;
		vacc7 += vx * v4f_ld(y7);
		y7 += 4;
	}
	float acc0 = v4f_reduce_sum(vacc0);
	float acc1 = v4f_reduce_sum(vacc1);
	float acc2 = v4f_reduce_sum(vacc2);
	float acc3 = v4f_reduce_sum(vacc3);
	float acc4 = v4f_reduce_sum(vacc4);
	float acc5 = v4f_reduce_sum(vacc5);
	float acc6 = v4f_reduce_sum(vacc6);
	float acc7 = v4f_reduce_sum(vacc7);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * (*y0++);
		acc1 += sx * (*y1++);
		acc2 += sx * (*y2++);
		acc3 += sx * (*y3++);
		acc4 += sx * (*y4++);
		acc5 += sx * (*y5++);
		acc6 += sx * (*y6++);
		acc7 += sx * (*y7++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
	sum[7] = acc7;
}
