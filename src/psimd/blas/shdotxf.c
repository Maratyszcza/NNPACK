#include <stddef.h>

#include <psimd.h>
#include <fp16.h>
#include <fp16/psimd.h>

void nnp_shdotxf1__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 4;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
	}
	sum[0] = acc0;
}

void nnp_shdotxf2__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1;
	vacc0 = vacc1 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
}

void nnp_shdotxf3__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1, vacc2;
	vacc0 = vacc1 = vacc2 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	const uint16_t *restrict y2 = y1 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;

		const psimd_f32x2 vy2 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y2));
		y2 += 8;
		vacc2 += vx_lo * vy2.lo;
		vacc2 += vx_hi * vy2.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	float acc2 = psimd_reduce_sum_f32(vacc2);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
		acc2 += sx * fp16_alt_to_fp32_value(*y2++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
}

void nnp_shdotxf4__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1, vacc2, vacc3;
	vacc0 = vacc1 = vacc2 = vacc3 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	const uint16_t *restrict y2 = y1 + stride_y;
	const uint16_t *restrict y3 = y2 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;

		const psimd_f32x2 vy2 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y2));
		y2 += 8;
		vacc2 += vx_lo * vy2.lo;
		vacc2 += vx_hi * vy2.hi;

		const psimd_f32x2 vy3 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y3));
		y3 += 8;
		vacc3 += vx_lo * vy3.lo;
		vacc3 += vx_hi * vy3.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	float acc2 = psimd_reduce_sum_f32(vacc2);
	float acc3 = psimd_reduce_sum_f32(vacc3);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
		acc2 += sx * fp16_alt_to_fp32_value(*y2++);
		acc3 += sx * fp16_alt_to_fp32_value(*y3++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
}

void nnp_shdotxf5__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1, vacc2, vacc3, vacc4;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	const uint16_t *restrict y2 = y1 + stride_y;
	const uint16_t *restrict y3 = y2 + stride_y;
	const uint16_t *restrict y4 = y3 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;

		const psimd_f32x2 vy2 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y2));
		y2 += 8;
		vacc2 += vx_lo * vy2.lo;
		vacc2 += vx_hi * vy2.hi;

		const psimd_f32x2 vy3 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y3));
		y3 += 8;
		vacc3 += vx_lo * vy3.lo;
		vacc3 += vx_hi * vy3.hi;

		const psimd_f32x2 vy4 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y4));
		y4 += 8;
		vacc4 += vx_lo * vy4.lo;
		vacc4 += vx_hi * vy4.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	float acc2 = psimd_reduce_sum_f32(vacc2);
	float acc3 = psimd_reduce_sum_f32(vacc3);
	float acc4 = psimd_reduce_sum_f32(vacc4);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
		acc2 += sx * fp16_alt_to_fp32_value(*y2++);
		acc3 += sx * fp16_alt_to_fp32_value(*y3++);
		acc4 += sx * fp16_alt_to_fp32_value(*y4++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
}

void nnp_shdotxf6__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = vacc5 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	const uint16_t *restrict y2 = y1 + stride_y;
	const uint16_t *restrict y3 = y2 + stride_y;
	const uint16_t *restrict y4 = y3 + stride_y;
	const uint16_t *restrict y5 = y4 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;

		const psimd_f32x2 vy2 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y2));
		y2 += 8;
		vacc2 += vx_lo * vy2.lo;
		vacc2 += vx_hi * vy2.hi;

		const psimd_f32x2 vy3 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y3));
		y3 += 8;
		vacc3 += vx_lo * vy3.lo;
		vacc3 += vx_hi * vy3.hi;

		const psimd_f32x2 vy4 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y4));
		y4 += 8;
		vacc4 += vx_lo * vy4.lo;
		vacc4 += vx_hi * vy4.hi;

		const psimd_f32x2 vy5 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y5));
		y5 += 8;
		vacc5 += vx_lo * vy5.lo;
		vacc5 += vx_hi * vy5.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	float acc2 = psimd_reduce_sum_f32(vacc2);
	float acc3 = psimd_reduce_sum_f32(vacc3);
	float acc4 = psimd_reduce_sum_f32(vacc4);
	float acc5 = psimd_reduce_sum_f32(vacc5);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
		acc2 += sx * fp16_alt_to_fp32_value(*y2++);
		acc3 += sx * fp16_alt_to_fp32_value(*y3++);
		acc4 += sx * fp16_alt_to_fp32_value(*y4++);
		acc5 += sx * fp16_alt_to_fp32_value(*y5++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
}

void nnp_shdotxf7__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = vacc5 = vacc6 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	const uint16_t *restrict y2 = y1 + stride_y;
	const uint16_t *restrict y3 = y2 + stride_y;
	const uint16_t *restrict y4 = y3 + stride_y;
	const uint16_t *restrict y5 = y4 + stride_y;
	const uint16_t *restrict y6 = y5 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;

		const psimd_f32x2 vy2 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y2));
		y2 += 8;
		vacc2 += vx_lo * vy2.lo;
		vacc2 += vx_hi * vy2.hi;

		const psimd_f32x2 vy3 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y3));
		y3 += 8;
		vacc3 += vx_lo * vy3.lo;
		vacc3 += vx_hi * vy3.hi;

		const psimd_f32x2 vy4 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y4));
		y4 += 8;
		vacc4 += vx_lo * vy4.lo;
		vacc4 += vx_hi * vy4.hi;

		const psimd_f32x2 vy5 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y5));
		y5 += 8;
		vacc5 += vx_lo * vy5.lo;
		vacc5 += vx_hi * vy5.hi;

		const psimd_f32x2 vy6 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y6));
		y6 += 8;
		vacc6 += vx_lo * vy6.lo;
		vacc6 += vx_hi * vy6.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	float acc2 = psimd_reduce_sum_f32(vacc2);
	float acc3 = psimd_reduce_sum_f32(vacc3);
	float acc4 = psimd_reduce_sum_f32(vacc4);
	float acc5 = psimd_reduce_sum_f32(vacc5);
	float acc6 = psimd_reduce_sum_f32(vacc6);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
		acc2 += sx * fp16_alt_to_fp32_value(*y2++);
		acc3 += sx * fp16_alt_to_fp32_value(*y3++);
		acc4 += sx * fp16_alt_to_fp32_value(*y4++);
		acc5 += sx * fp16_alt_to_fp32_value(*y5++);
		acc6 += sx * fp16_alt_to_fp32_value(*y6++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
}

void nnp_shdotxf8__psimd(
	const float x[restrict static 1],
	const uint16_t y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	psimd_f32 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
	vacc0 = vacc1 = vacc2 = vacc3 = vacc4 = vacc5 = vacc6 = vacc7 = psimd_zero_f32();
	const uint16_t *restrict y0 = y;
	const uint16_t *restrict y1 = y0 + stride_y;
	const uint16_t *restrict y2 = y1 + stride_y;
	const uint16_t *restrict y3 = y2 + stride_y;
	const uint16_t *restrict y4 = y3 + stride_y;
	const uint16_t *restrict y5 = y4 + stride_y;
	const uint16_t *restrict y6 = y5 + stride_y;
	const uint16_t *restrict y7 = y6 + stride_y;
	for (; n >= 8; n -= 8) {
		const psimd_f32 vx_lo = psimd_load_f32(x);
		const psimd_f32 vx_hi = psimd_load_f32(x + 4);
		x += 8;

		const psimd_f32x2 vy0 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y0));
		y0 += 8;
		vacc0 += vx_lo * vy0.lo;
		vacc0 += vx_hi * vy0.hi;

		const psimd_f32x2 vy1 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y1));
		y1 += 8;
		vacc1 += vx_lo * vy1.lo;
		vacc1 += vx_hi * vy1.hi;

		const psimd_f32x2 vy2 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y2));
		y2 += 8;
		vacc2 += vx_lo * vy2.lo;
		vacc2 += vx_hi * vy2.hi;

		const psimd_f32x2 vy3 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y3));
		y3 += 8;
		vacc3 += vx_lo * vy3.lo;
		vacc3 += vx_hi * vy3.hi;

		const psimd_f32x2 vy4 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y4));
		y4 += 8;
		vacc4 += vx_lo * vy4.lo;
		vacc4 += vx_hi * vy4.hi;

		const psimd_f32x2 vy5 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y5));
		y5 += 8;
		vacc5 += vx_lo * vy5.lo;
		vacc5 += vx_hi * vy5.hi;

		const psimd_f32x2 vy6 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y6));
		y6 += 8;
		vacc6 += vx_lo * vy6.lo;
		vacc6 += vx_hi * vy6.hi;

		const psimd_f32x2 vy7 =
			fp16_alt_to_fp32x2_psimd(psimd_load_u16(y7));
		y7 += 8;
		vacc7 += vx_lo * vy7.lo;
		vacc7 += vx_hi * vy7.hi;
	}
	float acc0 = psimd_reduce_sum_f32(vacc0);
	float acc1 = psimd_reduce_sum_f32(vacc1);
	float acc2 = psimd_reduce_sum_f32(vacc2);
	float acc3 = psimd_reduce_sum_f32(vacc3);
	float acc4 = psimd_reduce_sum_f32(vacc4);
	float acc5 = psimd_reduce_sum_f32(vacc5);
	float acc6 = psimd_reduce_sum_f32(vacc6);
	float acc7 = psimd_reduce_sum_f32(vacc7);
	while (n--) {
		const float sx = (*x++);
		acc0 += sx * fp16_alt_to_fp32_value(*y0++);
		acc1 += sx * fp16_alt_to_fp32_value(*y1++);
		acc2 += sx * fp16_alt_to_fp32_value(*y2++);
		acc3 += sx * fp16_alt_to_fp32_value(*y3++);
		acc4 += sx * fp16_alt_to_fp32_value(*y4++);
		acc5 += sx * fp16_alt_to_fp32_value(*y5++);
		acc6 += sx * fp16_alt_to_fp32_value(*y6++);
		acc7 += sx * fp16_alt_to_fp32_value(*y7++);
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
