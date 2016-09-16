#include <stddef.h>

#include <nnpack/arm_neon.h>


void nnp_sdotxf1__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
	}
	vst1_lane_f32(sum, vacc0, 0); 
}

void nnp_sdotxf2__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
}

void nnp_sdotxf3__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	float32x4_t vacc2q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
		vacc2q = vmuladdq_f32(vacc2q, vx, vld1q_f32(y2));
		y2 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	float32x2_t vacc2 = vadd_f32(vget_low_f32(vacc2q), vget_high_f32(vacc2q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
		vacc2 = vmuladd_f32(vacc2, vx, vld1_f32(y2));
		y2 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	vacc2 = vpadd_f32(vacc2, vacc2);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
		vacc2 = vmuladd_f32(vacc2, vx, vld1_dup_f32(y2));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
	vst1_lane_f32(&sum[2], vacc2, 0); 
}

void nnp_sdotxf4__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	float32x4_t vacc2q = vdupq_n_f32(0.0f);
	float32x4_t vacc3q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
		vacc2q = vmuladdq_f32(vacc2q, vx, vld1q_f32(y2));
		y2 += 4;
		vacc3q = vmuladdq_f32(vacc3q, vx, vld1q_f32(y3));
		y3 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	float32x2_t vacc2 = vadd_f32(vget_low_f32(vacc2q), vget_high_f32(vacc2q));
	float32x2_t vacc3 = vadd_f32(vget_low_f32(vacc3q), vget_high_f32(vacc3q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
		vacc2 = vmuladd_f32(vacc2, vx, vld1_f32(y2));
		y2 += 2;
		vacc3 = vmuladd_f32(vacc3, vx, vld1_f32(y3));
		y3 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	vacc2 = vpadd_f32(vacc2, vacc2);
	vacc3 = vpadd_f32(vacc3, vacc3);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
		vacc2 = vmuladd_f32(vacc2, vx, vld1_dup_f32(y2));
		vacc3 = vmuladd_f32(vacc3, vx, vld1_dup_f32(y3));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
	vst1_lane_f32(&sum[2], vacc2, 0); 
	vst1_lane_f32(&sum[3], vacc3, 0); 
}

void nnp_sdotxf5__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	float32x4_t vacc2q = vdupq_n_f32(0.0f);
	float32x4_t vacc3q = vdupq_n_f32(0.0f);
	float32x4_t vacc4q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
		vacc2q = vmuladdq_f32(vacc2q, vx, vld1q_f32(y2));
		y2 += 4;
		vacc3q = vmuladdq_f32(vacc3q, vx, vld1q_f32(y3));
		y3 += 4;
		vacc4q = vmuladdq_f32(vacc4q, vx, vld1q_f32(y4));
		y4 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	float32x2_t vacc2 = vadd_f32(vget_low_f32(vacc2q), vget_high_f32(vacc2q));
	float32x2_t vacc3 = vadd_f32(vget_low_f32(vacc3q), vget_high_f32(vacc3q));
	float32x2_t vacc4 = vadd_f32(vget_low_f32(vacc4q), vget_high_f32(vacc4q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
		vacc2 = vmuladd_f32(vacc2, vx, vld1_f32(y2));
		y2 += 2;
		vacc3 = vmuladd_f32(vacc3, vx, vld1_f32(y3));
		y3 += 2;
		vacc4 = vmuladd_f32(vacc4, vx, vld1_f32(y4));
		y4 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	vacc2 = vpadd_f32(vacc2, vacc2);
	vacc3 = vpadd_f32(vacc3, vacc3);
	vacc4 = vpadd_f32(vacc4, vacc4);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
		vacc2 = vmuladd_f32(vacc2, vx, vld1_dup_f32(y2));
		vacc3 = vmuladd_f32(vacc3, vx, vld1_dup_f32(y3));
		vacc4 = vmuladd_f32(vacc4, vx, vld1_dup_f32(y4));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
	vst1_lane_f32(&sum[2], vacc2, 0); 
	vst1_lane_f32(&sum[3], vacc3, 0); 
	vst1_lane_f32(&sum[4], vacc4, 0); 
}

void nnp_sdotxf6__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	float32x4_t vacc2q = vdupq_n_f32(0.0f);
	float32x4_t vacc3q = vdupq_n_f32(0.0f);
	float32x4_t vacc4q = vdupq_n_f32(0.0f);
	float32x4_t vacc5q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
		vacc2q = vmuladdq_f32(vacc2q, vx, vld1q_f32(y2));
		y2 += 4;
		vacc3q = vmuladdq_f32(vacc3q, vx, vld1q_f32(y3));
		y3 += 4;
		vacc4q = vmuladdq_f32(vacc4q, vx, vld1q_f32(y4));
		y4 += 4;
		vacc5q = vmuladdq_f32(vacc5q, vx, vld1q_f32(y5));
		y5 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	float32x2_t vacc2 = vadd_f32(vget_low_f32(vacc2q), vget_high_f32(vacc2q));
	float32x2_t vacc3 = vadd_f32(vget_low_f32(vacc3q), vget_high_f32(vacc3q));
	float32x2_t vacc4 = vadd_f32(vget_low_f32(vacc4q), vget_high_f32(vacc4q));
	float32x2_t vacc5 = vadd_f32(vget_low_f32(vacc5q), vget_high_f32(vacc5q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
		vacc2 = vmuladd_f32(vacc2, vx, vld1_f32(y2));
		y2 += 2;
		vacc3 = vmuladd_f32(vacc3, vx, vld1_f32(y3));
		y3 += 2;
		vacc4 = vmuladd_f32(vacc4, vx, vld1_f32(y4));
		y4 += 2;
		vacc5 = vmuladd_f32(vacc5, vx, vld1_f32(y5));
		y5 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	vacc2 = vpadd_f32(vacc2, vacc2);
	vacc3 = vpadd_f32(vacc3, vacc3);
	vacc4 = vpadd_f32(vacc4, vacc4);
	vacc5 = vpadd_f32(vacc5, vacc5);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
		vacc2 = vmuladd_f32(vacc2, vx, vld1_dup_f32(y2));
		vacc3 = vmuladd_f32(vacc3, vx, vld1_dup_f32(y3));
		vacc4 = vmuladd_f32(vacc4, vx, vld1_dup_f32(y4));
		vacc5 = vmuladd_f32(vacc5, vx, vld1_dup_f32(y5));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
	vst1_lane_f32(&sum[2], vacc2, 0); 
	vst1_lane_f32(&sum[3], vacc3, 0); 
	vst1_lane_f32(&sum[4], vacc4, 0); 
	vst1_lane_f32(&sum[5], vacc5, 0); 
}

void nnp_sdotxf7__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	float32x4_t vacc2q = vdupq_n_f32(0.0f);
	float32x4_t vacc3q = vdupq_n_f32(0.0f);
	float32x4_t vacc4q = vdupq_n_f32(0.0f);
	float32x4_t vacc5q = vdupq_n_f32(0.0f);
	float32x4_t vacc6q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	const float *restrict y6 = y5 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
		vacc2q = vmuladdq_f32(vacc2q, vx, vld1q_f32(y2));
		y2 += 4;
		vacc3q = vmuladdq_f32(vacc3q, vx, vld1q_f32(y3));
		y3 += 4;
		vacc4q = vmuladdq_f32(vacc4q, vx, vld1q_f32(y4));
		y4 += 4;
		vacc5q = vmuladdq_f32(vacc5q, vx, vld1q_f32(y5));
		y5 += 4;
		vacc6q = vmuladdq_f32(vacc6q, vx, vld1q_f32(y6));
		y6 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	float32x2_t vacc2 = vadd_f32(vget_low_f32(vacc2q), vget_high_f32(vacc2q));
	float32x2_t vacc3 = vadd_f32(vget_low_f32(vacc3q), vget_high_f32(vacc3q));
	float32x2_t vacc4 = vadd_f32(vget_low_f32(vacc4q), vget_high_f32(vacc4q));
	float32x2_t vacc5 = vadd_f32(vget_low_f32(vacc5q), vget_high_f32(vacc5q));
	float32x2_t vacc6 = vadd_f32(vget_low_f32(vacc6q), vget_high_f32(vacc6q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
		vacc2 = vmuladd_f32(vacc2, vx, vld1_f32(y2));
		y2 += 2;
		vacc3 = vmuladd_f32(vacc3, vx, vld1_f32(y3));
		y3 += 2;
		vacc4 = vmuladd_f32(vacc4, vx, vld1_f32(y4));
		y4 += 2;
		vacc5 = vmuladd_f32(vacc5, vx, vld1_f32(y5));
		y5 += 2;
		vacc6 = vmuladd_f32(vacc6, vx, vld1_f32(y6));
		y6 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	vacc2 = vpadd_f32(vacc2, vacc2);
	vacc3 = vpadd_f32(vacc3, vacc3);
	vacc4 = vpadd_f32(vacc4, vacc4);
	vacc5 = vpadd_f32(vacc5, vacc5);
	vacc6 = vpadd_f32(vacc6, vacc6);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
		vacc2 = vmuladd_f32(vacc2, vx, vld1_dup_f32(y2));
		vacc3 = vmuladd_f32(vacc3, vx, vld1_dup_f32(y3));
		vacc4 = vmuladd_f32(vacc4, vx, vld1_dup_f32(y4));
		vacc5 = vmuladd_f32(vacc5, vx, vld1_dup_f32(y5));
		vacc6 = vmuladd_f32(vacc6, vx, vld1_dup_f32(y6));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
	vst1_lane_f32(&sum[2], vacc2, 0); 
	vst1_lane_f32(&sum[3], vacc3, 0); 
	vst1_lane_f32(&sum[4], vacc4, 0); 
	vst1_lane_f32(&sum[5], vacc5, 0); 
	vst1_lane_f32(&sum[6], vacc6, 0); 
}

void nnp_sdotxf8__neon(
	const float x[restrict static 1],
	const float y[restrict static 1],
	size_t stride_y,
	float sum[restrict static 1],
	size_t n)
{
	float32x4_t vacc0q = vdupq_n_f32(0.0f);
	float32x4_t vacc1q = vdupq_n_f32(0.0f);
	float32x4_t vacc2q = vdupq_n_f32(0.0f);
	float32x4_t vacc3q = vdupq_n_f32(0.0f);
	float32x4_t vacc4q = vdupq_n_f32(0.0f);
	float32x4_t vacc5q = vdupq_n_f32(0.0f);
	float32x4_t vacc6q = vdupq_n_f32(0.0f);
	float32x4_t vacc7q = vdupq_n_f32(0.0f);
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	const float *restrict y6 = y5 + stride_y;
	const float *restrict y7 = y6 + stride_y;
	for (; n >= 4; n -= 4) {
		const float32x4_t vx = vld1q_f32(x);
		x += 4;

		vacc0q = vmuladdq_f32(vacc0q, vx, vld1q_f32(y0));
		y0 += 4;
		vacc1q = vmuladdq_f32(vacc1q, vx, vld1q_f32(y1));
		y1 += 4;
		vacc2q = vmuladdq_f32(vacc2q, vx, vld1q_f32(y2));
		y2 += 4;
		vacc3q = vmuladdq_f32(vacc3q, vx, vld1q_f32(y3));
		y3 += 4;
		vacc4q = vmuladdq_f32(vacc4q, vx, vld1q_f32(y4));
		y4 += 4;
		vacc5q = vmuladdq_f32(vacc5q, vx, vld1q_f32(y5));
		y5 += 4;
		vacc6q = vmuladdq_f32(vacc6q, vx, vld1q_f32(y6));
		y6 += 4;
		vacc7q = vmuladdq_f32(vacc7q, vx, vld1q_f32(y7));
		y7 += 4;
	}
	float32x2_t vacc0 = vadd_f32(vget_low_f32(vacc0q), vget_high_f32(vacc0q));
	float32x2_t vacc1 = vadd_f32(vget_low_f32(vacc1q), vget_high_f32(vacc1q));
	float32x2_t vacc2 = vadd_f32(vget_low_f32(vacc2q), vget_high_f32(vacc2q));
	float32x2_t vacc3 = vadd_f32(vget_low_f32(vacc3q), vget_high_f32(vacc3q));
	float32x2_t vacc4 = vadd_f32(vget_low_f32(vacc4q), vget_high_f32(vacc4q));
	float32x2_t vacc5 = vadd_f32(vget_low_f32(vacc5q), vget_high_f32(vacc5q));
	float32x2_t vacc6 = vadd_f32(vget_low_f32(vacc6q), vget_high_f32(vacc6q));
	float32x2_t vacc7 = vadd_f32(vget_low_f32(vacc7q), vget_high_f32(vacc7q));
	if (n >= 2) {
		n -= 2;

		const float32x2_t vx = vld1_f32(x);
		x += 2;

		vacc0 = vmuladd_f32(vacc0, vx, vld1_f32(y0));
		y0 += 2;
		vacc1 = vmuladd_f32(vacc1, vx, vld1_f32(y1));
		y1 += 2;
		vacc2 = vmuladd_f32(vacc2, vx, vld1_f32(y2));
		y2 += 2;
		vacc3 = vmuladd_f32(vacc3, vx, vld1_f32(y3));
		y3 += 2;
		vacc4 = vmuladd_f32(vacc4, vx, vld1_f32(y4));
		y4 += 2;
		vacc5 = vmuladd_f32(vacc5, vx, vld1_f32(y5));
		y5 += 2;
		vacc6 = vmuladd_f32(vacc6, vx, vld1_f32(y6));
		y6 += 2;
		vacc7 = vmuladd_f32(vacc7, vx, vld1_f32(y7));
		y7 += 2;
	}
	vacc0 = vpadd_f32(vacc0, vacc0);
	vacc1 = vpadd_f32(vacc1, vacc1);
	vacc2 = vpadd_f32(vacc2, vacc2);
	vacc3 = vpadd_f32(vacc3, vacc3);
	vacc4 = vpadd_f32(vacc4, vacc4);
	vacc5 = vpadd_f32(vacc5, vacc5);
	vacc6 = vpadd_f32(vacc6, vacc6);
	vacc7 = vpadd_f32(vacc7, vacc7);
	if (n != 0) {
		const float32x2_t vx = vld1_dup_f32(x);
		vacc0 = vmuladd_f32(vacc0, vx, vld1_dup_f32(y0));
		vacc1 = vmuladd_f32(vacc1, vx, vld1_dup_f32(y1));
		vacc2 = vmuladd_f32(vacc2, vx, vld1_dup_f32(y2));
		vacc3 = vmuladd_f32(vacc3, vx, vld1_dup_f32(y3));
		vacc4 = vmuladd_f32(vacc4, vx, vld1_dup_f32(y4));
		vacc5 = vmuladd_f32(vacc5, vx, vld1_dup_f32(y5));
		vacc6 = vmuladd_f32(vacc6, vx, vld1_dup_f32(y6));
		vacc7 = vmuladd_f32(vacc7, vx, vld1_dup_f32(y7));
	}
	vst1_lane_f32(&sum[0], vacc0, 0); 
	vst1_lane_f32(&sum[1], vacc1, 0); 
	vst1_lane_f32(&sum[2], vacc2, 0); 
	vst1_lane_f32(&sum[3], vacc3, 0); 
	vst1_lane_f32(&sum[4], vacc4, 0); 
	vst1_lane_f32(&sum[5], vacc5, 0); 
	vst1_lane_f32(&sum[6], vacc6, 0); 
	vst1_lane_f32(&sum[7], vacc7, 0); 
}
