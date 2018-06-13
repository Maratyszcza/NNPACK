#include <neon/winograd/f6x6k3x3.h>


void nnp_iwt_f6k3__neon(
	const float d[restrict static 32],
	float w[restrict static 32])
{
	float32x4_t w0 = vld1q_f32(d +  0);
	float32x4_t w1 = vld1q_f32(d +  4);
	float32x4_t w2 = vld1q_f32(d +  8);
	float32x4_t w3 = vld1q_f32(d + 12);
	float32x4_t w4 = vld1q_f32(d + 16);
	float32x4_t w5 = vld1q_f32(d + 20);
	float32x4_t w6 = vld1q_f32(d + 24);
	float32x4_t w7 = vld1q_f32(d + 28);

	winograd_f6k3_input_transform(
		w0, w1, w2, w3, w4, w5, w6, w7,
		&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	vst1q_f32(w +  0, w0);
	vst1q_f32(w +  4, w1);
	vst1q_f32(w +  8, w2);
	vst1q_f32(w + 12, w3);
	vst1q_f32(w + 16, w4);
	vst1q_f32(w + 20, w5);
	vst1q_f32(w + 24, w6);
	vst1q_f32(w + 28, w7);
}

void nnp_kwt_f6k3__neon(
	const float g[restrict static 12],
	float w[restrict static 32])
{
	const float32x4_t g0 = vld1q_f32(g + 0);
	const float32x4_t g1 = vld1q_f32(g + 4);
	const float32x4_t g2 = vld1q_f32(g + 8);

	float32x4_t w0, w1, w2, w3, w4, w5, w6, w7;
	winograd_f6k3_kernel_transform(
		g0, g1, g2,
		&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
		true /* rescale coefficients */);

	vst1q_f32(w +  0, w0);
	vst1q_f32(w +  4, w1);
	vst1q_f32(w +  8, w2);
	vst1q_f32(w + 12, w3);
	vst1q_f32(w + 16, w4);
	vst1q_f32(w + 20, w5);
	vst1q_f32(w + 24, w6);
	vst1q_f32(w + 28, w7);
}

void nnp_owt_f6k3__neon(
	const float m[restrict static 32],
	float s[restrict static 24])
{
	float32x4_t w0 = vld1q_f32(m +  0);
	float32x4_t w1 = vld1q_f32(m +  4);
	float32x4_t w2 = vld1q_f32(m +  8);
	float32x4_t w3 = vld1q_f32(m + 12);
	float32x4_t w4 = vld1q_f32(m + 16);
	float32x4_t w5 = vld1q_f32(m + 20);
	float32x4_t w6 = vld1q_f32(m + 24);
	float32x4_t w7 = vld1q_f32(m + 28);

	float32x4_t s0, s1, s2, s3, s4, s5;
	winograd_f6k3_output_transformq(
		w0, w1, w2, w3, w4, w5, w6, w7,
		&s0, &s1, &s2, &s3, &s4, &s5);

	vst1q_f32(s +  0, s0);
	vst1q_f32(s +  4, s1);
	vst1q_f32(s +  8, s2);
	vst1q_f32(s + 12, s3);
	vst1q_f32(s + 16, s4);
	vst1q_f32(s + 20, s5);
}
