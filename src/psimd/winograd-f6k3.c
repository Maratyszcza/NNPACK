#include <psimd/winograd/f6x6k3x3.h>


void nnp_iwt_f6k3__psimd(
	const float d[restrict static 32],
	float w[restrict static 32])
{
	const psimd_f32 d0 = psimd_load_f32(d +  0);
	const psimd_f32 d1 = psimd_load_f32(d +  4);
	const psimd_f32 d2 = psimd_load_f32(d +  8);
	const psimd_f32 d3 = psimd_load_f32(d + 12);
	const psimd_f32 d4 = psimd_load_f32(d + 16);
	const psimd_f32 d5 = psimd_load_f32(d + 20);
	const psimd_f32 d6 = psimd_load_f32(d + 24);
	const psimd_f32 d7 = psimd_load_f32(d + 28);

	psimd_f32 w0, w1, w2, w3, w4, w5, w6, w7;
	winograd_f6k3_input_transform(
		d0, d1, d2, d3, d4, d5, d6, d7,
		&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	psimd_store_f32(w +  0, w0);
	psimd_store_f32(w +  4, w1);
	psimd_store_f32(w +  8, w2);
	psimd_store_f32(w + 12, w3);
	psimd_store_f32(w + 16, w4);
	psimd_store_f32(w + 20, w5);
	psimd_store_f32(w + 24, w6);
	psimd_store_f32(w + 28, w7);
}

void nnp_kwt_f6k3__psimd(
	const float g[restrict static 12],
	float w[restrict static 32])
{
	const psimd_f32 g0 = psimd_load_f32(g + 0);
	const psimd_f32 g1 = psimd_load_f32(g + 4);
	const psimd_f32 g2 = psimd_load_f32(g + 8);

	psimd_f32 w0, w1, w2, w3, w4, w5, w6, w7;
	winograd_f6k3_kernel_transform(
		g0, g1, g2,
		&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
		true /* rescale coefficients */);

	psimd_store_f32(w +  0, w0);
	psimd_store_f32(w +  4, w1);
	psimd_store_f32(w +  8, w2);
	psimd_store_f32(w + 12, w3);
	psimd_store_f32(w + 16, w4);
	psimd_store_f32(w + 20, w5);
	psimd_store_f32(w + 24, w6);
	psimd_store_f32(w + 28, w7);
}

void nnp_owt_f6k3__psimd(
	const float m[restrict static 32],
	float s[restrict static 24])
{
	const psimd_f32 m0 = psimd_load_f32(m +  0);
	const psimd_f32 m1 = psimd_load_f32(m +  4);
	const psimd_f32 m2 = psimd_load_f32(m +  8);
	const psimd_f32 m3 = psimd_load_f32(m + 12);
	const psimd_f32 m4 = psimd_load_f32(m + 16);
	const psimd_f32 m5 = psimd_load_f32(m + 20);
	const psimd_f32 m6 = psimd_load_f32(m + 24);
	const psimd_f32 m7 = psimd_load_f32(m + 28);

	psimd_f32 s0, s1, s2, s3, s4, s5;
	winograd_f6k3_output_transform(
		m0, m1, m2, m3, m4, m5, m6, m7,
		&s0, &s1, &s2, &s3, &s4, &s5);

	psimd_store_f32(s +  0, s0);
	psimd_store_f32(s +  4, s1);
	psimd_store_f32(s +  8, s2);
	psimd_store_f32(s + 12, s3);
	psimd_store_f32(s + 16, s4);
	psimd_store_f32(s + 20, s5);
}
