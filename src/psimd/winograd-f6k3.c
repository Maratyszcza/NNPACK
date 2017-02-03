#include <psimd/winograd/f6x6k3x3.h>


void nnp_iwt_f6k3__psimd(
	const float d[restrict static 32],
	float w[restrict static 32])
{
	const v4f d0 = v4f_ld(d +  0);
	const v4f d1 = v4f_ld(d +  4);
	const v4f d2 = v4f_ld(d +  8);
	const v4f d3 = v4f_ld(d + 12);
	const v4f d4 = v4f_ld(d + 16);
	const v4f d5 = v4f_ld(d + 20);
	const v4f d6 = v4f_ld(d + 24);
	const v4f d7 = v4f_ld(d + 28);

	v4f w0, w1, w2, w3, w4, w5, w6, w7;
	winograd_f6k3_input_transform(
		d0, d1, d2, d3, d4, d5, d6, d7,
		&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	v4f_st(w +  0, w0);
	v4f_st(w +  4, w1);
	v4f_st(w +  8, w2);
	v4f_st(w + 12, w3);
	v4f_st(w + 16, w4);
	v4f_st(w + 20, w5);
	v4f_st(w + 24, w6);
	v4f_st(w + 28, w7);
}

void nnp_kwt_f6k3__psimd(
	const float g[restrict static 12],
	float w[restrict static 32])
{
	const v4f g0 = v4f_ld(g + 0);
	const v4f g1 = v4f_ld(g + 4);
	const v4f g2 = v4f_ld(g + 8);

	v4f w0, w1, w2, w3, w4, w5, w6, w7;
	winograd_f6k3_kernel_transform(
		g0, g1, g2,
		&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7,
		true /* rescale coefficients */);

	v4f_st(w +  0, w0);
	v4f_st(w +  4, w1);
	v4f_st(w +  8, w2);
	v4f_st(w + 12, w3);
	v4f_st(w + 16, w4);
	v4f_st(w + 20, w5);
	v4f_st(w + 24, w6);
	v4f_st(w + 28, w7);
}

void nnp_owt_f6k3__psimd(
	const float m[restrict static 32],
	float s[restrict static 24])
{
	const v4f m0 = v4f_ld(m +  0);
	const v4f m1 = v4f_ld(m +  4);
	const v4f m2 = v4f_ld(m +  8);
	const v4f m3 = v4f_ld(m + 12);
	const v4f m4 = v4f_ld(m + 16);
	const v4f m5 = v4f_ld(m + 20);
	const v4f m6 = v4f_ld(m + 24);
	const v4f m7 = v4f_ld(m + 28);

	v4f s0, s1, s2, s3, s4, s5;
	winograd_f6k3_output_transform(
		m0, m1, m2, m3, m4, m5, m6, m7,
		&s0, &s1, &s2, &s3, &s4, &s5);

	v4f_st(s +  0, s0);
	v4f_st(s +  4, s1);
	v4f_st(s +  8, s2);
	v4f_st(s + 12, s3);
	v4f_st(s + 16, s4);
	v4f_st(s + 20, s5);
}
