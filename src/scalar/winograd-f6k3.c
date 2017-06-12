#include <scalar/winograd/f6x6k3x3.h>


void nnp_iwt_f6k3__scalar(
	const float d[restrict static 8],
	float w[restrict static 8])
{
	winograd_f6k3_input_transform(
		d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
		&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]);
}

void nnp_kwt_f6k3__scalar(
	const float g[restrict static 3],
	float w[restrict static 8])
{
	winograd_f6k3_kernel_transform(
		g[0], g[1], g[2],
		&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7],
		true /* rescale coefficients */);
}

void nnp_owt_f6k3__scalar(
	const float m[restrict static 8],
	float s[restrict static 6])
{
	winograd_f6k3_output_transform(
		m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
		&s[0], &s[1], &s[2], &s[3], &s[4], &s[5]);
}
