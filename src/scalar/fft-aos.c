#include <scalar/fft/aos.h>


void nnp_fft4_aos__scalar(
	const float t[restrict static 8],
	float f[restrict static 8])
{
	float w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	scalar_fft4_aos(
		t, t + 4, 1, 0, 8,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i);
	f[0] = w0r;
	f[1] = w0i;
	f[2] = w1r;
	f[3] = w1i;
	f[4] = w2r;
	f[5] = w2i;
	f[6] = w3r;
	f[7] = w3i;
}

void nnp_fft8_aos__scalar(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	float w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	scalar_fft8_aos(
		t, t + 8, 1, 0, 16,
		&w0r, &w0i, &w1r, &w1i, &w2r, &w2i, &w3r, &w3i, &w4r, &w4i, &w5r, &w5i, &w6r, &w6i, &w7r, &w7i);
	f[ 0] = w0r;
	f[ 1] = w0i;
	f[ 2] = w1r;
	f[ 3] = w1i;
	f[ 4] = w2r;
	f[ 5] = w2i;
	f[ 6] = w3r;
	f[ 7] = w3i;
	f[ 8] = w4r;
	f[ 9] = w4i;
	f[10] = w5r;
	f[11] = w5i;
	f[12] = w6r;
	f[13] = w6i;
	f[14] = w7r;
	f[15] = w7i;
}

void nnp_ifft4_aos__scalar(
	const float f[restrict static 8],
	float t[restrict static 8])
{
	const float w0r = f[0];
	const float w0i = f[1];
	const float w1r = f[2];
	const float w1i = f[3];
	const float w2r = f[4];
	const float w2i = f[5];
	const float w3r = f[6];
	const float w3i = f[7];

	scalar_ifft4_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i,
		t, t + 4, 1);
}

void nnp_ifft8_aos__scalar(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	const float w0r = f[ 0];
	const float w0i = f[ 1];
	const float w1r = f[ 2];
	const float w1i = f[ 3];
	const float w2r = f[ 4];
	const float w2i = f[ 5];
	const float w3r = f[ 6];
	const float w3i = f[ 7];
	const float w4r = f[ 8];
	const float w4i = f[ 9];
	const float w5r = f[10];
	const float w5i = f[11];
	const float w6r = f[12];
	const float w6i = f[13];
	const float w7r = f[14];
	const float w7i = f[15];

	scalar_ifft8_aos(
		w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
		t, t + 8, 1);
}
