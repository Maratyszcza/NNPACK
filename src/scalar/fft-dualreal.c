#include <scalar/fft/dualreal.h>


void nnp_fft8_dualreal__scalar(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	float x0, y0, x1r, y1r, x2r, y2r, x3r, y3r;
	float x4, y4, x1i, y1i, x2i, y2i, x3i, y3i;
	scalar_fft8_dualreal(t,
		&x0, &y0, &x1r, &y1r, &x2r, &y2r, &x3r, &y3r,
		&x4, &y4, &x1i, &y1i, &x2i, &y2i, &x3i, &y3i);

	f[0] = x0;
	f[1] = y0;
	f[2] = x1r;
	f[3] = y1r;
	f[4] = x2r;
	f[5] = y2r;
	f[6] = x3r;
	f[7] = y3r;

	f[ 8] = x4;
	f[ 9] = y4;
	f[10] = x1i;
	f[11] = y1i;
	f[12] = x2i;
	f[13] = y2i;
	f[14] = x3i;
	f[15] = y3i;
}

void nnp_fft16_dualreal__scalar(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	float x0, y0, x1r, y1r, x2r, y2r, x3r, y3r, x4r, y4r, x5r, y5r, x6r, y6r, x7r, y7r;
	float x8, y8, x1i, y1i, x2i, y2i, x3i, y3i, x4i, y4i, x5i, y5i, x6i, y6i, x7i, y7i;
	scalar_fft16_dualreal(t,
		&x0, &y0, &x1r, &y1r, &x2r, &y2r, &x3r, &y3r, &x4r, &y4r, &x5r, &y5r, &x6r, &y6r, &x7r, &y7r,
		&x8, &y8, &x1i, &y1i, &x2i, &y2i, &x3i, &y3i, &x4i, &y4i, &x5i, &y5i, &x6i, &y6i, &x7i, &y7i);

	f[ 0] = x0;
	f[ 1] = y0;
	f[ 2] = x1r;
	f[ 3] = y1r;
	f[ 4] = x2r;
	f[ 5] = y2r;
	f[ 6] = x3r;
	f[ 7] = y3r;
	f[ 8] = x4r;
	f[ 9] = y4r;
	f[10] = x5r;
	f[11] = y5r;
	f[12] = x6r;
	f[13] = y6r;
	f[14] = x7r;
	f[15] = y7r;

	f[16] = x8;
	f[17] = y8;
	f[18] = x1i;
	f[19] = y1i;
	f[20] = x2i;
	f[21] = y2i;
	f[22] = x3i;
	f[23] = y3i;
	f[24] = x4i;
	f[25] = y4i;
	f[26] = x5i;
	f[27] = y5i;
	f[28] = x6i;
	f[29] = y6i;
	f[30] = x7i;
	f[31] = y7i;
}

void nnp_ifft8_dualreal__scalar(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	const float x0  = f[ 0];
	const float y0  = f[ 1];
	const float x1r = f[ 2];
	const float y1r = f[ 3];
	const float x2r = f[ 4];
	const float y2r = f[ 5];
	const float x3r = f[ 6];
	const float y3r = f[ 7];
	const float x4  = f[ 8];
	const float y4  = f[ 9];
	const float x1i = f[10];
	const float y1i = f[11];
	const float x2i = f[12];
	const float y2i = f[13];
	const float x3i = f[14];
	const float y3i = f[15];

	scalar_ifft8_dualreal(
		x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
		x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
		t);
}

void nnp_ifft16_dualreal__scalar(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	const float x0  = f[ 0];
	const float y0  = f[ 1];
	const float x1r = f[ 2];
	const float y1r = f[ 3];
	const float x2r = f[ 4];
	const float y2r = f[ 5];
	const float x3r = f[ 6];
	const float y3r = f[ 7];
	const float x4r = f[ 8];
	const float y4r = f[ 9];
	const float x5r = f[10];
	const float y5r = f[11];
	const float x6r = f[12];
	const float y6r = f[13];
	const float x7r = f[14];
	const float y7r = f[15];

	const float x8  = f[16];
	const float y8  = f[17];
	const float x1i = f[18];
	const float y1i = f[19];
	const float x2i = f[20];
	const float y2i = f[21];
	const float x3i = f[22];
	const float y3i = f[23];
	const float x4i = f[24];
	const float y4i = f[25];
	const float x5i = f[26];
	const float y5i = f[27];
	const float x6i = f[28];
	const float y6i = f[29];
	const float x7i = f[30];
	const float y7i = f[31];

	scalar_ifft16_dualreal(
		x0, y0, x1r, y1r, x2r, y2r, x3r, y3r, x4r, y4r, x5r, y5r, x6r, y6r, x7r, y7r,
		x8, y8, x1i, y1i, x2i, y2i, x3i, y3i, x4i, y4i, x5i, y5i, x6i, y6i, x7i, y7i,
		t);
}
