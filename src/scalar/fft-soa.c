#include <scalar/fft/soa.h>


void nnp_fft8_soa__scalar(
	const float t[restrict static 16],
	float f[restrict static 16])
{
	float f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r;
	float f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i;
	scalar_fft8_soa(t,
		&f0r, &f1r, &f2r, &f3r, &f4r, &f5r, &f6r, &f7r,
		&f0i, &f1i, &f2i, &f3i, &f4i, &f5i, &f6i, &f7i);

	f[0] = f0r;
	f[1] = f1r;
	f[2] = f2r;
	f[3] = f3r;
	f[4] = f4r;
	f[5] = f5r;
	f[6] = f6r;
	f[7] = f7r;

	f[ 8] = f0i;
	f[ 9] = f1i;
	f[10] = f2i;
	f[11] = f3i;
	f[12] = f4i;
	f[13] = f5i;
	f[14] = f6i;
	f[15] = f7i;
}

void nnp_fft16_soa__scalar(
	const float t[restrict static 32],
	float f[restrict static 32])
{
	float f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r, f13r, f14r, f15r;
	float f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i, f8i, f9i, f10i, f11i, f12i, f13i, f14i, f15i;
	scalar_fft16_soa(t,
		&f0r, &f1r, &f2r, &f3r, &f4r, &f5r, &f6r, &f7r, &f8r, &f9r, &f10r, &f11r, &f12r, &f13r, &f14r, &f15r,
		&f0i, &f1i, &f2i, &f3i, &f4i, &f5i, &f6i, &f7i, &f8i, &f9i, &f10i, &f11i, &f12i, &f13i, &f14i, &f15i);

	f[ 0] = f0r;
	f[ 1] = f1r;
	f[ 2] = f2r;
	f[ 3] = f3r;
	f[ 4] = f4r;
	f[ 5] = f5r;
	f[ 6] = f6r;
	f[ 7] = f7r;
	f[ 8] = f8r;
	f[ 9] = f9r;
	f[10] = f10r;
	f[11] = f11r;
	f[12] = f12r;
	f[13] = f13r;
	f[14] = f14r;
	f[15] = f15r;

	f[16] = f0i;
	f[17] = f1i;
	f[18] = f2i;
	f[19] = f3i;
	f[20] = f4i;
	f[21] = f5i;
	f[22] = f6i;
	f[23] = f7i;
	f[24] = f8i;
	f[25] = f9i;
	f[26] = f10i;
	f[27] = f11i;
	f[28] = f12i;
	f[29] = f13i;
	f[30] = f14i;
	f[31] = f15i;
}

void nnp_ifft8_soa__scalar(
	const float f[restrict static 16],
	float t[restrict static 16])
{
	const float f0r = f[0];
	const float f1r = f[1];
	const float f2r = f[2];
	const float f3r = f[3];
	const float f4r = f[4];
	const float f5r = f[5];
	const float f6r = f[6];
	const float f7r = f[7];

	const float f0i = f[ 8];
	const float f1i = f[ 9];
	const float f2i = f[10];
	const float f3i = f[11];
	const float f4i = f[12];
	const float f5i = f[13];
	const float f6i = f[14];
	const float f7i = f[15];

	scalar_ifft8_soa(
		f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
		f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
		t);
}

void nnp_ifft16_soa__scalar(
	const float f[restrict static 32],
	float t[restrict static 32])
{
	const float f0r  = f[ 0];
	const float f1r  = f[ 1];
	const float f2r  = f[ 2];
	const float f3r  = f[ 3];
	const float f4r  = f[ 4];
	const float f5r  = f[ 5];
	const float f6r  = f[ 6];
	const float f7r  = f[ 7];
	const float f8r  = f[ 8];
	const float f9r  = f[ 9];
	const float f10r = f[10];
	const float f11r = f[11];
	const float f12r = f[12];
	const float f13r = f[13];
	const float f14r = f[14];
	const float f15r = f[15];

	const float f0i  = f[16];
	const float f1i  = f[17];
	const float f2i  = f[18];
	const float f3i  = f[19];
	const float f4i  = f[20];
	const float f5i  = f[21];
	const float f6i  = f[22];
	const float f7i  = f[23];
	const float f8i  = f[24];
	const float f9i  = f[25];
	const float f10i = f[26];
	const float f11i = f[27];
	const float f12i = f[28];
	const float f13i = f[29];
	const float f14i = f[30];
	const float f15i = f[31];

	scalar_ifft16_soa(
		f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r, f8r, f9r, f10r, f11r, f12r, f13r, f14r, f15r,
		f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i, f8i, f9i, f10i, f11i, f12i, f13i, f14i, f15i,
		t);
}
