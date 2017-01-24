#include <stddef.h>
#include <stdint.h>


void nnp_cgemm_only_2x2__scalar(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float acc00r, acc01r, acc10r, acc11r;
	float acc00i, acc01i, acc10i, acc11i;
	acc00r = acc01r = acc10r = acc11r = 0.0f;
	acc00i = acc01i = acc10i = acc11i = 0.0f;
	do {
		const float a0r = a[0];
		const float a1r = a[2];
		const float a0i = a[1];
		const float a1i = a[3];
		a += 4;

		const float b0r = b[0];
		const float b1r = b[2];
		acc00r += a0r * b0r;
		acc01r += a0r * b1r;
		acc10r += a1r * b0r;
		acc11r += a1r * b1r;
		acc00i += a0i * b0r;
		acc01i += a0i * b1r;
		acc10i += a1i * b0r;
		acc11i += a1i * b1r;

		const float b0i = b[1];
		const float b1i = b[3];
		b += 4;

		acc00r -= a0i * b0i;
		acc01r -= a0i * b1i;
		acc10r -= a1i * b0i;
		acc11r -= a1i * b1i;
		acc00i += a0r * b0i;
		acc01i += a0r * b1i;
		acc10i += a1r * b0i;
		acc11i += a1r * b1i;
	} while (--k);

	if (update != 0) {
		c[0] += acc00r;
		c[1] += acc00i;
		c[2] += acc01r;
		c[3] += acc01i;
		c += row_stride_c;
		c[0] += acc10r;
		c[1] += acc10i;
		c[2] += acc11r;
		c[3] += acc11i;
	} else {
		c[0] = acc00r;
		c[1] = acc00i;
		c[2] = acc01r;
		c[3] = acc01i;
		c += row_stride_c;
		c[0] = acc10r;
		c[1] = acc10i;
		c[2] = acc11r;
		c[3] = acc11i;
	}
}

void nnp_cgemm_upto_2x2__scalar(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float acc00r, acc01r, acc10r, acc11r;
	float acc00i, acc01i, acc10i, acc11i;
	acc00r = acc01r = acc10r = acc11r = 0.0f;
	acc00i = acc01i = acc10i = acc11i = 0.0f;
	do {
		const float a0r = a[0];
		const float a0i = a[1];
		a += 2;

		float a1r, a1i;
		if (mr > 1) {
			a1r = a[0];
			a1i = a[1];
			a += 2;
		}

		const float b0r = b[0];
		const float b0i = b[1];
		b += 2;

		acc00r += a0r * b0r;
		acc10r += a1r * b0r;
		acc00i += a0i * b0r;
		acc10i += a1i * b0r;

		acc00r -= a0i * b0i;
		acc10r -= a1i * b0i;
		acc00i += a0r * b0i;
		acc10i += a1r * b0i;

		if (nr > 1) {
			const float b1r = b[0];
			const float b1i = b[1];
			b += 2;

			acc01r += a0r * b1r;
			acc11r += a1r * b1r;
			acc01i += a0i * b1r;
			acc11i += a1i * b1r;

			acc01r -= a0i * b1i;
			acc11r -= a1i * b1i;
			acc01i += a0r * b1i;
			acc11i += a1r * b1i;
		}
	} while (--k);

	if (update != 0) {
		c[0] += acc00r;
		c[1] += acc00i;
		if (nr > 1) {
			c[2] += acc01r;
			c[3] += acc01i;
		}
		if (mr > 1) {
			c += row_stride_c;
			c[0] += acc10r;
			c[1] += acc10i;
			if (nr > 1) {
				c[2] += acc11r;
				c[3] += acc11i;
			}
		}
	} else {
		c[0] = acc00r;
		c[1] = acc00i;
		if (nr > 1) {
			c[2] = acc01r;
			c[3] = acc01i;
		}
		if (mr > 1) {
			c += row_stride_c;
			c[0] = acc10r;
			c[1] = acc10i;
			if (nr > 1) {
				c[2] = acc11r;
				c[3] = acc11i;
			}
		}
	}
}
