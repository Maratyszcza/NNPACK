#include <stddef.h>
#include <stdint.h>

#include <nnpack/macros.h>


void nnp_sgemm_only_4x3__scalar(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	float acc00, acc01, acc02, acc10, acc11, acc12, acc20, acc21, acc22, acc30, acc31, acc32;
	acc00 = acc01 = acc02 = acc10 = acc11 = acc12 = acc20 = acc21 = acc22 = acc30 = acc31 = acc32 = 0.0f;
	do {
		const float b0 = b[0];
		const float b1 = b[1];
		const float b2 = b[2];
		b += 3;

		const float a0 = a[0];
		acc00 += a0 * b0;
		acc01 += a0 * b1;
		acc02 += a0 * b2;

		const float a1 = a[1];
		acc10 += a1 * b0;
		acc11 += a1 * b1;
		acc12 += a1 * b2;

		const float a2 = a[2];
		acc20 += a2 * b0;
		acc21 += a2 * b1;
		acc22 += a2 * b2;

		const float a3 = a[3];
		acc30 += a3 * b0;
		acc31 += a3 * b1;
		acc32 += a3 * b2;

		a += 4;
	} while (--k);

	if (update) {
		c[0] += acc00;
		c[1] += acc01;
		c[2] += acc02;
		c += row_stride_c;
		c[0] += acc10;
		c[1] += acc11;
		c[2] += acc12;
		c += row_stride_c;
		c[0] += acc20;
		c[1] += acc21;
		c[2] += acc22;
		c += row_stride_c;
		c[0] += acc30;
		c[1] += acc31;
		c[2] += acc32;
	} else {
		c[0] = acc00;
		c[1] = acc01;
		c[2] = acc02;
		c += row_stride_c;
		c[0] = acc10;
		c[1] = acc11;
		c[2] = acc12;
		c += row_stride_c;
		c[0] = acc20;
		c[1] = acc21;
		c[2] = acc22;
		c += row_stride_c;
		c[0] = acc30;
		c[1] = acc31;
		c[2] = acc32;
	}
}

void nnp_sgemm_upto_4x3__scalar(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	float acc00, acc01, acc02, acc10, acc11, acc12, acc20, acc21, acc22, acc30, acc31, acc32;
	acc00 = acc01 = acc02 = acc10 = acc11 = acc12 = acc20 = acc21 = acc22 = acc30 = acc31 = acc32 = 0.0f;
	do {
		float b0, b1, b2;
		
		b0 = *b++;
		if (nr > 1) {
			b1 = *b++;
			if (nr > 2) {
				b2 = *b++;
			}
		}

		const float a0 = *a++;
		acc00 += a0 * b0;
		acc01 += a0 * b1;
		acc02 += a0 * b2;

		if (mr > 1) {
			const float a1 = *a++;
			acc10 += a1 * b0;
			acc11 += a1 * b1;
			acc12 += a1 * b2;

			if (mr > 2) {
				const float a2 = *a++;
				acc20 += a2 * b0;
				acc21 += a2 * b1;
				acc22 += a2 * b2;

				if (mr > 3) {
					const float a3 = *a++;
					acc30 += a3 * b0;
					acc31 += a3 * b1;
					acc32 += a3 * b2;
				}
			}
		}
	} while (--k);

	if (update) {
		switch (nr) {
			case 1:
				c[0] += acc00;
				if (mr > 1) {
					c += row_stride_c;
					c[0] += acc10;
					if (mr > 2) {
						c += row_stride_c;
						c[0] += acc20;
						if (mr > 3) {
							c += row_stride_c;
							c[0] += acc30;
						}
					}
				}
				break;
			case 2:
				c[0] += acc00;
				c[1] += acc01;
				if (mr > 1) {
					c += row_stride_c;
					c[0] += acc10;
					c[1] += acc11;
					if (mr > 2) {
						c += row_stride_c;
						c[0] += acc20;
						c[1] += acc21;
						if (mr > 3) {
							c += row_stride_c;
							c[0] += acc30;
							c[1] += acc31;
						}
					}
				}
				break;
			case 3:
				c[0] += acc00;
				c[1] += acc01;
				c[2] += acc02;
				if (mr > 1) {
					c += row_stride_c;
					c[0] += acc10;
					c[1] += acc11;
					c[2] += acc12;
					if (mr > 2) {
						c += row_stride_c;
						c[0] += acc20;
						c[1] += acc21;
						c[2] += acc22;
						if (mr > 3) {
							c += row_stride_c;
							c[0] += acc30;
							c[1] += acc31;
							c[2] += acc32;
						}
					}
				}
				break;
			default:
				NNP_UNREACHABLE;
		}
	} else {
		switch (nr) {
			case 1:
				c[0] = acc00;
				if (mr > 1) {
					c += row_stride_c;
					c[0] = acc10;
					if (mr > 2) {
						c += row_stride_c;
						c[0] = acc20;
						if (mr > 3) {
							c += row_stride_c;
							c[0] = acc30;
						}
					}
				}
				break;
			case 2:
				c[0] = acc00;
				c[1] = acc01;
				if (mr > 1) {
					c += row_stride_c;
					c[0] = acc10;
					c[1] = acc11;
					if (mr > 2) {
						c += row_stride_c;
						c[0] = acc20;
						c[1] = acc21;
						if (mr > 3) {
							c += row_stride_c;
							c[0] = acc30;
							c[1] = acc31;
						}
					}
				}
				break;
			case 3:
				c[0] = acc00;
				c[1] = acc01;
				c[2] = acc02;
				if (mr > 1) {
					c += row_stride_c;
					c[0] = acc10;
					c[1] = acc11;
					c[2] = acc12;
					if (mr > 2) {
						c += row_stride_c;
						c[0] = acc20;
						c[1] = acc21;
						c[2] = acc22;
						if (mr > 3) {
							c += row_stride_c;
							c[0] = acc30;
							c[1] = acc31;
							c[2] = acc32;
						}
					}
				}
				break;
			default:
				NNP_UNREACHABLE;
		}
	}
}
