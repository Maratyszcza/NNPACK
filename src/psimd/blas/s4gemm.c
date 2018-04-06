#include <stddef.h>
#include <stdint.h>

#include <psimd.h>


void nnp_s4gemm_only_3x4__psimd(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	psimd_f32 acc00 = psimd_zero_f32(), acc01 = psimd_zero_f32(), acc02 = psimd_zero_f32(), acc03 = psimd_zero_f32();
	psimd_f32 acc10 = psimd_zero_f32(), acc11 = psimd_zero_f32(), acc12 = psimd_zero_f32(), acc13 = psimd_zero_f32();
	psimd_f32 acc20 = psimd_zero_f32(), acc21 = psimd_zero_f32(), acc22 = psimd_zero_f32(), acc23 = psimd_zero_f32();
	do {
		const psimd_f32 a0 = psimd_load_f32(a + 0);
		const psimd_f32 a1 = psimd_load_f32(a + 4);
		const psimd_f32 a2 = psimd_load_f32(a + 8);

		const psimd_f32 b0 = psimd_load_f32(b +  0);
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;
		const psimd_f32 b1 = psimd_load_f32(b +  4);
		acc01 += a0 * b1;
		acc11 += a1 * b1;
		acc21 += a2 * b1;
		const psimd_f32 b2 = psimd_load_f32(b +  8);
		acc02 += a0 * b2;
		acc12 += a1 * b2;
		acc22 += a2 * b2;
		const psimd_f32 b3 = psimd_load_f32(b + 12);
		acc03 += a0 * b3;
		acc13 += a1 * b3;
		acc23 += a2 * b3;

		a += 12;
		b += 16;
	} while (--k);

	if (update != 0) {
		psimd_store_f32(c +  0, psimd_load_f32(c +  0) + acc00);
		psimd_store_f32(c +  4, psimd_load_f32(c +  4) + acc01);
		psimd_store_f32(c +  8, psimd_load_f32(c +  8) + acc02);
		psimd_store_f32(c + 12, psimd_load_f32(c + 12) + acc03);
		c += row_stride_c;
		psimd_store_f32(c +  0, psimd_load_f32(c +  0) + acc10);
		psimd_store_f32(c +  4, psimd_load_f32(c +  4) + acc11);
		psimd_store_f32(c +  8, psimd_load_f32(c +  8) + acc12);
		psimd_store_f32(c + 12, psimd_load_f32(c + 12) + acc13);
		c += row_stride_c;
		psimd_store_f32(c +  0, psimd_load_f32(c +  0) + acc20);
		psimd_store_f32(c +  4, psimd_load_f32(c +  4) + acc21);
		psimd_store_f32(c +  8, psimd_load_f32(c +  8) + acc22);
		psimd_store_f32(c + 12, psimd_load_f32(c + 12) + acc23);
	} else {
		psimd_store_f32(c +  0, acc00);
		psimd_store_f32(c +  4, acc01);
		psimd_store_f32(c +  8, acc02);
		psimd_store_f32(c + 12, acc03);
		c += row_stride_c;
		psimd_store_f32(c +  0, acc10);
		psimd_store_f32(c +  4, acc11);
		psimd_store_f32(c +  8, acc12);
		psimd_store_f32(c + 12, acc13);
		c += row_stride_c;
		psimd_store_f32(c +  0, acc20);
		psimd_store_f32(c +  4, acc21);
		psimd_store_f32(c +  8, acc22);
		psimd_store_f32(c + 12, acc23);
	}
}

void nnp_s4gemm_upto_3x4__psimd(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	psimd_f32 acc00 = psimd_zero_f32(), acc01 = psimd_zero_f32(), acc02 = psimd_zero_f32(), acc03 = psimd_zero_f32();
	psimd_f32 acc10 = psimd_zero_f32(), acc11 = psimd_zero_f32(), acc12 = psimd_zero_f32(), acc13 = psimd_zero_f32();
	psimd_f32 acc20 = psimd_zero_f32(), acc21 = psimd_zero_f32(), acc22 = psimd_zero_f32(), acc23 = psimd_zero_f32();
	do {
		psimd_f32 a0, a1, a2;

		a0 = psimd_load_f32(a);
		a += 4;
		if (mr > 1) {
			a1 = psimd_load_f32(a);
			a += 4;
			if (mr > 2) {
				a2 = psimd_load_f32(a);
				a += 4;
			}
		}

		const psimd_f32 b0 = psimd_load_f32(b);
		b += 4;
		acc00 += a0 * b0;
		acc10 += a1 * b0;
		acc20 += a2 * b0;
		if (nr > 1) {
			const psimd_f32 b1 = psimd_load_f32(b);
			b += 4;
			acc01 += a0 * b1;
			acc11 += a1 * b1;
			acc21 += a2 * b1;
			if (nr > 2) {
				const psimd_f32 b2 = psimd_load_f32(b);
				b += 4;
				acc02 += a0 * b2;
				acc12 += a1 * b2;
				acc22 += a2 * b2;
				if (nr > 3) {
					const psimd_f32 b3 = psimd_load_f32(b);
					b += 4;
					acc03 += a0 * b3;
					acc13 += a1 * b3;
					acc23 += a2 * b3;
				}
			}
		}
	} while (--k);

	if (update != 0) {
		psimd_store_f32(c, psimd_load_f32(c) + acc00);
		if (nr > 1) {
			psimd_store_f32(c + 4, psimd_load_f32(c + 4) + acc01);
			if (nr > 2) {
				psimd_store_f32(c + 8, psimd_load_f32(c + 8) + acc02);
				if (nr > 3) {
					psimd_store_f32(c + 12, psimd_load_f32(c + 12) + acc03);
				}
			}
		}
		if (mr > 1) {
			c += row_stride_c;
			psimd_store_f32(c, psimd_load_f32(c) + acc10);
			if (nr > 1) {
				psimd_store_f32(c + 4, psimd_load_f32(c + 4) + acc11);
				if (nr > 2) {
					psimd_store_f32(c + 8, psimd_load_f32(c + 8) + acc12);
					if (nr > 3) {
						psimd_store_f32(c + 12, psimd_load_f32(c + 12) + acc13);
					}
				}
			}
			if (mr > 2) {
				c += row_stride_c;
				psimd_store_f32(c, psimd_load_f32(c) + acc20);
				if (nr > 1) {
					psimd_store_f32(c + 4, psimd_load_f32(c + 4) + acc21);
					if (nr > 2) {
						psimd_store_f32(c + 8, psimd_load_f32(c + 8) + acc22);
						if (nr > 3) {
							psimd_store_f32(c + 12, psimd_load_f32(c + 12) + acc23);
						}
					}
				}
			}
		}
	} else {
		psimd_store_f32(c, acc00);
		if (nr > 1) {
			psimd_store_f32(c + 4, acc01);
			if (nr > 2) {
				psimd_store_f32(c + 8, acc02);
				if (nr > 3) {
					psimd_store_f32(c + 12, acc03);
				}
			}
		}
		if (mr > 1) {
			c += row_stride_c;
			psimd_store_f32(c, acc10);
			if (nr > 1) {
				psimd_store_f32(c + 4, acc11);
				if (nr > 2) {
					psimd_store_f32(c + 8, acc12);
					if (nr > 3) {
						psimd_store_f32(c + 12, acc13);
					}
				}
			}
			if (mr > 2) {
				c += row_stride_c;
				psimd_store_f32(c, acc20);
				if (nr > 1) {
					psimd_store_f32(c + 4, acc21);
					if (nr > 2) {
						psimd_store_f32(c + 8, acc22);
						if (nr > 3) {
							psimd_store_f32(c + 12, acc23);
						}
					}
				}
			}
		}
	}
}
