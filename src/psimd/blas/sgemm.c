#include <stddef.h>
#include <stdint.h>

#include <nnpack/simd.h>


void nnp_sgemm_only_4x8__psimd(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	v4f vc00, vc01, vc10, vc11, vc20, vc21, vc30, vc31;
	vc00 = vc01 = vc10 = vc11 = vc20 = vc21 = vc30 = vc31 = v4f_zero();
	do {
		const v4f va = v4f_ld(a);
		a += 4;

		const v4f vb0 = v4f_ld(b + 0);
		const v4f vb1 = v4f_ld(b + 4);
		b += 8;

		const v4f va0 = __builtin_shufflevector(va, va, 0, 0, 0, 0);
		vc00 += va0 * vb0;
		vc01 += va0 * vb1;

		const v4f va1 = __builtin_shufflevector(va, va, 1, 1, 1, 1);
		vc10 += va1 * vb0;
		vc11 += va1 * vb1;

		const v4f va2 = __builtin_shufflevector(va, va, 2, 2, 2, 2);
		vc20 += va2 * vb0;
		vc21 += va2 * vb1;

		const v4f va3 = __builtin_shufflevector(va, va, 3, 3, 3, 3);
		vc30 += va3 * vb0;
		vc31 += va3 * vb1;
	} while (--k);

	if (update) {
		v4f_st(c + 0, v4f_ld(c + 0) + vc00);
		v4f_st(c + 4, v4f_ld(c + 4) + vc01);
		c += row_stride_c;
		v4f_st(c + 0, v4f_ld(c + 0) + vc10);
		v4f_st(c + 4, v4f_ld(c + 4) + vc11);
		c += row_stride_c;
		v4f_st(c + 0, v4f_ld(c + 0) + vc20);
		v4f_st(c + 4, v4f_ld(c + 4) + vc21);
		c += row_stride_c;
		v4f_st(c + 0, v4f_ld(c + 0) + vc30);
		v4f_st(c + 4, v4f_ld(c + 4) + vc31);
	} else {
		v4f_st(c + 0, vc00);
		v4f_st(c + 4, vc01);
		c += row_stride_c;
		v4f_st(c + 0, vc10);
		v4f_st(c + 4, vc11);
		c += row_stride_c;
		v4f_st(c + 0, vc20);
		v4f_st(c + 4, vc21);
		c += row_stride_c;
		v4f_st(c + 0, vc30);
		v4f_st(c + 4, vc31);
	}
}

void nnp_sgemm_upto_4x8__psimd(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	v4f vc00, vc01, vc10, vc11, vc20, vc21, vc30, vc31;
	vc00 = vc01 = vc10 = vc11 = vc20 = vc21 = vc30 = vc31 = v4f_zero();
	do {
		v4f vb0, vb1;
		
		vb0 = v4f_ld(b);
		b += 4;
		if (nr >= 4) {
			vb1 = v4f_ld(b);
			b += 4;
		}

		const v4f va0 = v4f_splat(*a++);
		vc00 += va0 * vb0;
		vc01 += va0 * vb1;

		if (mr >= 2) {
			const v4f va1 = v4f_splat(*a++);
			vc10 += va1 * vb0;
			vc11 += va1 * vb1;

			if (mr > 3) {
				const v4f va2 = v4f_splat(*a++);
				vc20 += va2 * vb0;
				vc21 += va2 * vb1;

				if (mr >= 4) {
					const v4f va3 = v4f_splat(*a++);
					vc30 += va3 * vb0;
					vc31 += va3 * vb1;
				}
			}
		}
	} while (--k);

	NNP_ALIGN(16) float block[4][8];
	v4f_st(&block[0][0], vc00);
	v4f_st(&block[0][4], vc01);
	v4f_st(&block[1][0], vc10);
	v4f_st(&block[1][4], vc11);
	v4f_st(&block[2][0], vc20);
	v4f_st(&block[2][4], vc21);
	v4f_st(&block[3][0], vc30);
	v4f_st(&block[3][4], vc31);
	if (update) {
		for (size_t m = 0; m < mr; m++) {
			for (size_t n = 0; n < nr; n++) {
				c[n] += block[m][n];
			}
			c += row_stride_c;
		}
	} else {
		for (size_t m = 0; m < mr; m++) {
			for (size_t n = 0; n < nr; n++) {
				c[n] = block[m][n];
			}
			c += row_stride_c;
		}
	}
}
