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
		if (nr > 4) {
			vb1 = v4f_ld(b);
			b += 4;
		}

		const v4f va0 = v4f_splat(*a++);
		vc00 += va0 * vb0;
		vc01 += va0 * vb1;

		if (mr > 1) {
			const v4f va1 = v4f_splat(*a++);
			vc10 += va1 * vb0;
			vc11 += va1 * vb1;

			if (mr > 2) {
				const v4f va2 = v4f_splat(*a++);
				vc20 += va2 * vb0;
				vc21 += va2 * vb1;

				if (mr > 3) {
					const v4f va3 = v4f_splat(*a++);
					vc30 += va3 * vb0;
					vc31 += va3 * vb1;
				}
			}
		}
	} while (--k);

	if (update) {
		v4f vc0n = vc00;
		uint32_t nr0 = nr;
		float* c0n = c;
		if (nr0 > 4) {
			v4f_st(c0n, v4f_ld(c0n) + vc0n);
			c0n += 4;
			nr0 -= 4;
			vc0n = vc01;
		}
		switch (nr0) {
			case 4:
				v4f_st(c0n, v4f_ld(c0n) + vc0n);
				break;
			case 3:
				c0n[2] += vc0n[2];
			case 2:
				c0n[1] += vc0n[1];
			case 1:
				c0n[0] += vc0n[0];
				break;
			default:
				NNP_UNREACHABLE;
		}
		if (mr > 1) {
			c += row_stride_c;
			v4f vc1n = vc10;
			uint32_t nr1 = nr;
			float* c1n = c;
			if (nr1 > 4) {
				v4f_st(c1n, v4f_ld(c1n) + vc1n);
				c1n += 4;
				nr1 -= 4;
				vc1n = vc11;
			}
			switch (nr1) {
				case 4:
					v4f_st(c1n, v4f_ld(c1n) + vc1n);
					break;
				case 3:
					c1n[2] += vc1n[2];
				case 2:
					c1n[1] += vc1n[1];
				case 1:
					c1n[0] += vc1n[0];
					break;
				default:
					NNP_UNREACHABLE;
			}
			if (mr > 2) {
				c += row_stride_c;
				v4f vc2n = vc20;
				uint32_t nr2 = nr;
				float* c2n = c;
				if (nr2 > 4) {
					v4f_st(c2n, v4f_ld(c2n) + vc2n);
					c2n += 4;
					nr2 -= 4;
					vc2n = vc21;
				}
				switch (nr2) {
					case 4:
						v4f_st(c2n, v4f_ld(c2n) + vc2n);
						break;
					case 3:
						c2n[2] += vc2n[2];
					case 2:
						c2n[1] += vc2n[1];
					case 1:
						c2n[0] += vc2n[0];
						break;
					default:
						NNP_UNREACHABLE;
				}
				if (mr > 3) {
					c += row_stride_c;
					v4f vc3n = vc30;
					uint32_t nr3 = nr;
					float* c3n = c;
					if (nr3 > 4) {
						v4f_st(c3n, v4f_ld(c3n) + vc3n);
						c3n += 4;
						nr3 -= 4;
						vc3n = vc31;
					}
					switch (nr3) {
						case 4:
							v4f_st(c3n, v4f_ld(c3n) + vc3n);
							break;
						case 3:
							c3n[2] += vc3n[2];
						case 2:
							c3n[1] += vc3n[1];
						case 1:
							c3n[0] += vc3n[0];
							break;
						default:
							NNP_UNREACHABLE;
					}
				}
			}
		}
	} else {
		v4f vc0n = vc00;
		uint32_t nr0 = nr;
		float* c0n = c;
		if (nr0 > 4) {
			v4f_st(c0n, vc0n);
			c0n += 4;
			nr0 -= 4;
			vc0n = vc01;
		}
		switch (nr0) {
			case 4:
				v4f_st(c0n, vc0n);
				break;
			case 3:
				c0n[2] = vc0n[2];
			case 2:
				c0n[1] = vc0n[1];
			case 1:
				c0n[0] = vc0n[0];
				break;
			default:
				NNP_UNREACHABLE;
		}
		if (mr > 1) {
			c += row_stride_c;
			v4f vc1n = vc10;
			uint32_t nr1 = nr;
			float* c1n = c;
			if (nr1 > 4) {
				v4f_st(c1n, vc1n);
				c1n += 4;
				nr1 -= 4;
				vc1n = vc11;
			}
			switch (nr1) {
				case 4:
					v4f_st(c1n, vc1n);
					break;
				case 3:
					c1n[2] = vc1n[2];
				case 2:
					c1n[1] = vc1n[1];
				case 1:
					c1n[0] = vc1n[0];
					break;
				default:
					NNP_UNREACHABLE;
			}
			if (mr > 2) {
				c += row_stride_c;
				v4f vc2n = vc20;
				uint32_t nr2 = nr;
				float* c2n = c;
				if (nr2 > 4) {
					v4f_st(c2n, vc2n);
					c2n += 4;
					nr2 -= 4;
					vc2n = vc21;
				}
				switch (nr2) {
					case 4:
						v4f_st(c2n, vc2n);
						break;
					case 3:
						c2n[2] = vc2n[2];
					case 2:
						c2n[1] = vc2n[1];
					case 1:
						c2n[0] = vc2n[0];
						break;
					default:
						NNP_UNREACHABLE;
				}
				if (mr > 3) {
					c += row_stride_c;
					v4f vc3n = vc30;
					uint32_t nr3 = nr;
					float* c3n = c;
					if (nr3 > 4) {
						v4f_st(c3n, vc3n);
						c3n += 4;
						nr3 -= 4;
						vc3n = vc31;
					}
					switch (nr3) {
						case 4:
							v4f_st(c3n, vc3n);
							break;
						case 3:
							c3n[2] = vc3n[2];
						case 2:
							c3n[1] = vc3n[1];
						case 1:
							c3n[0] = vc3n[0];
							break;
						default:
							NNP_UNREACHABLE;
					}
				}
			}
		}
	}
}
