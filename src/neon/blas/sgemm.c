#include <stddef.h>
#include <stdint.h>

#include <nnpack/macros.h>
#include <nnpack/arm_neon.h>


void nnp_sgemm_only_4x12__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
	float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
	float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
	float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
	do {
		const float32x4_t va = vld1q_f32_aligned(a);
		a += 4;

		const float32x4_t vb0 = vld1q_f32_aligned(b + 0);
		const float32x4_t vb1 = vld1q_f32_aligned(b + 4);
		const float32x4_t vb2 = vld1q_f32_aligned(b + 8);
		b += 12;

		#if defined(__aarch64__)
			vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va), 0);
			vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va), 1);
			vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
			vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
			vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va), 0);
			vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va), 1);
			vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
			vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
			vc02 = vfmaq_lane_f32(vc02, vb2, vget_low_f32(va), 0);
			vc12 = vfmaq_lane_f32(vc12, vb2, vget_low_f32(va), 1);
			vc22 = vfmaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
			vc32 = vfmaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
		#else
			vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va), 0);
			vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va), 1);
			vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va), 0);
			vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va), 1);
			vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va), 0);
			vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va), 1);
			vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va), 0);
			vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va), 1);
			vc02 = vmlaq_lane_f32(vc02, vb2, vget_low_f32(va), 0);
			vc12 = vmlaq_lane_f32(vc12, vb2, vget_low_f32(va), 1);
			vc22 = vmlaq_lane_f32(vc22, vb2, vget_high_f32(va), 0);
			vc32 = vmlaq_lane_f32(vc32, vb2, vget_high_f32(va), 1);
		#endif
	} while (--k);

	if (update) {
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
		vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc02));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
		vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc12));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
		vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc22));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
		vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8), vc32));
	} else {
		vst1q_f32(c + 0, vc00);
		vst1q_f32(c + 4, vc01);
		vst1q_f32(c + 8, vc02);
		c += row_stride_c;
		vst1q_f32(c + 0, vc10);
		vst1q_f32(c + 4, vc11);
		vst1q_f32(c + 8, vc12);
		c += row_stride_c;
		vst1q_f32(c + 0, vc20);
		vst1q_f32(c + 4, vc21);
		vst1q_f32(c + 8, vc22);
		c += row_stride_c;
		vst1q_f32(c + 0, vc30);
		vst1q_f32(c + 4, vc31);
		vst1q_f32(c + 8, vc32);
	}
}

void nnp_sgemm_upto_4x12__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f), vc02 = vdupq_n_f32(0.0f);
	float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f), vc12 = vdupq_n_f32(0.0f);
	float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f), vc22 = vdupq_n_f32(0.0f);
	float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f), vc32 = vdupq_n_f32(0.0f);
	do {
		float32x4_t vb0, vb1, vb2;
		
		vb0 = vld1q_f32_aligned(b);
		b += 4;
		if (nr > 4) {
			vb1 = vld1q_f32_aligned(b);
			b += 4;
			if (nr > 8) {
				vb2 = vld1q_f32_aligned(b);
				b += 4;
			}
		}

		const float32x4_t va0 = vld1q_dup_f32(a);
		a += 1;
		vc00 = vmuladdq_f32(vc00, va0, vb0);
		vc01 = vmuladdq_f32(vc01, va0, vb1);
		vc02 = vmuladdq_f32(vc02, va0, vb2);

		if (mr > 1) {
			const float32x4_t va1 = vld1q_dup_f32(a);
			a += 1;
			vc10 = vmuladdq_f32(vc10, va1, vb0);
			vc11 = vmuladdq_f32(vc11, va1, vb1);
			vc12 = vmuladdq_f32(vc12, va1, vb2);

			if (mr > 2) {
				const float32x4_t va2 = vld1q_dup_f32(a);
				a += 1;
				vc20 = vmuladdq_f32(vc20, va2, vb0);
				vc21 = vmuladdq_f32(vc21, va2, vb1);
				vc22 = vmuladdq_f32(vc22, va2, vb2);

				if (mr > 3) {
					const float32x4_t va3 = vld1q_dup_f32(a);
					a += 1;
					vc30 = vmuladdq_f32(vc30, va3, vb0);
					vc31 = vmuladdq_f32(vc31, va3, vb1);
					vc32 = vmuladdq_f32(vc32, va3, vb2);
				}
			}
		}
	} while (--k);

	if (update) {
		float32x4_t vc0n = vc00;
		uint32_t nr0 = nr;
		float* c0n = c;
		if (nr0 > 4) {
			vst1q_f32(c0n, vaddq_f32(vld1q_f32(c0n), vc0n));
			c0n += 4;
			nr0 -= 4;
			vc0n = vc01;
			if (nr0 > 4) {
				vst1q_f32(c0n, vaddq_f32(vld1q_f32(c0n), vc0n));
				c0n += 4;
				nr0 -= 4;
				vc0n = vc02;
			}
		}
		switch (nr0) {
			case 4:
				vst1q_f32(c0n, vaddq_f32(vld1q_f32(c0n), vc0n));
				break;
			case 3:
				vst1_lane_f32(c0n + 2, vadd_f32(vld1_dup_f32(c0n + 2), vget_high_f32(vc0n)), 0);
			case 2:
				vst1_f32(c0n, vadd_f32(vld1_f32(c0n), vget_low_f32(vc0n)));
				break;
			case 1:
				vst1_lane_f32(c0n, vadd_f32(vld1_dup_f32(c0n), vget_low_f32(vc0n)), 0);
				break;
			default:
				NNP_UNREACHABLE;
		}
		if (mr > 1) {
			c += row_stride_c;
			float32x4_t vc1n = vc10;
			uint32_t nr1 = nr;
			float* c1n = c;
			if (nr1 > 4) {
				vst1q_f32(c1n, vaddq_f32(vld1q_f32(c1n), vc1n));
				c1n += 4;
				nr1 -= 4;
				vc1n = vc11;
				if (nr1 > 4) {
					vst1q_f32(c1n, vaddq_f32(vld1q_f32(c1n), vc1n));
					c1n += 4;
					nr1 -= 4;
					vc1n = vc12;
				}
			}
			switch (nr1) {
				case 4:
					vst1q_f32(c1n, vaddq_f32(vld1q_f32(c1n), vc1n));
					break;
				case 3:
					vst1_lane_f32(c1n + 2, vadd_f32(vld1_dup_f32(c1n + 2), vget_high_f32(vc1n)), 0);
				case 2:
					vst1_f32(c1n, vadd_f32(vld1_f32(c1n), vget_low_f32(vc1n)));
					break;
				case 1:
					vst1_lane_f32(c1n, vadd_f32(vld1_dup_f32(c1n), vget_low_f32(vc1n)), 0);
					break;
				default:
					NNP_UNREACHABLE;
			}
			if (mr > 2) {
				c += row_stride_c;
				float32x4_t vc2n = vc20;
				uint32_t nr2 = nr;
				float* c2n = c;
				if (nr2 > 4) {
					vst1q_f32(c2n, vaddq_f32(vld1q_f32(c2n), vc2n));
					c2n += 4;
					nr2 -= 4;
					vc2n = vc21;
					if (nr2 > 4) {
						vst1q_f32(c2n, vaddq_f32(vld1q_f32(c2n), vc2n));
						c2n += 4;
						nr2 -= 4;
						vc2n = vc22;
					}
				}
				switch (nr2) {
					case 4:
						vst1q_f32(c2n, vaddq_f32(vld1q_f32(c2n), vc2n));
						break;
					case 3:
						vst1_lane_f32(c2n + 2, vadd_f32(vld1_dup_f32(c2n + 2), vget_high_f32(vc2n)), 0);
					case 2:
						vst1_f32(c2n, vadd_f32(vld1_f32(c2n), vget_low_f32(vc2n)));
						break;
					case 1:
						vst1_lane_f32(c2n, vadd_f32(vld1_dup_f32(c2n), vget_low_f32(vc2n)), 0);
						break;
					default:
						NNP_UNREACHABLE;
				}
				if (mr > 3) {
					c += row_stride_c;
					float32x4_t vc3n = vc30;
					uint32_t nr3 = nr;
					float* c3n = c;
					if (nr3 > 4) {
						vst1q_f32(c3n, vaddq_f32(vld1q_f32(c3n), vc3n));
						c3n += 4;
						nr3 -= 4;
						vc3n = vc31;
						if (nr3 > 4) {
							vst1q_f32(c3n, vaddq_f32(vld1q_f32(c3n), vc3n));
							c3n += 4;
							nr3 -= 4;
							vc3n = vc32;
						}
					}
					switch (nr3) {
						case 4:
							vst1q_f32(c3n, vaddq_f32(vld1q_f32(c3n), vc3n));
							break;
						case 3:
							vst1_lane_f32(c3n + 2, vadd_f32(vld1_dup_f32(c3n + 2), vget_high_f32(vc3n)), 0);
						case 2:
							vst1_f32(c3n, vadd_f32(vld1_f32(c3n), vget_low_f32(vc3n)));
							break;
						case 1:
							vst1_lane_f32(c3n, vadd_f32(vld1_dup_f32(c3n), vget_low_f32(vc3n)), 0);
							break;
						default:
							NNP_UNREACHABLE;
					}
				}
			}
		}
	} else {
		float32x4_t vc0n = vc00;
		uint32_t nr0 = nr;
		float* c0n = c;
		if (nr0 > 4) {
			vst1q_f32(c0n, vc0n);
			c0n += 4;
			nr0 -= 4;
			vc0n = vc01;
			if (nr0 > 4) {
				vst1q_f32(c0n, vc0n);
				c0n += 4;
				nr0 -= 4;
				vc0n = vc02;
			}
		}
		switch (nr0) {
			case 4:
				vst1q_f32(c0n, vc0n);
				break;
			case 3:
				vst1_lane_f32(c0n + 2, vget_high_f32(vc0n), 0);
			case 2:
				vst1_f32(c0n, vget_low_f32(vc0n));
				break;
			case 1:
				vst1_lane_f32(c0n, vget_low_f32(vc0n), 0);
				break;
			default:
				NNP_UNREACHABLE;
		}
		if (mr > 1) {
			c += row_stride_c;
			float32x4_t vc1n = vc10;
			uint32_t nr1 = nr;
			float* c1n = c;
			if (nr1 > 4) {
				vst1q_f32(c1n, vc1n);
				c1n += 4;
				nr1 -= 4;
				vc1n = vc11;
				if (nr1 > 4) {
					vst1q_f32(c1n, vc1n);
					c1n += 4;
					nr1 -= 4;
					vc1n = vc12;
				}
			}
			switch (nr1) {
				case 4:
					vst1q_f32(c1n, vc1n);
					break;
				case 3:
					vst1_lane_f32(c1n + 2, vget_high_f32(vc1n), 0);
				case 2:
					vst1_f32(c1n, vget_low_f32(vc1n));
					break;
				case 1:
					vst1_lane_f32(c1n, vget_low_f32(vc1n), 0);
					break;
				default:
					NNP_UNREACHABLE;
			}
			if (mr > 2) {
				c += row_stride_c;
				float32x4_t vc2n = vc20;
				uint32_t nr2 = nr;
				float* c2n = c;
				if (nr2 > 4) {
					vst1q_f32(c2n, vc2n);
					c2n += 4;
					nr2 -= 4;
					vc2n = vc21;
					if (nr2 > 4) {
						vst1q_f32(c2n, vc2n);
						c2n += 4;
						nr2 -= 4;
						vc2n = vc22;
					}
				}
				switch (nr2) {
					case 4:
						vst1q_f32(c2n, vc2n);
						break;
					case 3:
						vst1_lane_f32(c2n + 2, vget_high_f32(vc2n), 0);
					case 2:
						vst1_f32(c2n, vget_low_f32(vc2n));
						break;
					case 1:
						vst1_lane_f32(c2n, vget_low_f32(vc2n), 0);
						break;
					default:
						NNP_UNREACHABLE;
				}
				if (mr > 3) {
					c += row_stride_c;
					float32x4_t vc3n = vc30;
					uint32_t nr3 = nr;
					float* c3n = c;
					if (nr3 > 4) {
						vst1q_f32(c3n, vc3n);
						c3n += 4;
						nr3 -= 4;
						vc3n = vc31;
						if (nr3 > 4) {
							vst1q_f32(c3n, vc3n);
							c3n += 4;
							nr3 -= 4;
							vc3n = vc32;
						}
					}
					switch (nr3) {
						case 4:
							vst1q_f32(c3n, vc3n);
							break;
						case 3:
							vst1_lane_f32(c3n + 2, vget_high_f32(vc3n), 0);
						case 2:
							vst1_f32(c3n, vget_low_f32(vc3n));
							break;
						case 1:
							vst1_lane_f32(c3n, vget_low_f32(vc3n), 0);
							break;
						default:
							NNP_UNREACHABLE;
					}
				}
			}
		}
	}
}
