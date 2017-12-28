#include <stddef.h>
#include <stdint.h>

#include <nnpack/macros.h>
#include <nnpack/arm_neon.h>


void nnp_sgemm_only_6x8__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
	float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
	float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
	float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
	float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
	float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
	do {
		const float32x4_t va0123 = vld1q_f32(a);
		const float32x2_t va45   = vld1_f32(a + 4);
		a += 6;

		const float32x4_t vb0 = vld1q_f32_aligned(b + 0);
		const float32x4_t vb1 = vld1q_f32_aligned(b + 4);
		b += 8;

		#if defined(__aarch64__)
			vc00 = vfmaq_lane_f32(vc00, vb0, vget_low_f32(va0123), 0);
			vc10 = vfmaq_lane_f32(vc10, vb0, vget_low_f32(va0123), 1);
			vc20 = vfmaq_lane_f32(vc20, vb0, vget_high_f32(va0123), 0);
			vc30 = vfmaq_lane_f32(vc30, vb0, vget_high_f32(va0123), 1);
			vc40 = vfmaq_lane_f32(vc40, vb0, va45, 0);
			vc50 = vfmaq_lane_f32(vc50, vb0, va45, 1);

			vc01 = vfmaq_lane_f32(vc01, vb1, vget_low_f32(va0123), 0);
			vc11 = vfmaq_lane_f32(vc11, vb1, vget_low_f32(va0123), 1);
			vc21 = vfmaq_lane_f32(vc21, vb1, vget_high_f32(va0123), 0);
			vc31 = vfmaq_lane_f32(vc31, vb1, vget_high_f32(va0123), 1);
			vc41 = vfmaq_lane_f32(vc41, vb1, va45, 0);
			vc51 = vfmaq_lane_f32(vc51, vb1, va45, 1);
		#else
			vc00 = vmlaq_lane_f32(vc00, vb0, vget_low_f32(va0123), 0);
			vc10 = vmlaq_lane_f32(vc10, vb0, vget_low_f32(va0123), 1);
			vc20 = vmlaq_lane_f32(vc20, vb0, vget_high_f32(va0123), 0);
			vc30 = vmlaq_lane_f32(vc30, vb0, vget_high_f32(va0123), 1);
			vc40 = vmlaq_lane_f32(vc40, vb0, va45, 0);
			vc50 = vmlaq_lane_f32(vc50, vb0, va45, 1);

			vc01 = vmlaq_lane_f32(vc01, vb1, vget_low_f32(va0123), 0);
			vc11 = vmlaq_lane_f32(vc11, vb1, vget_low_f32(va0123), 1);
			vc21 = vmlaq_lane_f32(vc21, vb1, vget_high_f32(va0123), 0);
			vc31 = vmlaq_lane_f32(vc31, vb1, vget_high_f32(va0123), 1);
			vc41 = vmlaq_lane_f32(vc41, vb1, va45, 0);
			vc51 = vmlaq_lane_f32(vc51, vb1, va45, 1);
		#endif
	} while (--k);

	if (update) {
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc40));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc41));
		c += row_stride_c;
		vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc50));
		vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc51));
	} else {
		vst1q_f32(c + 0, vc00);
		vst1q_f32(c + 4, vc01);
		c += row_stride_c;
		vst1q_f32(c + 0, vc10);
		vst1q_f32(c + 4, vc11);
		c += row_stride_c;
		vst1q_f32(c + 0, vc20);
		vst1q_f32(c + 4, vc21);
		c += row_stride_c;
		vst1q_f32(c + 0, vc30);
		vst1q_f32(c + 4, vc31);
		c += row_stride_c;
		vst1q_f32(c + 0, vc40);
		vst1q_f32(c + 4, vc41);
		c += row_stride_c;
		vst1q_f32(c + 0, vc50);
		vst1q_f32(c + 4, vc51);
	}
}

void nnp_sgemm_upto_6x8__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c) {
	float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
	float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
	float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
	float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
	float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
	float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
	do {
		float32x4_t vb0, vb1;
		
		vb0 = vld1q_f32_aligned(b);
		b += 4;
		if (nr > 4) {
			vb1 = vld1q_f32_aligned(b);
			b += 4;
		}

		const float32x4_t va0 = vld1q_dup_f32(a);
		a += 1;
		vc00 = vmuladdq_f32(vc00, va0, vb0);
		vc01 = vmuladdq_f32(vc01, va0, vb1);

		if (mr > 1) {
			const float32x4_t va1 = vld1q_dup_f32(a);
			a += 1;
			vc10 = vmuladdq_f32(vc10, va1, vb0);
			vc11 = vmuladdq_f32(vc11, va1, vb1);

			if (mr > 2) {
				const float32x4_t va2 = vld1q_dup_f32(a);
				a += 1;
				vc20 = vmuladdq_f32(vc20, va2, vb0);
				vc21 = vmuladdq_f32(vc21, va2, vb1);

				if (mr > 3) {
					const float32x4_t va3 = vld1q_dup_f32(a);
					a += 1;
					vc30 = vmuladdq_f32(vc30, va3, vb0);
					vc31 = vmuladdq_f32(vc31, va3, vb1);

					if (mr > 4) {
						const float32x4_t va4 = vld1q_dup_f32(a);
						a += 1;
						vc40 = vmuladdq_f32(vc40, va4, vb0);
						vc41 = vmuladdq_f32(vc41, va4, vb1);

						if (mr > 5) {
							const float32x4_t va5 = vld1q_dup_f32(a);
							a += 1;
							vc50 = vmuladdq_f32(vc50, va5, vb0);
							vc51 = vmuladdq_f32(vc51, va5, vb1);
						}
					}
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
					if (mr > 4) {
						c += row_stride_c;
						float32x4_t vc4n = vc40;
						uint32_t nr4 = nr;
						float* c4n = c;
						if (nr4 > 4) {
							vst1q_f32(c4n, vaddq_f32(vld1q_f32(c4n), vc4n));
							c4n += 4;
							nr4 -= 4;
							vc4n = vc41;
						}
						switch (nr4) {
							case 4:
								vst1q_f32(c4n, vaddq_f32(vld1q_f32(c4n), vc4n));
								break;
							case 3:
								vst1_lane_f32(c4n + 2, vadd_f32(vld1_dup_f32(c4n + 2), vget_high_f32(vc4n)), 0);
							case 2:
								vst1_f32(c4n, vadd_f32(vld1_f32(c4n), vget_low_f32(vc4n)));
								break;
							case 1:
								vst1_lane_f32(c4n, vadd_f32(vld1_dup_f32(c4n), vget_low_f32(vc4n)), 0);
								break;
							default:
								NNP_UNREACHABLE;
						}
						if (mr > 5) {
							c += row_stride_c;
							float32x4_t vc5n = vc50;
							uint32_t nr5 = nr;
							float* c5n = c;
							if (nr5 > 4) {
								vst1q_f32(c5n, vaddq_f32(vld1q_f32(c5n), vc5n));
								c5n += 4;
								nr5 -= 4;
								vc5n = vc51;
							}
							switch (nr5) {
								case 4:
									vst1q_f32(c5n, vaddq_f32(vld1q_f32(c5n), vc5n));
									break;
								case 3:
									vst1_lane_f32(c5n + 2, vadd_f32(vld1_dup_f32(c5n + 2), vget_high_f32(vc5n)), 0);
								case 2:
									vst1_f32(c5n, vadd_f32(vld1_f32(c5n), vget_low_f32(vc5n)));
									break;
								case 1:
									vst1_lane_f32(c5n, vadd_f32(vld1_dup_f32(c5n), vget_low_f32(vc5n)), 0);
									break;
								default:
									NNP_UNREACHABLE;
							}
						}
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
					if (mr > 4) {
						c += row_stride_c;
						float32x4_t vc4n = vc40;
						uint32_t nr4 = nr;
						float* c4n = c;
						if (nr4 > 4) {
							vst1q_f32(c4n, vc4n);
							c4n += 4;
							nr4 -= 4;
							vc4n = vc41;
						}
						switch (nr4) {
							case 4:
								vst1q_f32(c4n, vc4n);
								break;
							case 3:
								vst1_lane_f32(c4n + 2, vget_high_f32(vc4n), 0);
							case 2:
								vst1_f32(c4n, vget_low_f32(vc4n));
								break;
							case 1:
								vst1_lane_f32(c4n, vget_low_f32(vc4n), 0);
								break;
							default:
								NNP_UNREACHABLE;
						}
						if (mr > 5) {
							c += row_stride_c;
							float32x4_t vc5n = vc50;
							uint32_t nr5 = nr;
							float* c5n = c;
							if (nr5 > 4) {
								vst1q_f32(c5n, vc5n);
								c5n += 4;
								nr5 -= 4;
								vc5n = vc51;
							}
							switch (nr5) {
								case 4:
									vst1q_f32(c5n, vc5n);
									break;
								case 3:
									vst1_lane_f32(c5n + 2, vget_high_f32(vc5n), 0);
								case 2:
									vst1_f32(c5n, vget_low_f32(vc5n));
									break;
								case 1:
									vst1_lane_f32(c5n, vget_low_f32(vc5n), 0);
									break;
								default:
									NNP_UNREACHABLE;
							}
						}
					}
				}
			}
		}
	}
}
