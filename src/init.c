#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include <pthread.h>

#include <cpuinfo.h>

#include <nnpack.h>
#include <nnpack/hwinfo.h>
#include <nnpack/blas.h>
#include <nnpack/transform.h>
#include <nnpack/relu.h>
#include <nnpack/softmax.h>

struct hardware_info nnp_hwinfo = { };
static pthread_once_t hwinfo_init_control = PTHREAD_ONCE_INIT;


#if (CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) && !defined(__ANDROID__)
	static void init_x86_hwinfo(void) {
		const struct cpuinfo_cache* l1d = cpuinfo_get_l1d_cache(0);
		if (l1d != NULL) {
			nnp_hwinfo.cache.l1 = (struct cache_info) {
				.size          = l1d->size,
				.associativity = l1d->associativity,
				.threads       = l1d->processor_count,
			};
			const struct cpuinfo_cache* l2 = cpuinfo_get_l2_cache(0);
			if (l2 != NULL) {
				nnp_hwinfo.cache.l2 = (struct cache_info) {
					.size          = l2->size,
					.associativity = l2->associativity,
					.threads       = l2->processor_count,
					.inclusive     = !!(l2->flags & CPUINFO_CACHE_INCLUSIVE),
				};
				const struct cpuinfo_cache* l3 = cpuinfo_get_l3_cache(0);
				if (l3 != NULL) {
					nnp_hwinfo.cache.l3 = (struct cache_info) {
						.size          = l3->size,
						.associativity = l3->associativity,
						.threads       = l3->processor_count,
						.inclusive     = !!(l3->flags & CPUINFO_CACHE_INCLUSIVE),
					};
					const struct cpuinfo_cache* l4 = cpuinfo_get_l4_cache(0);
					if (l4 != NULL) {
						nnp_hwinfo.cache.l4 = (struct cache_info) {
							.size          = l4->size,
							.associativity = l4->associativity,
							.threads       = l4->processor_count,
							.inclusive     = !!(l4->flags & CPUINFO_CACHE_INCLUSIVE),
						};
					}
				}
			}
		}
	}
#endif

#if !(CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) || defined(__ANDROID__)
	static void init_static_hwinfo(void) {
		nnp_hwinfo.cache.l1 = (struct cache_info) {
			.size = 16 * 1024,
			.associativity = 4,
			.threads = 1,
			.inclusive = true,
		};
		nnp_hwinfo.cache.l2 = (struct cache_info) {
			.size = 128 * 1024,
			.associativity = 4,
			.threads = 1,
			.inclusive = true,
		};
		nnp_hwinfo.cache.l3 = (struct cache_info) {
			.size = 2 * 1024 * 1024,
			.associativity = 8,
			.threads = 1,
			.inclusive = true,
		};
	}
#endif

#if !CPUINFO_ARCH_X86 && !CPUINFO_ARCH_X86_64 && defined(__APPLE__)
	static void init_static_ios_hwinfo(void) {
		nnp_hwinfo.cache.l1 = (struct cache_info) {
			.size = 32 * 1024,
			.associativity = 1,
			.threads = 1,
			.inclusive = false,
		};
		nnp_hwinfo.cache.l2 = (struct cache_info) {
			.size = 1 * 1024 * 1024,
			.associativity = 1,
			.threads = 1,
			.inclusive = false,
		};
		nnp_hwinfo.cache.l3 = (struct cache_info) {
			.size = 2 * 1024 * 1024,
			.associativity = 8,
			.threads = 1,
			.inclusive = false,
		};
	}
#endif

#if !NNP_CONVOLUTION_ONLY
	#if NNP_BACKEND_X86_64
		static const nnp_sdotxf_function sdotxf[8] = {
			[0] = nnp_sdotxf1__avx2,
			[1] = nnp_sdotxf2__avx2,
			[2] = nnp_sdotxf3__avx2,
			[3] = nnp_sdotxf4__avx2,
			[4] = nnp_sdotxf5__avx2,
			[5] = nnp_sdotxf6__avx2,
			[6] = nnp_sdotxf7__avx2,
			[7] = nnp_sdotxf8__avx2,
		};

		static const nnp_shdotxf_function shdotxf[8] = {
			[0] = nnp_shdotxf1__avx2,
			[1] = nnp_shdotxf2__avx2,
			[2] = nnp_shdotxf3__avx2,
			[3] = nnp_shdotxf4__avx2,
			[4] = nnp_shdotxf5__avx2,
			[5] = nnp_shdotxf6__avx2,
			[6] = nnp_shdotxf7__avx2,
			[7] = nnp_shdotxf8__avx2,
		};
	#elif NNP_BACKEND_ARM
		static const nnp_sdotxf_function sdotxf[8] = {
			[0] = nnp_sdotxf1__neon,
			[1] = nnp_sdotxf2__neon,
			[2] = nnp_sdotxf3__neon,
			[3] = nnp_sdotxf4__neon,
			[4] = nnp_sdotxf5__neon,
			[5] = nnp_sdotxf6__neon,
			[6] = nnp_sdotxf7__neon,
			[7] = nnp_sdotxf8__neon,
		};

		static const nnp_shdotxf_function shdotxf[8] = {
			[0] = nnp_shdotxf1__psimd,
			[1] = nnp_shdotxf2__psimd,
			[2] = nnp_shdotxf3__psimd,
			[3] = nnp_shdotxf4__psimd,
			[4] = nnp_shdotxf5__psimd,
			[5] = nnp_shdotxf6__psimd,
			[6] = nnp_shdotxf7__psimd,
			[7] = nnp_shdotxf8__psimd,
		};
	#elif NNP_BACKEND_PSIMD
		static const nnp_sdotxf_function sdotxf[8] = {
			[0] = nnp_sdotxf1__psimd,
			[1] = nnp_sdotxf2__psimd,
			[2] = nnp_sdotxf3__psimd,
			[3] = nnp_sdotxf4__psimd,
			[4] = nnp_sdotxf5__psimd,
			[5] = nnp_sdotxf6__psimd,
			[6] = nnp_sdotxf7__psimd,
			[7] = nnp_sdotxf8__psimd,
		};

		static const nnp_shdotxf_function shdotxf[8] = {
			[0] = nnp_shdotxf1__psimd,
			[1] = nnp_shdotxf2__psimd,
			[2] = nnp_shdotxf3__psimd,
			[3] = nnp_shdotxf4__psimd,
			[4] = nnp_shdotxf5__psimd,
			[5] = nnp_shdotxf6__psimd,
			[6] = nnp_shdotxf7__psimd,
			[7] = nnp_shdotxf8__psimd,
		};
	#elif NNP_BACKEND_SCALAR
		static const nnp_sdotxf_function sdotxf[8] = {
			[0] = nnp_sdotxf1__scalar,
			[1] = nnp_sdotxf2__scalar,
			[2] = nnp_sdotxf3__scalar,
			[3] = nnp_sdotxf4__scalar,
			[4] = nnp_sdotxf5__scalar,
			[5] = nnp_sdotxf6__scalar,
			[6] = nnp_sdotxf7__scalar,
			[7] = nnp_sdotxf8__scalar,
		};

		static const nnp_shdotxf_function shdotxf[8] = {
			[0] = nnp_shdotxf1__scalar,
			[1] = nnp_shdotxf2__scalar,
			[2] = nnp_shdotxf3__scalar,
			[3] = nnp_shdotxf4__scalar,
			[4] = nnp_shdotxf5__scalar,
			[5] = nnp_shdotxf6__scalar,
			[6] = nnp_shdotxf7__scalar,
			[7] = nnp_shdotxf8__scalar,
		};
	#endif
#endif /* !NNP_CONVOLUTION_ONLY */

static void init_hwinfo(void) {
	#if (CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) && !defined(__ANDROID__)
		init_x86_hwinfo();
	#elif !CPUINFO_ARCH_X86 && !CPUINFO_ARCH_X86_64 && defined(__APPLE__)
		init_static_ios_hwinfo();
	#else
		init_static_hwinfo();
	#endif

	/* Compute high-level cache blocking parameters */
	nnp_hwinfo.blocking.l1 = nnp_hwinfo.cache.l1.size;
	if (nnp_hwinfo.cache.l1.threads > 1) {
		nnp_hwinfo.blocking.l1 /= nnp_hwinfo.cache.l1.threads;
	}
	if (nnp_hwinfo.cache.l2.size != 0) {
		nnp_hwinfo.blocking.l2 = nnp_hwinfo.cache.l2.size;
		if (nnp_hwinfo.cache.l2.inclusive) {
			nnp_hwinfo.blocking.l2 -= nnp_hwinfo.cache.l1.size;
		}
		if (nnp_hwinfo.cache.l2.threads > 1) {
			nnp_hwinfo.blocking.l2 /= nnp_hwinfo.cache.l2.threads;
		}
	}
	if (nnp_hwinfo.cache.l3.size != 0) {
		nnp_hwinfo.blocking.l3 = nnp_hwinfo.cache.l3.size;
		if (nnp_hwinfo.cache.l3.inclusive) {
			nnp_hwinfo.blocking.l3 -= nnp_hwinfo.cache.l2.size;
		}
	}
	nnp_hwinfo.blocking.l4 = nnp_hwinfo.cache.l4.size;
	if (nnp_hwinfo.cache.l1.size && nnp_hwinfo.cache.l2.size && nnp_hwinfo.cache.l3.size) {
		#if NNP_BACKEND_X86_64
			if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
				nnp_hwinfo.simd_width = 8;
				nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset_and_store__avx2;
				nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset_and_stream__avx2;
#if !NNP_INFERENCE_ONLY
				nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset) nnp_ifft8x8_with_offset__avx2;
#endif /* !NNP_INFERENCE_ONLY */
				nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias__avx2;
				nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias_with_relu__avx2;
				nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset_and_store__avx2;
				nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset_and_stream__avx2;
#if !NNP_INFERENCE_ONLY
				nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset) nnp_ifft16x16_with_offset__avx2;
#endif /* !NNP_INFERENCE_ONLY */
				nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias__avx2;
				nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias_with_relu__avx2;
				nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset_and_store__avx2;
				nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset_and_stream__avx2;
				nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_kwt8x8_3x3_and_stream__avx2;
#if !NNP_INFERENCE_ONLY
				nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset) nnp_kwt8x8_3Rx3R_and_stream__avx2;
				nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_owt8x8_3x3__avx2;
#endif /* !NNP_INFERENCE_ONLY */
				nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias__avx2;
				nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias_with_relu__avx2;
#if !NNP_CONVOLUTION_ONLY
				nnp_hwinfo.activations.relu = nnp_relu__avx2;
				nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__avx2;
				nnp_hwinfo.activations.grad_relu = nnp_grad_relu__avx2;
				nnp_hwinfo.activations.softmax = nnp_softmax__avx2;
				nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__avx2;
				nnp_hwinfo.sdotxf = (struct sdotxf) {
					.functions = sdotxf,
					.fusion = NNP_COUNT_OF(sdotxf),
				};
				nnp_hwinfo.shdotxf = (struct shdotxf) {
					.functions = shdotxf,
					.fusion = NNP_COUNT_OF(shdotxf),
				};
#endif /* !NNP_CONVOLUTION_ONLY */
				nnp_hwinfo.conv1x1 = (struct convolution) {
					.mr = 2,
					.nr = 4,
					.only_mr_x_nr = nnp_conv1x1_only_2x4__fma3,
					.upto_mr_x_nr = nnp_conv1x1_upto_2x4__fma3,
				};
				nnp_hwinfo.sgemm = (struct sgemm) {
					.mr = 4,
					.nr = 24,
					.only_mr_x_nr = nnp_sgemm_only_4x24__fma3,
					.upto_mr_x_nr = nnp_sgemm_upto_4x24__fma3,
				};
				nnp_hwinfo.sxgemm = (struct sxgemm) {
					.mr = 3,
					.nr = 4,
					.only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s8gemm_only_3x4__fma3,
					.upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s8gemm_upto_3x4__fma3,
				};
				nnp_hwinfo.cxgemm = (struct cxgemm) {
					.mr = 2,
					.nr = 2,
#if !NNP_INFERENCE_ONLY
					.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c6gemm_only_2x2__fma3,
					.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c6gemm_upto_2x2__fma3,
					.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c8gemm_only_2x2__fma3,
					.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c8gemm_upto_2x2__fma3,
#endif /* !NNP_INFERENCE_ONLY */
					.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c6gemm_conjb_only_2x2__fma3,
					.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c6gemm_conjb_upto_2x2__fma3,
					.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c8gemm_conjb_only_2x2__fma3,
					.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c8gemm_conjb_upto_2x2__fma3,
#if !NNP_INFERENCE_ONLY
					.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c6gemm_conjb_transc_only_2x2__fma3,
					.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c6gemm_conjb_transc_upto_2x2__fma3,
					.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c8gemm_conjb_transc_only_2x2__fma3,
					.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c8gemm_conjb_transc_upto_2x2__fma3,
#endif /* !NNP_INFERENCE_ONLY */
				};
				nnp_hwinfo.supported = true;
			}
		#elif NNP_BACKEND_PSIMD
			nnp_hwinfo.simd_width = 4;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset__psimd;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset) nnp_ifft8x8_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias__psimd;
			nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias_with_relu__psimd;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset__psimd;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset) nnp_ifft16x16_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias__psimd;
			nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias_with_relu__psimd;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__psimd;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__psimd;
			nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_kwt8x8_3x3__psimd;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset) nnp_kwt8x8_3Rx3R__psimd;
			nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_owt8x8_3x3__psimd;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias__psimd;
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias_with_relu__psimd;
#if !NNP_CONVOLUTION_ONLY
			nnp_hwinfo.activations.relu = nnp_relu__psimd;
			nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__psimd;
			nnp_hwinfo.activations.grad_relu = nnp_grad_relu__psimd;
			nnp_hwinfo.activations.softmax = nnp_softmax__psimd;
			nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__psimd;
			nnp_hwinfo.sdotxf = (struct sdotxf) {
				.functions = sdotxf,
				.fusion = NNP_COUNT_OF(sdotxf),
			};
			nnp_hwinfo.shdotxf = (struct shdotxf) {
				.functions = shdotxf,
				.fusion = NNP_COUNT_OF(shdotxf),
			};
#endif /* !NNP_CONVOLUTION_ONLY */
			nnp_hwinfo.conv1x1 = (struct convolution) {
				.mr = 2,
				.nr = 4,
				.only_mr_x_nr = nnp_conv1x1_only_2x4__psimd,
				.upto_mr_x_nr = nnp_conv1x1_upto_2x4__psimd,
			};
			nnp_hwinfo.sgemm = (struct sgemm) {
				.mr = 4,
				.nr = 8,
				.only_mr_x_nr = nnp_sgemm_only_4x8__psimd,
				.upto_mr_x_nr = nnp_sgemm_upto_4x8__psimd,
			};
			nnp_hwinfo.sxgemm = (struct sxgemm) {
				.mr = 3,
				.nr = 4,
				.only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4gemm_only_3x4__psimd,
				.upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4gemm_upto_3x4__psimd,
			};
			nnp_hwinfo.cxgemm = (struct cxgemm) {
				.mr = 2,
				.nr = 2,
#if !NNP_INFERENCE_ONLY
				.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c2gemm_only_2x2__psimd,
				.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c2gemm_upto_2x2__psimd,
				.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c4gemm_only_2x2__psimd,
				.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c4gemm_upto_2x2__psimd,
#endif /* !NNP_INFERENCE_ONLY */
				.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c2gemm_conjb_only_2x2__psimd,
				.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c2gemm_conjb_upto_2x2__psimd,
				.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c4gemm_conjb_only_2x2__psimd,
				.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c4gemm_conjb_upto_2x2__psimd,
#if !NNP_INFERENCE_ONLY
				.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c2gemm_conjb_transc_only_2x2__psimd,
				.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c2gemm_conjb_transc_upto_2x2__psimd,
				.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c4gemm_conjb_transc_only_2x2__psimd,
				.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c4gemm_conjb_transc_upto_2x2__psimd,
#endif /* !NNP_INFERENCE_ONLY */
			};
			nnp_hwinfo.supported = true;
		#elif NNP_BACKEND_ARM
			nnp_hwinfo.simd_width = 4;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset__psimd;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset) nnp_ifft8x8_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias__psimd;
			nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias_with_relu__psimd;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset__psimd;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset) nnp_ifft16x16_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias__psimd;
			nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias_with_relu__psimd;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__neon;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__neon;
			nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_kwt8x8_3x3__neon;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset) nnp_kwt8x8_3Rx3R__neon;
			nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_owt8x8_3x3__neon;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias__neon;
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias_with_relu__neon;
			if (cpuinfo_has_arm_neon_fp16()) {
				nnp_hwinfo.transforms.iwt_f6x6_3x3_fp16_with_offset = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_fp16_with_offset__neonhp;
				nnp_hwinfo.transforms.kwt_f6x6_3x3_fp16 = (nnp_transform_2d_with_offset) nnp_kwt8x8_3x3_fp16__neonhp;
				nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_fp16_with_bias__neonhp;
				nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_fp16_with_bias_with_relu__neonhp;
			}
#if !NNP_CONVOLUTION_ONLY
			nnp_hwinfo.activations.relu = nnp_relu__neon;
			nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__neon;
			nnp_hwinfo.activations.grad_relu = nnp_grad_relu__neon;
			nnp_hwinfo.activations.softmax = nnp_softmax__psimd;
			nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__psimd;
			nnp_hwinfo.sdotxf = (struct sdotxf) {
				.functions = sdotxf,
				.fusion = NNP_COUNT_OF(sdotxf),
			};
			nnp_hwinfo.shdotxf = (struct shdotxf) {
				.functions = shdotxf,
				.fusion = NNP_COUNT_OF(shdotxf),
			};
#endif /* !NNP_CONVOLUTION_ONLY */
			nnp_hwinfo.conv1x1 = (struct convolution) {
				.mr = 2,
				.nr = 4,
				.only_mr_x_nr = nnp_conv1x1_only_2x4__neon,
				.upto_mr_x_nr = nnp_conv1x1_upto_2x4__neon,
			};
			nnp_hwinfo.sgemm = (struct sgemm) {
				.mr = 6,
				.nr = 8,
				#if CPUINFO_ARCH_ARM
					.only_mr_x_nr = nnp_sgemm_only_6x8__aarch32_neon,
				#else
					.only_mr_x_nr = nnp_sgemm_only_6x8__neon,
				#endif
				.upto_mr_x_nr = nnp_sgemm_upto_6x8__neon,
			};
			nnp_hwinfo.sxgemm = (struct sxgemm) {
				.mr = 3,
				.nr = 4,
				.only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4gemm_only_3x4__neon,
				.upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4gemm_upto_3x4__neon,
			};
			if (cpuinfo_has_arm_neon_fp16()) {
				nnp_hwinfo.hxgemm = (struct hxgemm) {
					.mr = 3,
					.nr = 4,
					.only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_h4gemm_only_3x4__neonhp,
					.upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_h4gemm_upto_3x4__neonhp,
				};
			}
			nnp_hwinfo.cxgemm = (struct cxgemm) {
				.mr = 2,
				.nr = 2,
#if !NNP_INFERENCE_ONLY
				.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c2gemm_only_2x2__neon,
				.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c2gemm_upto_2x2__neon,
				.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c4gemm_only_2x2__neon,
				.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c4gemm_upto_2x2__neon,
#endif /* !NNP_INFERENCE_ONLY */
				.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c2gemm_conjb_only_2x2__neon,
				.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c2gemm_conjb_upto_2x2__neon,
				.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c4gemm_conjb_only_2x2__neon,
				.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c4gemm_conjb_upto_2x2__neon,
#if !NNP_INFERENCE_ONLY
				.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s4c2gemm_conjb_transc_only_2x2__neon,
				.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s4c2gemm_conjb_transc_upto_2x2__neon,
				.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_c4gemm_conjb_transc_only_2x2__neon,
				.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_c4gemm_conjb_transc_upto_2x2__neon,
#endif /* !NNP_INFERENCE_ONLY */
			};
			nnp_hwinfo.supported = cpuinfo_has_arm_neon();
		#elif NNP_BACKEND_SCALAR
			nnp_hwinfo.simd_width = 1;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset__scalar;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft8x8_with_offset__scalar;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset) nnp_ifft8x8_with_offset__scalar;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias__scalar;
			nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft8x8_with_bias_with_relu__scalar;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset__scalar;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_fft16x16_with_offset__scalar;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset) nnp_ifft16x16_with_offset__scalar;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias__scalar;
			nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_ifft16x16_with_bias_with_relu__scalar;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__scalar;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__scalar;
			nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_kwt8x8_3x3__scalar;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset) nnp_kwt8x8_3Rx3R__scalar;
			nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset) nnp_owt8x8_3x3__scalar;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias__scalar;
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias) nnp_owt8x8_3x3_with_bias_with_relu__scalar;
#if !NNP_CONVOLUTION_ONLY
			nnp_hwinfo.activations.relu = nnp_relu__scalar;
			nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__scalar;
			nnp_hwinfo.activations.grad_relu = nnp_grad_relu__scalar;
			nnp_hwinfo.activations.softmax = nnp_softmax__scalar;
			nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__scalar;
			nnp_hwinfo.sdotxf = (struct sdotxf) {
				.functions = sdotxf,
				.fusion = NNP_COUNT_OF(sdotxf),
			};
			nnp_hwinfo.shdotxf = (struct shdotxf) {
				.functions = shdotxf,
				.fusion = NNP_COUNT_OF(shdotxf),
			};
#endif /* !NNP_CONVOLUTION_ONLY */
			nnp_hwinfo.conv1x1 = (struct convolution) {
				.mr = 2,
				.nr = 4,
				.only_mr_x_nr = nnp_conv1x1_only_2x4__scalar,
				.upto_mr_x_nr = nnp_conv1x1_upto_2x4__scalar,
			};
			nnp_hwinfo.sgemm = (struct sgemm) {
				.mr = 4,
				.nr = 3,
				.only_mr_x_nr = nnp_sgemm_only_4x3__scalar,
				.upto_mr_x_nr = nnp_sgemm_upto_4x3__scalar,
			};
			nnp_hwinfo.sxgemm = (struct sxgemm) {
				.mr = 4,
				.nr = 3,
				.only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_sgemm_only_4x3__scalar,
				.upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_sgemm_upto_4x3__scalar,
			};
			nnp_hwinfo.cxgemm = (struct cxgemm) {
				.mr = 2,
				.nr = 2,
#if !NNP_INFERENCE_ONLY
				.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s2gemm_only_2x2__scalar,
				.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s2gemm_upto_2x2__scalar,
				.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_cgemm_only_2x2__scalar,
				.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_cgemm_upto_2x2__scalar,
#endif /* !NNP_INFERENCE_ONLY */
				.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s2gemm_only_2x2__scalar,
				.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s2gemm_upto_2x2__scalar,
				.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_cgemm_conjb_only_2x2__scalar,
				.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_cgemm_conjb_upto_2x2__scalar,
#if !NNP_INFERENCE_ONLY
				.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_s2gemm_transc_only_2x2__scalar,
				.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_s2gemm_transc_upto_2x2__scalar,
				.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function) nnp_cgemm_conjb_transc_only_2x2__scalar,
				.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function) nnp_cgemm_conjb_transc_upto_2x2__scalar,
#endif /* !NNP_INFERENCE_ONLY */
			};
			nnp_hwinfo.supported = true;
		#else
			#error Unsupported backend
		#endif
	}

	nnp_hwinfo.initialized = true;
}

enum nnp_status nnp_initialize(void) {
	if (!cpuinfo_initialize()) {
		return nnp_status_out_of_memory;
	}
	pthread_once(&hwinfo_init_control, &init_hwinfo);
	if (nnp_hwinfo.supported) {
		return nnp_status_success;
	} else {
		return nnp_status_unsupported_hardware;
	}
}

enum nnp_status nnp_deinitialize(void) {
	cpuinfo_deinitialize();
	return nnp_status_success;
}
