#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <nnpack/macros.h>
#include <nnpack/transform.h>
#include <nnpack/blas.h>

#ifdef __cplusplus
extern "C" {
#endif

struct isa_info {
	bool has_avx;
	bool has_fma3;
	bool has_avx2;
};

struct cache_info {
	uint32_t size;
	uint32_t associativity;
	uint32_t threads;
	bool inclusive;
};

struct cache_hierarchy_info {
	struct cache_info l1;
	struct cache_info l2;
	struct cache_info l3;
	struct cache_info l4;
};

struct cache_blocking_info {
	size_t l1;
	size_t l2;
	size_t l3;
	size_t l4;
};

struct transforms {
	nnp_transform_2d fft8x8_and_store;
	nnp_transform_2d fft8x8_and_stream;
	nnp_fourier_transform_2d_and_mac fft8x8_and_mac_by_conj;
	nnp_transform_2d ifft8x8;
	nnp_transform_2d_with_bias ifft8x8_with_bias;
	nnp_transform_2d fft16x16_and_store;
	nnp_transform_2d fft16x16_and_stream;
	nnp_fourier_transform_2d_and_mac fft16x16_and_mac_by_conj;
	nnp_transform_2d ifft16x16;
	nnp_transform_2d_with_bias ifft16x16_with_bias;
	nnp_transform_2d iwt_f6x6_3x3_and_store;
	nnp_transform_2d iwt_f6x6_3x3_and_stream;
	nnp_transform_2d kwt_f6x6_3x3;
	nnp_transform_2d kwt_f6x6_3Rx3R;
	nnp_kernel_winograd_transform_2d_and_mac kwt_f6x6_3x3_and_mac;
	nnp_transform_2d owt_f6x6_3x3;
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias;
};

struct blockmac {
	nnp_blockmac fourier8x8_mac_with_conj;
	nnp_blockmac fourier16x16_mac_with_conj;
	nnp_blockmac winograd8x8_mac;
};

struct sgemm {
	uint32_t mr;
	uint32_t nr;
	nnp_fast_sgemm_function fast_sgemm_function;
	nnp_full_sgemm_function full_sgemm_function;
};

struct sxgemm {
	uint32_t mr;
	uint32_t nr;
	nnp_tuple_gemm_function* functions;
};

struct cxgemm {
	uint32_t mr;
	uint32_t nr;
	nnp_tuple_gemm_function* functions;
};

struct sdotxf {
	uint32_t fusion;
	nnp_sdotxf_function* functions;
};

struct hardware_info {
	bool initialized;
	bool supported;
	uint32_t simd_width;

	struct cache_hierarchy_info cache;
	struct cache_blocking_info blocking;

	struct transforms transforms;
	struct blockmac blockmac;
	struct sgemm sgemm;
	struct sxgemm sxgemm;
	struct cxgemm cxgemm;
	struct sdotxf sdotxf;

	struct isa_info isa;
};

extern struct hardware_info nnp_hwinfo;

#ifdef __cplusplus
} /* extern "C" */
#endif
