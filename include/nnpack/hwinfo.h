#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <nnpack/macros.h>
#include <nnpack/transform.h>
#include <nnpack/blas.h>
#include <nnpack/relu.h>
#include <nnpack/softmax.h>

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
	nnp_transform_2d_with_offset fft8x8_with_offset_and_store;
	nnp_transform_2d_with_offset fft8x8_with_offset_and_stream;
	nnp_transform_2d_with_offset ifft8x8_with_offset;
	nnp_transform_2d_with_bias ifft8x8_with_bias;
	nnp_transform_2d_with_bias ifft8x8_with_bias_with_relu;
	nnp_transform_2d_with_offset fft16x16_with_offset_and_store;
	nnp_transform_2d_with_offset fft16x16_with_offset_and_stream;
	nnp_transform_2d_with_offset ifft16x16_with_offset;
	nnp_transform_2d_with_bias ifft16x16_with_bias;
	nnp_transform_2d_with_bias ifft16x16_with_bias_with_relu;
	nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_store;
	nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_stream;
	nnp_transform_2d_with_offset kwt_f6x6_3x3;
	nnp_transform_2d_with_offset kwt_f6x6_3Rx3R;
	nnp_transform_2d_with_offset owt_f6x6_3x3;
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias;
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias_with_relu;
};

struct blockmac {
	nnp_blockmac fourier8x8_mac_with_conj;
	nnp_blockmac fourier16x16_mac_with_conj;
	nnp_blockmac winograd8x8_mac;
};

struct activations {
	nnp_inplace_relu_function inplace_relu;
	nnp_outplace_relu_function outplace_relu;
	nnp_gradient_relu_function outplace_grad_relu;
	nnp_inplace_softmax_function inplace_softmax;
	nnp_outplace_softmax_function outplace_softmax;
};

struct sgemm {
	nnp_fast_sgemm_function only_mr_x_nr;
	nnp_full_sgemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sxgemm {
	nnp_fast_tuple_gemm_function only_mr_x_nr;
	nnp_full_tuple_gemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct cxgemm {
	nnp_fast_tuple_gemm_function s4cX_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function s4cX_conjb_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_conjb_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_conjb_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_conjb_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function s4cX_conjb_transc_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_conjb_transc_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_conjb_transc_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_conjb_transc_upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sdotxf {
	const nnp_sdotxf_function* functions;
	uint32_t fusion;
};

struct shdotxf {
	const nnp_shdotxf_function* functions;
	uint32_t fusion;
};

struct hardware_info {
	bool initialized;
	bool supported;
	uint32_t simd_width;

	struct cache_hierarchy_info cache;
	struct cache_blocking_info blocking;

	struct transforms transforms;
	struct blockmac blockmac;
	struct activations activations;
	struct sgemm sgemm;
	struct sxgemm sxgemm;
	struct cxgemm cxgemm;
	struct sdotxf sdotxf;
	struct shdotxf shdotxf;

	struct isa_info isa;
};

extern struct hardware_info nnp_hwinfo;

#ifdef __cplusplus
} /* extern "C" */
#endif
