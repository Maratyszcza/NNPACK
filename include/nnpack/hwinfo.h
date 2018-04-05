#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <nnpack/macros.h>

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

#if NNP_BACKEND_SCALAR
	#define NNP_COMPLEX_TUPLE_INDEX 2
#else
	#define NNP_COMPLEX_TUPLE_INDEX 1
#endif

typedef void (*nnp_transform_2d)(const void*, void*, size_t, size_t, uint32_t, uint32_t);
typedef void (*nnp_transform_2d_with_bias)(const void*, void*, const void*, size_t, size_t, uint32_t, uint32_t);
typedef void (*nnp_transform_2d_with_offset)(const void*, void*, size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t);

typedef void (*nnp_fast_sgemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*nnp_full_sgemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);

typedef void (*nnp_fast_conv_function)(size_t, size_t, const float*, const float*, float*);
typedef void (*nnp_full_conv_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*);

typedef void (*nnp_fast_tuple_gemm_function)(size_t, size_t, const void*, const void*, void*, size_t);
typedef void (*nnp_full_tuple_gemm_function)(uint32_t, uint32_t, size_t, size_t, const void*, const void*, void*, size_t);

typedef void (*nnp_sdotxf_function)(const float*, const float*, size_t, float*, size_t);
typedef void (*nnp_shdotxf_function)(const float*, const void*, size_t, float*, size_t);

typedef void (*nnp_relu_function)(const float*, float*, size_t, float);
typedef void (*nnp_inplace_relu_function)(float*, size_t, float);
typedef void (*nnp_grad_relu_function)(const float*, const float*, float*, size_t, float);

typedef void (*nnp_softmax_function)(size_t, const float*, float*);
typedef void (*nnp_inplace_softmax_function)(size_t, float*);

struct transforms {
	nnp_transform_2d_with_offset fft8x8_with_offset_and_store;
	nnp_transform_2d_with_offset fft8x8_with_offset_and_stream;
#if !NNP_INFERENCE_ONLY
	nnp_transform_2d_with_offset ifft8x8_with_offset;
#endif
	nnp_transform_2d_with_bias ifft8x8_with_bias;
	nnp_transform_2d_with_bias ifft8x8_with_bias_with_relu;
	nnp_transform_2d_with_offset fft16x16_with_offset_and_store;
	nnp_transform_2d_with_offset fft16x16_with_offset_and_stream;
#if !NNP_INFERENCE_ONLY
	nnp_transform_2d_with_offset ifft16x16_with_offset;
#endif
	nnp_transform_2d_with_bias ifft16x16_with_bias;
	nnp_transform_2d_with_bias ifft16x16_with_bias_with_relu;
	nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_store;
	nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_stream;
	nnp_transform_2d_with_offset kwt_f6x6_3x3;
#if !NNP_INFERENCE_ONLY
	nnp_transform_2d_with_offset kwt_f6x6_3Rx3R;
	nnp_transform_2d_with_offset owt_f6x6_3x3;
#endif
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias;
	nnp_transform_2d_with_bias owt_f6x6_3x3s2_with_bias;
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias_with_relu;
	nnp_transform_2d_with_bias owt_f6x6_3x3s2_with_bias_with_relu;
#if NNP_BACKEND_ARM
	nnp_transform_2d_with_offset iwt_f6x6_3x3_fp16_with_offset;
	nnp_transform_2d_with_offset kwt_f6x6_3x3_fp16;
	nnp_transform_2d_with_bias owt_f6x6_3x3_fp16_with_bias;
	nnp_transform_2d_with_bias owt_f6x6_3x3_fp16_with_bias_with_relu;
#endif /* NNP_BACKEND_ARM */
};

#if !NNP_CONVOLUTION_ONLY
struct activations {
	nnp_relu_function relu;
	nnp_inplace_relu_function inplace_relu;
	nnp_grad_relu_function grad_relu;
	nnp_softmax_function softmax;
	nnp_inplace_softmax_function inplace_softmax;
};
#endif

struct convolution {
	nnp_fast_conv_function only_mr_x_nr;
	nnp_full_conv_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
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

#if NNP_BACKEND_ARM
struct hxgemm {
	nnp_fast_tuple_gemm_function only_mr_x_nr;
	nnp_full_tuple_gemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};
#endif /* NNP_BACKEND_ARM */

struct cxgemm {
#if !NNP_INFERENCE_ONLY
	nnp_fast_tuple_gemm_function s4cX_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_upto_mr_x_nr;
#endif
	nnp_fast_tuple_gemm_function s4cX_conjb_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_conjb_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_conjb_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_conjb_upto_mr_x_nr;
#if !NNP_INFERENCE_ONLY
	nnp_fast_tuple_gemm_function s4cX_conjb_transc_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_conjb_transc_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_conjb_transc_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_conjb_transc_upto_mr_x_nr;
#endif
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
#if !NNP_CONVOLUTION_ONLY
	struct activations activations;
#endif
	struct convolution conv1x1;
	struct sgemm sgemm;
	struct sxgemm sxgemm;
#if NNP_BACKEND_ARM
	struct hxgemm hxgemm;
#endif /* NNP_BACKEND_ARM */
	struct cxgemm cxgemm;
#if !NNP_CONVOLUTION_ONLY
	struct sdotxf sdotxf;
	struct shdotxf shdotxf;
#endif

	struct isa_info isa;
};

extern struct hardware_info nnp_hwinfo;

#ifdef __cplusplus
} /* extern "C" */
#endif
