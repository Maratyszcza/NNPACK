#pragma once

#include <math.h>


static inline float relu(float data, float negative_slope) {
	return signbit(data) ? data * negative_slope : data;
}

static inline float grad_relu(float grad_output_data, float input_data, float negative_slope) {
	return signbit(input_data) ? grad_output_data * negative_slope : grad_output_data;
}

#ifdef PSIMD_H
	static inline psimd_f32 psimd_relu_f32(psimd_f32 data, psimd_f32 negative_slope) {
		return psimd_signblend_f32(data, data * negative_slope, data);
	}

	static inline psimd_f32 psimd_grad_relu_f32(psimd_f32 grad_output_data, psimd_f32 input_data, psimd_f32 negative_slope) {
		return psimd_signblend_f32(input_data, grad_output_data * negative_slope, grad_output_data);
	}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	#include <nnpack/arm_neon.h>

	static inline float32x4_t neon_reluq_f32(float32x4_t data, float32x4_t negative_slope) {
		const uint32x4_t negative_mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(data), 31));
		return vbslq_f32(negative_mask, vmulq_f32(data, negative_slope), data);
	}

	static inline float32x4_t neon_grad_reluq_f32(float32x4_t grad_output_data, float32x4_t input_data, float32x4_t negative_slope) {
		const uint32x4_t negative_mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(input_data), 31));
		return vbslq_f32(negative_mask, vmulq_f32(grad_output_data, negative_slope), grad_output_data);
	}

	static inline float32x2_t neon_relu_f32(float32x2_t data, float32x2_t negative_slope) {
		const uint32x2_t negative_mask = vreinterpret_u32_s32(vshr_n_s32(vreinterpret_s32_f32(data), 31));
		return vbsl_f32(negative_mask, vmul_f32(data, negative_slope), data);
	}

	static inline float32x2_t neon_grad_relu_f32(float32x2_t grad_output_data, float32x2_t input_data, float32x2_t negative_slope) {
		const uint32x2_t negative_mask = vreinterpret_u32_s32(vshr_n_s32(vreinterpret_s32_f32(input_data), 31));
		return vbsl_f32(negative_mask, vmul_f32(grad_output_data, negative_slope), grad_output_data);
	}
#endif
