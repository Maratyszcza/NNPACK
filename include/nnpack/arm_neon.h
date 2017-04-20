#pragma once

#include <arm_neon.h>

static inline float32x4_t vld1q_f32_aligned(const float* address) {
	return vld1q_f32((const float*) __builtin_assume_aligned(address, sizeof(float32x4_t)));
}

static inline void vst1q_f32_aligned(float* address, float32x4_t vector) {
	vst1q_f32((float*) __builtin_assume_aligned(address, sizeof(float32x4_t)), vector);
}

static inline float32x4_t vmuladdq_f32(float32x4_t c, float32x4_t a, float32x4_t b) {
	#if defined(__aarch64__)
		return vfmaq_f32(c, a, b);
	#else
		return vmlaq_f32(c, a, b);
	#endif
}

static inline float32x4_t vmulsubq_f32(float32x4_t c, float32x4_t a, float32x4_t b) {
	#if defined(__aarch64__)
		return vfmsq_f32(c, a, b);
	#else
		return vmlsq_f32(c, a, b);
	#endif
}

static inline float32x2_t vmuladd_f32(float32x2_t c, float32x2_t a, float32x2_t b) {
	#if defined(__aarch64__)
		return vfma_f32(c, a, b);
	#else
		return vmla_f32(c, a, b);
	#endif
}

static inline float32x2_t vmulsub_f32(float32x2_t c, float32x2_t a, float32x2_t b) {
	#if defined(__aarch64__)
		return vfms_f32(c, a, b);
	#else
		return vmls_f32(c, a, b);
	#endif
}

