#pragma once

#include <arm_neon.h>


static inline void vswapq_f32(
	float32x4_t a[restrict static 1],
	float32x4_t b[restrict static 1])
{
	const float32x4_t new_a = *b;
	const float32x4_t new_b = *a;
	*a = new_a;
	*b = new_b;
}

static inline float32x4_t vld1q_f32_aligned(const float* address) {
	return vld1q_f32((const float*) __builtin_assume_aligned(address, sizeof(float32x4_t)));
}

static inline void vst1q_f32_aligned(float* address, float32x4_t vector) {
	vst1q_f32((float*) __builtin_assume_aligned(address, sizeof(float32x4_t)), vector);
}

#if defined(__aarch64__) || (defined(__ARM_NEON_FP) && (__ARM_NEON_FP & 2))
	#ifdef __clang__
		static inline float32x4_t vld1q_f32_f16(const void* address) {
			return vcvt_f32_f16(vld1_f16((const __fp16*) address));
		}

		static inline float32x4_t vld1q_f32_f16_aligned(const void* address) {
			return vcvt_f32_f16(vld1_f16((const __fp16*)
				__builtin_assume_aligned(address, sizeof(float16x4_t))));
		}

		static inline void vst1q_f16_f32(void* address, float32x4_t vector) {
			vst1_f16((__fp16*) address, vcvt_f16_f32(vector));
		}

		static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector) {
			vst1_f16((__fp16*) __builtin_assume_aligned(address, sizeof(float16x4_t)),
				vcvt_f16_f32(vector));
		}
	#else
		// GCC 4.x doesn't support vst1_f16/vld1_f16, workaround.
		static inline float32x4_t vld1q_f32_f16(const void* address) {
			return vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
		}

		static inline float32x4_t vld1q_f32_f16_aligned(const void* address) {
			return vcvt_f32_f16((float16x4_t)
				vld1_u16((const uint16_t*) __builtin_assume_aligned(address, sizeof(float16x4_t))));
		}

		static inline void vst1q_f16_f32(void* address, float32x4_t vector) {
			vst1_u16((uint16_t*) address, (uint16x4_t) vcvt_f16_f32(vector));
		}

		static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector) {
			vst1_u16((uint16_t*) __builtin_assume_aligned(address, sizeof(uint16x4_t)),
				(uint16x4_t) vcvt_f16_f32(vector));
		}
	#endif
#endif

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
