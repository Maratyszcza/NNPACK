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
