#pragma once

#include <math.h>


static inline float relu(float data, float negative_slope) {
	return signbit(data) ? data * negative_slope : data;
}

static inline float grad_relu(float grad_output_data, float input_data, float negative_slope) {
	return signbit(input_data) ? grad_output_data * negative_slope : grad_output_data;
}
