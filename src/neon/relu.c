#include <stddef.h>

#include <nnpack/arm_neon.h>
#include <nnpack/activations.h>


void nnp_relu__neon(
	const float input[restrict static 4],
	float output[restrict static 4],
	size_t length,
	float negative_slope)
{
	const float32x4_t vec_negative_slope = vdupq_n_f32(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		vst1q_f32(output,
			neon_relu_f32(vld1q_f32(input), vec_negative_slope));

		input  += 4;
		output += 4;
		length -= 4;
	} while (length != 0);
}

void nnp_inplace_relu__neon(
	float data[restrict static 4],
	size_t length,
	float negative_slope)
{
	const float32x4_t vec_negative_slope = vdupq_n_f32(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		vst1q_f32(data,
			neon_relu_f32(vld1q_f32(data), vec_negative_slope));

		data += 4;
		length -= 4;
	} while (length != 0);
}

void nnp_grad_relu__neon(
	const float output_gradient[restrict static 4],
	const float input[restrict static 4],
	float input_gradient[restrict static 4],
	size_t length,
	float negative_slope)
{
	const float32x4_t vec_negative_slope = vdupq_n_f32(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		vst1q_f32(input_gradient,
			neon_grad_relu_f32(vld1q_f32(output_gradient), vld1q_f32(input), vec_negative_slope));

		output_gradient += 4;
		input += 4;
		input_gradient += 4;
		length -= 4;
	} while (length != 0);
}
