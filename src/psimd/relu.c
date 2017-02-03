#include <stddef.h>

#include <nnpack/simd.h>


static inline v4f v4f_relu(v4f data, v4f negative_slope) {
	return v4f_blend(data > v4f_zero(), data, data * negative_slope);
}

static inline v4f v4f_grad_relu(v4f grad_output_data, v4f input_data, v4f negative_slope) {
	return v4f_signblend(input_data, grad_output_data * negative_slope, grad_output_data);
}

void nnp_inplace_relu_forward__psimd(
	float data[restrict static 4],
	size_t length,
	float negative_slope)
{
	const v4f vec_negative_slope = v4f_splat(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		v4f_st(data, v4f_relu(v4f_ld(data), vec_negative_slope));

		data += 4;
		length -= 4;
	} while (length != 0);
}

void nnp_outplace_relu_forward__psimd(
	const float input[restrict static 4],
	float output[restrict static 4],
	size_t length,
	float negative_slope)
{
	const v4f vec_negative_slope = v4f_splat(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		v4f_st(output, v4f_relu(v4f_ld(input), vec_negative_slope));

		input  += 4;
		output += 4;
		length -= 4;
	} while (length != 0);
}

void nnp_relu_backward__psimd(
	const float output_gradient[restrict static 4],
	const float input[restrict static 4],
	float input_gradient[restrict static 4],
	size_t length,
	float negative_slope)
{
	const v4f vec_negative_slope = v4f_splat(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		v4f_st(input_gradient, v4f_grad_relu(v4f_ld(output_gradient), v4f_ld(input), vec_negative_slope));

		output_gradient += 4;
		input += 4;
		input_gradient += 4;
		length -= 4;
	} while (length != 0);
}
