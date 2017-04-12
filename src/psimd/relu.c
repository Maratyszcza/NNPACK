#include <stddef.h>

#include <psimd.h>

#include <nnpack/activations.h>


void nnp_relu__psimd(
	const float input[restrict static 4],
	float output[restrict static 4],
	size_t length,
	float negative_slope)
{
	const psimd_f32 vec_negative_slope = psimd_splat_f32(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		psimd_store_f32(output,
			psimd_relu_f32(psimd_load_f32(input), vec_negative_slope));

		input  += 4;
		output += 4;
		length -= 4;
	} while (length != 0);
}

void nnp_inplace_relu__psimd(
	float data[restrict static 4],
	size_t length,
	float negative_slope)
{
	const psimd_f32 vec_negative_slope = psimd_splat_f32(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		psimd_store_f32(data,
			psimd_relu_f32(psimd_load_f32(data), vec_negative_slope));

		data += 4;
		length -= 4;
	} while (length != 0);
}

void nnp_grad_relu__psimd(
	const float output_gradient[restrict static 4],
	const float input[restrict static 4],
	float input_gradient[restrict static 4],
	size_t length,
	float negative_slope)
{
	const psimd_f32 vec_negative_slope = psimd_splat_f32(negative_slope);

	/* Length is always non-zero and proportional to SIMD width */
	do {
		psimd_store_f32(input_gradient,
			psimd_grad_relu_f32(psimd_load_f32(output_gradient), psimd_load_f32(input), vec_negative_slope));

		output_gradient += 4;
		input += 4;
		input_gradient += 4;
		length -= 4;
	} while (length != 0);
}
