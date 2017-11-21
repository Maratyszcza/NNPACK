#include <stdint.h>
#include <stddef.h>

#include <psimd.h>

#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <psimd/winograd/f6x6k3x3.h>
#include <psimd/transpose.h>


void nnp_iwt8x8_3x3_with_offset__psimd(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	NNP_SIMD_ALIGN float block[8][8] = { 0 };
	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			block[row_offset + i][column_offset + j] = data[i * data_stride + j];
		}
	}

	psimd_f32 wd[8][2];
	for (size_t col = 0; col < 2; col++) {
		winograd_f6k3_input_transform(
			psimd_load_f32(&block[0][col * 4]),
			psimd_load_f32(&block[1][col * 4]),
			psimd_load_f32(&block[2][col * 4]),
			psimd_load_f32(&block[3][col * 4]),
			psimd_load_f32(&block[4][col * 4]),
			psimd_load_f32(&block[5][col * 4]),
			psimd_load_f32(&block[6][col * 4]),
			psimd_load_f32(&block[7][col * 4]),
			&wd[0][col], &wd[1][col], &wd[2][col], &wd[3][col], &wd[4][col], &wd[5][col], &wd[6][col], &wd[7][col]);
		psimd_transpose4x4_f32(
			wd[0][col], wd[1][col], wd[2][col], wd[3][col],
			&wd[0][col], &wd[1][col], &wd[2][col], &wd[3][col]);
		psimd_transpose4x4_f32(
			wd[4][col], wd[5][col], wd[6][col], wd[7][col],
			&wd[4][col], &wd[5][col], &wd[6][col], &wd[7][col]);
	}
	psimd_swap_f32(&wd[4][0], &wd[0][1]);
	psimd_swap_f32(&wd[5][0], &wd[1][1]);
	psimd_swap_f32(&wd[6][0], &wd[2][1]);
	psimd_swap_f32(&wd[7][0], &wd[3][1]);

	for (size_t col = 0; col < 2; col++) {
		winograd_f6k3_input_transform(
			wd[0][col], wd[1][col], wd[2][col], wd[3][col], wd[4][col], wd[5][col], wd[6][col], wd[7][col],
			&wd[0][col], &wd[1][col], &wd[2][col], &wd[3][col], &wd[4][col], &wd[5][col], &wd[6][col], &wd[7][col]);
	}
	for (size_t col = 0; col < 2; col++) {
		for (size_t row = 0; row < 8; row++) {        
			psimd_store_f32(transform, wd[row][col]);
			transform += transform_stride;
		}
	}
}

void nnp_kwt8x8_3x3__psimd(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	const psimd_f32 g0 = psimd_load_f32(g);
	const psimd_f32 g1 = psimd_load_f32(g + 3);
	const psimd_f32 g5678 = psimd_load_f32(g + 5);
	#ifdef __clang__
		const psimd_f32 g2 = __builtin_shufflevector(g5678, g5678, 1, 2, 3, -1);
	#else
		const psimd_f32 g2 = __builtin_shuffle(g5678, g5678, (psimd_s32) { 1, 2, 3, -1 });
	#endif

	psimd_f32 w[8];
	winograd_f6k3_kernel_transform(g0, g1, g2,
		&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7],
		true /* rescale coefficients */);

	psimd_transpose4x4_f32(
		w[0], w[1], w[2], w[3],
		&w[0], &w[1], &w[2], &w[3]);
	psimd_transpose4x4_f32(
		w[4], w[5], w[6], w[7],
		&w[4], &w[5], &w[6], &w[7]);

	psimd_f32 wg[8][2];
	winograd_f6k3_kernel_transform(w[0], w[1], w[2],
		&wg[0][0], &wg[1][0], &wg[2][0], &wg[3][0], &wg[4][0], &wg[5][0], &wg[6][0], &wg[7][0],
		true /* rescale coefficients */);
	winograd_f6k3_kernel_transform(w[4], w[5], w[6],
		&wg[0][1], &wg[1][1], &wg[2][1], &wg[3][1], &wg[4][1], &wg[5][1], &wg[6][1], &wg[7][1],
		true /* rescale coefficients */);

	for (size_t col = 0; col < 2; col++) {
		for (size_t row = 0; row < 8; row++) {
			psimd_store_f32(transform, wg[row][col]);
			transform += transform_stride;
		}
	}
}

#if !NNP_INFERENCE_ONLY
void nnp_kwt8x8_3Rx3R__psimd(
	const float g[restrict static 9],
	float transform[restrict static 1],
	size_t stride_g, size_t stride_wg,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	stride_wg /= sizeof(float);

	const psimd_f32 g5678 = psimd_load_f32(g + 5);
	const psimd_f32 g2345 = psimd_load_f32(g + 2);
	const psimd_f32 g0123 = psimd_load_f32(g);

	#ifdef __clang__
		const psimd_f32 g0 = __builtin_shufflevector(g5678, g5678, 3, 2, 1, -1);
		const psimd_f32 g1 = __builtin_shufflevector(g2345, g2345, 3, 2, 1, -1);
		const psimd_f32 g2 = __builtin_shufflevector(g0123, g0123, 2, 1, 0, -1);
	#else
		const psimd_f32 g0 = __builtin_shuffle(g5678, (psimd_s32) { 3, 2, 1, -1 });
		const psimd_f32 g1 = __builtin_shuffle(g2345, (psimd_s32) { 3, 2, 1, -1 });
		const psimd_f32 g2 = __builtin_shuffle(g0123, (psimd_s32) { 2, 1, 0, -1 });
	#endif

	psimd_f32 w[8];
	winograd_f6k3_kernel_transform(g0, g1, g2,
		&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7],
		true /* rescale coefficients */);

	psimd_transpose4x4_f32(
		w[0], w[1], w[2], w[3],
		&w[0], &w[1], &w[2], &w[3]);
	psimd_transpose4x4_f32(
		w[4], w[5], w[6], w[7],
		&w[4], &w[5], &w[6], &w[7]);

	psimd_f32 wg[8][2];
	winograd_f6k3_kernel_transform(w[0], w[1], w[2],
		&wg[0][0], &wg[1][0], &wg[2][0], &wg[3][0], &wg[4][0], &wg[5][0], &wg[6][0], &wg[7][0],
		true /* rescale coefficients */);
	winograd_f6k3_kernel_transform(w[4], w[5], w[6],
		&wg[0][1], &wg[1][1], &wg[2][1], &wg[3][1], &wg[4][1], &wg[5][1], &wg[6][1], &wg[7][1],
		true /* rescale coefficients */);

	for (size_t col = 0; col < 2; col++) {
		for (size_t row = 0; row < 8; row++) {
			psimd_store_f32(transform, wg[row][col]);
			transform += stride_wg;
		}
	}
}

void nnp_owt8x8_3x3__psimd(
	const float transform[restrict static 1],
	float output[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	psimd_f32 s[8][2];
	for (size_t col = 0; col < 2; col++) {
		const psimd_f32 m0 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m1 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m2 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m3 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m4 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m5 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m6 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m7 = psimd_load_f32(transform);
		transform += transform_stride;

		winograd_f6k3_output_transform(m0, m1, m2, m3, m4, m5, m6, m7,
			&s[0][col], &s[1][col], &s[2][col], &s[3][col], &s[4][col], &s[5][col]);
		psimd_transpose4x4_f32(
			s[0][col], s[1][col], s[2][col], s[3][col],
			&s[0][col], &s[1][col], &s[2][col], &s[3][col]);
		psimd_transpose4x4_f32(
			s[4][col], s[5][col], s[6][col], s[7][col],
			&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
	}

	psimd_swap_f32(&s[4][0], &s[0][1]);
	psimd_swap_f32(&s[5][0], &s[1][1]);
	psimd_swap_f32(&s[6][0], &s[2][1]);
	psimd_swap_f32(&s[7][0], &s[3][1]);

	NNP_SIMD_ALIGN float block[6][8];
	for (size_t col = 0; col < 2; col++) {
		psimd_f32 t0, t1, t2, t3, t4, t5;
		winograd_f6k3_output_transform(
			s[0][col],
			s[1][col],
			s[2][col],
			s[3][col],
			s[4][col],
			s[5][col],
			s[6][col],
			s[7][col],
			&t0, &t1, &t2, &t3, &t4, &t5);
		psimd_store_f32(&block[0][col * 4], t0);
		psimd_store_f32(&block[1][col * 4], t1);
		psimd_store_f32(&block[2][col * 4], t2);
		psimd_store_f32(&block[3][col * 4], t3);
		psimd_store_f32(&block[4][col * 4], t4);
		psimd_store_f32(&block[5][col * 4], t5);
	}

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j];
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_owt8x8_3x3_with_bias__psimd(
	const float transform[restrict static 1],
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	psimd_f32 s[8][2];
	for (size_t col = 0; col < 2; col++) {
		const psimd_f32 m0 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m1 = (col == 0) ? psimd_load_f32(transform) + (psimd_f32) { 0, *bias, 0, 0 } : psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m2 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m3 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m4 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m5 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m6 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m7 = psimd_load_f32(transform);
		transform += transform_stride;

		winograd_f6k3_output_transform(m0, m1, m2, m3, m4, m5, m6, m7,
			&s[0][col], &s[1][col], &s[2][col], &s[3][col], &s[4][col], &s[5][col]);
		psimd_transpose4x4_f32(
			s[0][col], s[1][col], s[2][col], s[3][col],
			&s[0][col], &s[1][col], &s[2][col], &s[3][col]);
		psimd_transpose4x4_f32(
			s[4][col], s[5][col], s[6][col], s[7][col],
			&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
	}

	psimd_swap_f32(&s[4][0], &s[0][1]);
	psimd_swap_f32(&s[5][0], &s[1][1]);
	psimd_swap_f32(&s[6][0], &s[2][1]);
	psimd_swap_f32(&s[7][0], &s[3][1]);

	NNP_SIMD_ALIGN float block[6][8];
	for (size_t col = 0; col < 2; col++) {
		psimd_f32 t0, t1, t2, t3, t4, t5;
		winograd_f6k3_output_transform(
			s[0][col],
			s[1][col],
			s[2][col],
			s[3][col],
			s[4][col],
			s[5][col],
			s[6][col],
			s[7][col],
			&t0, &t1, &t2, &t3, &t4, &t5);
		psimd_store_f32(&block[0][col * 4], t0);
		psimd_store_f32(&block[1][col * 4], t1);
		psimd_store_f32(&block[2][col * 4], t2);
		psimd_store_f32(&block[3][col * 4], t3);
		psimd_store_f32(&block[4][col * 4], t4);
		psimd_store_f32(&block[5][col * 4], t5);
	}

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j];
		}
	}
}

void nnp_owt8x8_3x3_with_bias_with_relu__psimd(
	const float transform[restrict static 1],
	float output[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t output_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	psimd_f32 s[8][2];
	for (size_t col = 0; col < 2; col++) {
		const psimd_f32 m0 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m1 = (col == 0) ? psimd_load_f32(transform) + (psimd_f32) { 0, *bias, 0, 0 } : psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m2 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m3 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m4 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m5 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m6 = psimd_load_f32(transform);
		transform += transform_stride;
		const psimd_f32 m7 = psimd_load_f32(transform);
		transform += transform_stride;

		winograd_f6k3_output_transform(m0, m1, m2, m3, m4, m5, m6, m7,
			&s[0][col], &s[1][col], &s[2][col], &s[3][col], &s[4][col], &s[5][col]);
		psimd_transpose4x4_f32(
			s[0][col], s[1][col], s[2][col], s[3][col],
			&s[0][col], &s[1][col], &s[2][col], &s[3][col]);
		psimd_transpose4x4_f32(
			s[4][col], s[5][col], s[6][col], s[7][col],
			&s[4][col], &s[5][col], &s[6][col], &s[7][col]);
	}

	psimd_swap_f32(&s[4][0], &s[0][1]);
	psimd_swap_f32(&s[5][0], &s[1][1]);
	psimd_swap_f32(&s[6][0], &s[2][1]);
	psimd_swap_f32(&s[7][0], &s[3][1]);

	NNP_SIMD_ALIGN float block[6][8];
	for (size_t col = 0; col < 2; col++) {
		psimd_f32 t0, t1, t2, t3, t4, t5;
		winograd_f6k3_output_transform(
			s[0][col],
			s[1][col],
			s[2][col],
			s[3][col],
			s[4][col],
			s[5][col],
			s[6][col],
			s[7][col],
			&t0, &t1, &t2, &t3, &t4, &t5);
		psimd_store_f32(&block[0][col * 4], psimd_relu_f32(t0, psimd_zero_f32()));
		psimd_store_f32(&block[1][col * 4], psimd_relu_f32(t1, psimd_zero_f32()));
		psimd_store_f32(&block[2][col * 4], psimd_relu_f32(t2, psimd_zero_f32()));
		psimd_store_f32(&block[3][col * 4], psimd_relu_f32(t3, psimd_zero_f32()));
		psimd_store_f32(&block[4][col * 4], psimd_relu_f32(t4, psimd_zero_f32()));
		psimd_store_f32(&block[5][col * 4], psimd_relu_f32(t5, psimd_zero_f32()));
	}

	for (size_t i = 0; i < row_count; i++) {
		for (size_t j = 0; j < column_count; j++) {
			output[i * output_stride + j] = block[i][j];
		}
	}
}
