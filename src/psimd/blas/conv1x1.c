#include <stddef.h>
#include <stdint.h>

#include <psimd.h>

#include <nnpack/macros.h>


static NNP_SIMD_ALIGN int32_t mask_array[8] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};


void nnp_conv1x1_only_2x4__psimd(
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float* input0 = input;
	const float* input1 = input + image_size;

	const psimd_f32 vkernel00 = psimd_splat_f32(kernel[0]);
	const psimd_f32 vkernel01 = psimd_splat_f32(kernel[1]);
	kernel += input_channels;
	const psimd_f32 vkernel10 = psimd_splat_f32(kernel[0]);
	const psimd_f32 vkernel11 = psimd_splat_f32(kernel[1]);
	kernel += input_channels;
	const psimd_f32 vkernel20 = psimd_splat_f32(kernel[0]);
	const psimd_f32 vkernel21 = psimd_splat_f32(kernel[1]);
	kernel += input_channels;
	const psimd_f32 vkernel30 = psimd_splat_f32(kernel[0]);
	const psimd_f32 vkernel31 = psimd_splat_f32(kernel[1]);

	float* output0 = output;
	float* output1 = output0 + image_size;
	float* output2 = output1 + image_size;
	float* output3 = output2 + image_size;
	while (image_size >= 4) {
		psimd_f32 voutput0 = psimd_load_f32(output0);
		psimd_f32 voutput1 = psimd_load_f32(output1);
		psimd_f32 voutput2 = psimd_load_f32(output2);
		psimd_f32 voutput3 = psimd_load_f32(output3);

		const psimd_f32 vinput0 = psimd_load_f32(input0);
		input0 += 4;
		voutput0 += vkernel00 * vinput0;
		voutput1 += vkernel10 * vinput0;
		voutput2 += vkernel20 * vinput0;
		voutput3 += vkernel30 * vinput0;

		const psimd_f32 vinput1 = psimd_load_f32(input1);
		input1 += 4;
		voutput0 += vkernel01 * vinput1;
		voutput1 += vkernel11 * vinput1;
		voutput2 += vkernel21 * vinput1;
		voutput3 += vkernel31 * vinput1;

		psimd_store_f32(output0, voutput0);
		output0 += 4;
		psimd_store_f32(output1, voutput1);
		output1 += 4;
		psimd_store_f32(output2, voutput2);
		output2 += 4;
		psimd_store_f32(output3, voutput3);
		output3 += 4;

		image_size -= 4;
	}
	if (image_size != 0) {
		const psimd_s32 vmask = psimd_load_s32(&mask_array[image_size]);

		output0 = output0 + image_size - 4;
		psimd_f32 voutput0 = psimd_load_f32(output0);
		output1 = output1 + image_size - 4;
		psimd_f32 voutput1 = psimd_load_f32(output1);
		output2 = output2 + image_size - 4;
		psimd_f32 voutput2 = psimd_load_f32(output2);
		output3 = output3 + image_size - 4;
		psimd_f32 voutput3 = psimd_load_f32(output3);

		const psimd_f32 vinput0 = psimd_andmask_f32(vmask, psimd_load_f32(&input0[image_size - 4]));
		voutput0 += vkernel00 * vinput0;
		voutput1 += vkernel10 * vinput0;
		voutput2 += vkernel20 * vinput0;
		voutput3 += vkernel30 * vinput0;

		const psimd_f32 vinput1 = psimd_andmask_f32(vmask, psimd_load_f32(&input1[image_size - 4]));
		voutput0 += vkernel01 * vinput1;
		voutput1 += vkernel11 * vinput1;
		voutput2 += vkernel21 * vinput1;
		voutput3 += vkernel31 * vinput1;

		psimd_store_f32(output0, voutput0);
		psimd_store_f32(output1, voutput1);
		psimd_store_f32(output2, voutput2);
		psimd_store_f32(output3, voutput3);
	}
}

void nnp_conv1x1_upto_2x4__psimd(
	uint32_t input_channels_subblock_size,
	uint32_t output_channels_subblock_size,
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float* input0 = input;
	const float* input1 = input + image_size;

	psimd_f32 vkernel00, vkernel01, vkernel10, vkernel11, vkernel20, vkernel21, vkernel30, vkernel31;
	vkernel00 = psimd_splat_f32(kernel[0]);
	if (input_channels_subblock_size > 1) {
		vkernel01 = psimd_splat_f32(kernel[1]);
	}
	if (output_channels_subblock_size > 1) {
		kernel += input_channels;
		vkernel10 = psimd_splat_f32(kernel[0]);
		if (input_channels_subblock_size > 1) {
			vkernel11 = psimd_splat_f32(kernel[1]);
		}
		if (output_channels_subblock_size > 2) {
			kernel += input_channels;
			vkernel20 = psimd_splat_f32(kernel[0]);
			if (input_channels_subblock_size > 1) {
				vkernel21 = psimd_splat_f32(kernel[1]);
			}
			if (output_channels_subblock_size > 3) {
				kernel += input_channels;
				vkernel30 = psimd_splat_f32(kernel[0]);
				if (input_channels_subblock_size > 1) {
					vkernel31 = psimd_splat_f32(kernel[1]);
				}
			}
		}
	}

	float* output0 = output;
	float* output1 = output0 + image_size;
	float* output2 = output1 + image_size;
	float* output3 = output2 + image_size;
	while (image_size >= 4) {
		psimd_f32 voutput0, voutput1, voutput2, voutput3;
		voutput0 = psimd_load_f32(output0);
		if (output_channels_subblock_size > 1) {
			voutput1 = psimd_load_f32(output1);
			if (output_channels_subblock_size > 2) {
				voutput2 = psimd_load_f32(output2);
				if (output_channels_subblock_size > 3) {
					voutput3 = psimd_load_f32(output3);
				}
			}
		}

		const psimd_f32 vinput0 = psimd_load_f32(input0);
		input0 += 4;
		voutput0 += vkernel00 * vinput0;
		voutput1 += vkernel10 * vinput0;
		voutput2 += vkernel20 * vinput0;
		voutput3 += vkernel30 * vinput0;

		if (input_channels_subblock_size > 1) {
			const psimd_f32 vinput1 = psimd_load_f32(input1);
			input1 += 4;
			voutput0 += vkernel01 * vinput1;
			voutput1 += vkernel11 * vinput1;
			voutput2 += vkernel21 * vinput1;
			voutput3 += vkernel31 * vinput1;
		}

		psimd_store_f32(output0, voutput0);
		output0 += 4;
		if (output_channels_subblock_size > 1) {
			psimd_store_f32(output1, voutput1);
			output1 += 4;
			if (output_channels_subblock_size > 2) {
				psimd_store_f32(output2, voutput2);
				output2 += 4;
				if (output_channels_subblock_size > 3) {
					psimd_store_f32(output3, voutput3);
					output3 += 4;
				}
			}
		}

		image_size -= 4;
	}
	if (image_size != 0) {
		const psimd_s32 vmask = psimd_load_s32(&mask_array[image_size]);

		psimd_f32 voutput0, voutput1, voutput2, voutput3;
		output0 += image_size - 4;
		voutput0 = psimd_load_f32(output0);
		if (output_channels_subblock_size > 1) {
			output1 += image_size - 4;
			voutput1 = psimd_load_f32(output1);
			if (output_channels_subblock_size > 2) {
				output2 += image_size - 4;
				voutput2 = psimd_load_f32(output2);
				if (output_channels_subblock_size > 3) {
					output3 += image_size - 4;
					voutput3 = psimd_load_f32(output3);
				}
			}
		}

		const psimd_f32 vinput0 = psimd_andmask_f32(vmask, psimd_load_f32(&input0[image_size - 4]));
		voutput0 += vkernel00 * vinput0;
		voutput1 += vkernel10 * vinput0;
		voutput2 += vkernel20 * vinput0;
		voutput3 += vkernel30 * vinput0;

		if (input_channels_subblock_size > 1) {
			const psimd_f32 vinput1 = psimd_andmask_f32(vmask, psimd_load_f32(&input1[image_size - 4]));
			voutput0 += vkernel01 * vinput1;
			voutput1 += vkernel11 * vinput1;
			voutput2 += vkernel21 * vinput1;
			voutput3 += vkernel31 * vinput1;
		}

		psimd_store_f32(output0, voutput0);
		if (output_channels_subblock_size > 1) {
			psimd_store_f32(output1, voutput1);
			if (output_channels_subblock_size > 2) {
				psimd_store_f32(output2, voutput2);
				if (output_channels_subblock_size > 3) {
					psimd_store_f32(output3, voutput3);
				}
			}
		}
	}
}
