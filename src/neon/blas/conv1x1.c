#include <stddef.h>
#include <stdint.h>

#include <nnpack/macros.h>
#include <nnpack/arm_neon.h>


void nnp_conv1x1_only_4x4__neon(
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float* input0 = input;
	const float* input1 = input0 + image_size;
	const float* input2 = input1 + image_size;
	const float* input3 = input2 + image_size;

	const float32x4_t vkernel0x = vld1q_f32(kernel);
	kernel += input_channels;
	const float32x4_t vkernel1x = vld1q_f32(kernel);
	kernel += input_channels;
	const float32x4_t vkernel2x = vld1q_f32(kernel);
	kernel += input_channels;
	const float32x4_t vkernel3x = vld1q_f32(kernel);

	float* output0 = output;
	float* output1 = output0 + image_size;
	float* output2 = output1 + image_size;
	float* output3 = output2 + image_size;
	while (image_size >= 4) {
		float32x4_t voutput0 = vld1q_f32(output0);
		float32x4_t voutput1 = vld1q_f32(output1);
		float32x4_t voutput2 = vld1q_f32(output2);
		float32x4_t voutput3 = vld1q_f32(output3);

		const float32x4_t vinput0 = vld1q_f32(input0); input0 += 4;
		voutput0 = vmuladdq_lane0_f32(voutput0, vinput0, vget_low_f32(vkernel0x));
		voutput1 = vmuladdq_lane0_f32(voutput1, vinput0, vget_low_f32(vkernel1x));
		voutput2 = vmuladdq_lane0_f32(voutput2, vinput0, vget_low_f32(vkernel2x));
		voutput3 = vmuladdq_lane0_f32(voutput3, vinput0, vget_low_f32(vkernel3x));

		const float32x4_t vinput1 = vld1q_f32(input1); input1 += 4;
		voutput0 = vmuladdq_lane1_f32(voutput0, vinput1, vget_low_f32(vkernel0x));
		voutput1 = vmuladdq_lane1_f32(voutput1, vinput1, vget_low_f32(vkernel1x));
		voutput2 = vmuladdq_lane1_f32(voutput2, vinput1, vget_low_f32(vkernel2x));
		voutput3 = vmuladdq_lane1_f32(voutput3, vinput1, vget_low_f32(vkernel3x));

		const float32x4_t vinput2 = vld1q_f32(input2); input2 += 4;
		voutput0 = vmuladdq_lane0_f32(voutput0, vinput2, vget_high_f32(vkernel0x));
		voutput1 = vmuladdq_lane0_f32(voutput1, vinput2, vget_high_f32(vkernel1x));
		voutput2 = vmuladdq_lane0_f32(voutput2, vinput2, vget_high_f32(vkernel2x));
		voutput3 = vmuladdq_lane0_f32(voutput3, vinput2, vget_high_f32(vkernel3x));

		const float32x4_t vinput3 = vld1q_f32(input3); input3 += 4;
		voutput0 = vmuladdq_lane1_f32(voutput0, vinput3, vget_high_f32(vkernel0x));
		voutput1 = vmuladdq_lane1_f32(voutput1, vinput3, vget_high_f32(vkernel1x));
		voutput2 = vmuladdq_lane1_f32(voutput2, vinput3, vget_high_f32(vkernel2x));
		voutput3 = vmuladdq_lane1_f32(voutput3, vinput3, vget_high_f32(vkernel3x));

		vst1q_f32(output0, voutput0); output0 += 4;
		vst1q_f32(output1, voutput1); output1 += 4;
		vst1q_f32(output2, voutput2); output2 += 4;
		vst1q_f32(output3, voutput3); output3 += 4;

		image_size -= 4;
	}
	if (image_size >= 2) {
		float32x2_t voutput0 = vld1_f32(output0);
		float32x2_t voutput1 = vld1_f32(output1);
		float32x2_t voutput2 = vld1_f32(output2);
		float32x2_t voutput3 = vld1_f32(output3);

		const float32x2_t vinput0 = vld1_f32(input0); input0 += 2;
		voutput0 = vmuladd_lane0_f32(voutput0, vinput0, vget_low_f32(vkernel0x));
		voutput1 = vmuladd_lane0_f32(voutput1, vinput0, vget_low_f32(vkernel1x));
		voutput2 = vmuladd_lane0_f32(voutput2, vinput0, vget_low_f32(vkernel2x));
		voutput3 = vmuladd_lane0_f32(voutput3, vinput0, vget_low_f32(vkernel3x));

		const float32x2_t vinput1 = vld1_f32(input1); input1 += 2;
		voutput0 = vmuladd_lane1_f32(voutput0, vinput1, vget_low_f32(vkernel0x));
		voutput1 = vmuladd_lane1_f32(voutput1, vinput1, vget_low_f32(vkernel1x));
		voutput2 = vmuladd_lane1_f32(voutput2, vinput1, vget_low_f32(vkernel2x));
		voutput3 = vmuladd_lane1_f32(voutput3, vinput1, vget_low_f32(vkernel3x));

		const float32x2_t vinput2 = vld1_f32(input2); input2 += 2;
		voutput0 = vmuladd_lane0_f32(voutput0, vinput2, vget_high_f32(vkernel0x));
		voutput1 = vmuladd_lane0_f32(voutput1, vinput2, vget_high_f32(vkernel1x));
		voutput2 = vmuladd_lane0_f32(voutput2, vinput2, vget_high_f32(vkernel2x));
		voutput3 = vmuladd_lane0_f32(voutput3, vinput2, vget_high_f32(vkernel3x));

		const float32x2_t vinput3 = vld1_f32(input3); input3 += 2;
		voutput0 = vmuladd_lane1_f32(voutput0, vinput3, vget_high_f32(vkernel0x));
		voutput1 = vmuladd_lane1_f32(voutput1, vinput3, vget_high_f32(vkernel1x));
		voutput2 = vmuladd_lane1_f32(voutput2, vinput3, vget_high_f32(vkernel2x));
		voutput3 = vmuladd_lane1_f32(voutput3, vinput3, vget_high_f32(vkernel3x));

		vst1_f32(output0, voutput0); output0 += 2;
		vst1_f32(output1, voutput1); output1 += 2;
		vst1_f32(output2, voutput2); output2 += 2;
		vst1_f32(output3, voutput3); output3 += 2;

		image_size -= 2;
	}
	if (image_size != 0) {
		float32x2_t voutput0 = vld1_dup_f32(output0);
		float32x2_t voutput1 = vld1_dup_f32(output1);
		float32x2_t voutput2 = vld1_dup_f32(output2);
		float32x2_t voutput3 = vld1_dup_f32(output3);

		const float32x2_t vinput0 = vld1_dup_f32(input0);
		voutput0 = vmuladd_lane0_f32(voutput0, vinput0, vget_low_f32(vkernel0x));
		voutput1 = vmuladd_lane0_f32(voutput1, vinput0, vget_low_f32(vkernel1x));
		voutput2 = vmuladd_lane0_f32(voutput2, vinput0, vget_low_f32(vkernel2x));
		voutput3 = vmuladd_lane0_f32(voutput3, vinput0, vget_low_f32(vkernel3x));

		const float32x2_t vinput1 = vld1_dup_f32(input1);
		voutput0 = vmuladd_lane1_f32(voutput0, vinput1, vget_low_f32(vkernel0x));
		voutput1 = vmuladd_lane1_f32(voutput1, vinput1, vget_low_f32(vkernel1x));
		voutput2 = vmuladd_lane1_f32(voutput2, vinput1, vget_low_f32(vkernel2x));
		voutput3 = vmuladd_lane1_f32(voutput3, vinput1, vget_low_f32(vkernel3x));

		const float32x2_t vinput2 = vld1_dup_f32(input2);
		voutput0 = vmuladd_lane0_f32(voutput0, vinput2, vget_high_f32(vkernel0x));
		voutput1 = vmuladd_lane0_f32(voutput1, vinput2, vget_high_f32(vkernel1x));
		voutput2 = vmuladd_lane0_f32(voutput2, vinput2, vget_high_f32(vkernel2x));
		voutput3 = vmuladd_lane0_f32(voutput3, vinput2, vget_high_f32(vkernel3x));

		const float32x2_t vinput3 = vld1_dup_f32(input3);
		voutput0 = vmuladd_lane1_f32(voutput0, vinput3, vget_high_f32(vkernel0x));
		voutput1 = vmuladd_lane1_f32(voutput1, vinput3, vget_high_f32(vkernel1x));
		voutput2 = vmuladd_lane1_f32(voutput2, vinput3, vget_high_f32(vkernel2x));
		voutput3 = vmuladd_lane1_f32(voutput3, vinput3, vget_high_f32(vkernel3x));

		vst1_lane_f32(output0, voutput0, 0);
		vst1_lane_f32(output1, voutput1, 0);
		vst1_lane_f32(output2, voutput2, 0);
		vst1_lane_f32(output3, voutput3, 0);
	}
}

void nnp_conv1x1_upto_4x4__neon(
	uint32_t input_channels_subblock_size,
	uint32_t output_channels_subblock_size,
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float*restrict input0 = input;
	const float*restrict input1 = input_channels_subblock_size > 1 ? input0 + image_size : input0;
	const float*restrict input2 = input_channels_subblock_size > 2 ? input1 + image_size : input1;
	const float*restrict input3 = input_channels_subblock_size > 3 ? input2 + image_size : input2;

	const float*restrict kernel0 = kernel;
	const float*restrict kernel1 = output_channels_subblock_size > 1 ? kernel0 + input_channels : kernel0;
	const float*restrict kernel2 = output_channels_subblock_size > 2 ? kernel1 + input_channels : kernel1;
	const float*restrict kernel3 = output_channels_subblock_size > 3 ? kernel2 + input_channels : kernel2;

	float32x4_t vkernel0x = vld1q_dup_f32(kernel0);
	float32x4_t vkernel1x = vld1q_dup_f32(kernel1);
	float32x4_t vkernel2x = vld1q_dup_f32(kernel2);
	float32x4_t vkernel3x = vld1q_dup_f32(kernel3);
	if (input_channels_subblock_size > 1) {
		vkernel0x = vld1q_lane_f32(kernel0 + 1, vkernel0x, 1);
		vkernel1x = vld1q_lane_f32(kernel1 + 1, vkernel1x, 1);
		vkernel2x = vld1q_lane_f32(kernel2 + 1, vkernel2x, 1);
		vkernel3x = vld1q_lane_f32(kernel3 + 1, vkernel3x, 1);
		if (input_channels_subblock_size > 2) {
			vkernel0x = vld1q_lane_f32(kernel0 + 2, vkernel0x, 2);
			vkernel1x = vld1q_lane_f32(kernel1 + 2, vkernel1x, 2);
			vkernel2x = vld1q_lane_f32(kernel2 + 2, vkernel2x, 2);
			vkernel3x = vld1q_lane_f32(kernel3 + 2, vkernel3x, 2);
			if (input_channels_subblock_size > 3) {
				vkernel0x = vld1q_lane_f32(kernel0 + 3, vkernel0x, 3);
				vkernel1x = vld1q_lane_f32(kernel1 + 3, vkernel1x, 3);
				vkernel2x = vld1q_lane_f32(kernel2 + 3, vkernel2x, 3);
				vkernel3x = vld1q_lane_f32(kernel3 + 3, vkernel3x, 3);
			}
		}
	}

	float*restrict output0 = output;
	float*restrict output1 = output_channels_subblock_size > 1 ? output0 + image_size : output0;
	float*restrict output2 = output_channels_subblock_size > 2 ? output1 + image_size : output1;
	float*restrict output3 = output_channels_subblock_size > 3 ? output2 + image_size : output2;
	while (image_size >= 4) {
		float32x4_t voutput0 = vld1q_f32(output0);
		float32x4_t voutput1 = vld1q_f32(output1);
		float32x4_t voutput2 = vld1q_f32(output2);
		float32x4_t voutput3 = vld1q_f32(output3);

		const float32x4_t vinput0 = vld1q_f32(input0); input0 += 4;
		voutput0 = vmuladdq_lane0_f32(voutput0, vinput0, vget_low_f32(vkernel0x));
		voutput1 = vmuladdq_lane0_f32(voutput1, vinput0, vget_low_f32(vkernel1x));
		voutput2 = vmuladdq_lane0_f32(voutput2, vinput0, vget_low_f32(vkernel2x));
		voutput3 = vmuladdq_lane0_f32(voutput3, vinput0, vget_low_f32(vkernel3x));

		if (input_channels_subblock_size > 1) {
			const float32x4_t vinput1 = vld1q_f32(input1); input1 += 4;
			voutput0 = vmuladdq_lane1_f32(voutput0, vinput1, vget_low_f32(vkernel0x));
			voutput1 = vmuladdq_lane1_f32(voutput1, vinput1, vget_low_f32(vkernel1x));
			voutput2 = vmuladdq_lane1_f32(voutput2, vinput1, vget_low_f32(vkernel2x));
			voutput3 = vmuladdq_lane1_f32(voutput3, vinput1, vget_low_f32(vkernel3x));

			if (input_channels_subblock_size > 2) {
				const float32x4_t vinput2 = vld1q_f32(input2); input2 += 4;
				voutput0 = vmuladdq_lane0_f32(voutput0, vinput2, vget_high_f32(vkernel0x));
				voutput1 = vmuladdq_lane0_f32(voutput1, vinput2, vget_high_f32(vkernel1x));
				voutput2 = vmuladdq_lane0_f32(voutput2, vinput2, vget_high_f32(vkernel2x));
				voutput3 = vmuladdq_lane0_f32(voutput3, vinput2, vget_high_f32(vkernel3x));

				if (input_channels_subblock_size > 3) {
					const float32x4_t vinput3 = vld1q_f32(input3); input3 += 4;
					voutput0 = vmuladdq_lane1_f32(voutput0, vinput3, vget_high_f32(vkernel0x));
					voutput1 = vmuladdq_lane1_f32(voutput1, vinput3, vget_high_f32(vkernel1x));
					voutput2 = vmuladdq_lane1_f32(voutput2, vinput3, vget_high_f32(vkernel2x));
					voutput3 = vmuladdq_lane1_f32(voutput3, vinput3, vget_high_f32(vkernel3x));
				}
			}
		}

		vst1q_f32(output0, voutput0); output0 += 4;
		if (output_channels_subblock_size > 1) {
			vst1q_f32(output1, voutput1); output1 += 4;
			if (output_channels_subblock_size > 2) {
				vst1q_f32(output2, voutput2); output2 += 4;
				if (output_channels_subblock_size > 3) {
					vst1q_f32(output3, voutput3); output3 += 4;
				}
			}
		}

		image_size -= 4;
	}
	if (image_size >= 2) {
		float32x2_t voutput0 = vld1_f32(output0);
		float32x2_t voutput1 = vld1_f32(output1);
		float32x2_t voutput2 = vld1_f32(output2);
		float32x2_t voutput3 = vld1_f32(output3);

		const float32x2_t vinput0 = vld1_f32(input0); input0 += 2;
		voutput0 = vmuladd_lane0_f32(voutput0, vinput0, vget_low_f32(vkernel0x));
		voutput1 = vmuladd_lane0_f32(voutput1, vinput0, vget_low_f32(vkernel1x));
		voutput2 = vmuladd_lane0_f32(voutput2, vinput0, vget_low_f32(vkernel2x));
		voutput3 = vmuladd_lane0_f32(voutput3, vinput0, vget_low_f32(vkernel3x));

		if (input_channels_subblock_size > 1) {
			const float32x2_t vinput1 = vld1_f32(input1); input1 += 2;
			voutput0 = vmuladd_lane1_f32(voutput0, vinput1, vget_low_f32(vkernel0x));
			voutput1 = vmuladd_lane1_f32(voutput1, vinput1, vget_low_f32(vkernel1x));
			voutput2 = vmuladd_lane1_f32(voutput2, vinput1, vget_low_f32(vkernel2x));
			voutput3 = vmuladd_lane1_f32(voutput3, vinput1, vget_low_f32(vkernel3x));

			if (input_channels_subblock_size > 2) {
				const float32x2_t vinput2 = vld1_f32(input2); input2 += 2;
				voutput0 = vmuladd_lane0_f32(voutput0, vinput2, vget_high_f32(vkernel0x));
				voutput1 = vmuladd_lane0_f32(voutput1, vinput2, vget_high_f32(vkernel1x));
				voutput2 = vmuladd_lane0_f32(voutput2, vinput2, vget_high_f32(vkernel2x));
				voutput3 = vmuladd_lane0_f32(voutput3, vinput2, vget_high_f32(vkernel3x));

				if (input_channels_subblock_size > 3) {
					const float32x2_t vinput3 = vld1_f32(input3); input3 += 2;
					voutput0 = vmuladd_lane1_f32(voutput0, vinput3, vget_high_f32(vkernel0x));
					voutput1 = vmuladd_lane1_f32(voutput1, vinput3, vget_high_f32(vkernel1x));
					voutput2 = vmuladd_lane1_f32(voutput2, vinput3, vget_high_f32(vkernel2x));
					voutput3 = vmuladd_lane1_f32(voutput3, vinput3, vget_high_f32(vkernel3x));
				}
			}
		}

		vst1_f32(output0, voutput0); output0 += 2;
		if (output_channels_subblock_size > 1) {
			vst1_f32(output1, voutput1); output1 += 2;
			if (output_channels_subblock_size > 2) {
				vst1_f32(output2, voutput2); output2 += 2;
				if (output_channels_subblock_size > 3) {
					vst1_f32(output3, voutput3); output3 += 2;
				}
			}
		}

		image_size -= 2;
	}
	if (image_size != 0) {
		float32x2_t voutput0 = vld1_dup_f32(output0);
		float32x2_t voutput1 = vld1_dup_f32(output1);
		float32x2_t voutput2 = vld1_dup_f32(output2);
		float32x2_t voutput3 = vld1_dup_f32(output3);

		const float32x2_t vinput0 = vld1_dup_f32(input0);
		voutput0 = vmuladd_lane0_f32(voutput0, vinput0, vget_low_f32(vkernel0x));
		voutput1 = vmuladd_lane0_f32(voutput1, vinput0, vget_low_f32(vkernel1x));
		voutput2 = vmuladd_lane0_f32(voutput2, vinput0, vget_low_f32(vkernel2x));
		voutput3 = vmuladd_lane0_f32(voutput3, vinput0, vget_low_f32(vkernel3x));

		if (input_channels_subblock_size > 1) {
			const float32x2_t vinput1 = vld1_dup_f32(input1);
			voutput0 = vmuladd_lane1_f32(voutput0, vinput1, vget_low_f32(vkernel0x));
			voutput1 = vmuladd_lane1_f32(voutput1, vinput1, vget_low_f32(vkernel1x));
			voutput2 = vmuladd_lane1_f32(voutput2, vinput1, vget_low_f32(vkernel2x));
			voutput3 = vmuladd_lane1_f32(voutput3, vinput1, vget_low_f32(vkernel3x));

			if (input_channels_subblock_size > 2) {
				const float32x2_t vinput2 = vld1_dup_f32(input2);
				voutput0 = vmuladd_lane0_f32(voutput0, vinput2, vget_high_f32(vkernel0x));
				voutput1 = vmuladd_lane0_f32(voutput1, vinput2, vget_high_f32(vkernel1x));
				voutput2 = vmuladd_lane0_f32(voutput2, vinput2, vget_high_f32(vkernel2x));
				voutput3 = vmuladd_lane0_f32(voutput3, vinput2, vget_high_f32(vkernel3x));

				if (input_channels_subblock_size > 3) {
					const float32x2_t vinput3 = vld1_dup_f32(input3);
					voutput0 = vmuladd_lane1_f32(voutput0, vinput3, vget_high_f32(vkernel0x));
					voutput1 = vmuladd_lane1_f32(voutput1, vinput3, vget_high_f32(vkernel1x));
					voutput2 = vmuladd_lane1_f32(voutput2, vinput3, vget_high_f32(vkernel2x));
					voutput3 = vmuladd_lane1_f32(voutput3, vinput3, vget_high_f32(vkernel3x));
				}
			}
		}

		vst1_lane_f32(output0, voutput0, 0);
		if (output_channels_subblock_size > 1) {
			vst1_lane_f32(output1, voutput1, 0);
			if (output_channels_subblock_size > 2) {
				vst1_lane_f32(output2, voutput2, 0);
				if (output_channels_subblock_size > 3) {
					vst1_lane_f32(output3, voutput3, 0);
				}
			}
		}
	}
}
