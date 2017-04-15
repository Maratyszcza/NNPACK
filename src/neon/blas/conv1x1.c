#include <stddef.h>
#include <stdint.h>

#include <nnpack/macros.h>
#include <nnpack/arm_neon.h>


static NNP_SIMD_ALIGN int32_t mask_array[8] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};


void nnp_conv1x1_only_2x4__neon(
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float* input0 = input;
	const float* input1 = input + image_size;

	const float32x2_t vkernel0x = vld1_f32(kernel);
	kernel += input_channels;
	const float32x2_t vkernel1x = vld1_f32(kernel);
	kernel += input_channels;
	const float32x2_t vkernel2x = vld1_f32(kernel);
	kernel += input_channels;
	const float32x2_t vkernel3x = vld1_f32(kernel);

	float* output0 = output;
	float* output1 = output0 + image_size;
	float* output2 = output1 + image_size;
	float* output3 = output2 + image_size;
	while (image_size >= 4) {
		float32x4_t voutput0 = vld1q_f32(output0);
		float32x4_t voutput1 = vld1q_f32(output1);
		float32x4_t voutput2 = vld1q_f32(output2);
		float32x4_t voutput3 = vld1q_f32(output3);

		const float32x4_t vinput0 = vld1q_f32(input0);
		input0 += 4;
		#if defined(__aarch64__)
			voutput0 = vfmaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vfmaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vfmaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vfmaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#else
			voutput0 = vmlaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vmlaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vmlaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vmlaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#endif

		const float32x4_t vinput1 = vld1q_f32(input1);
		input1 += 4;
		#if defined(__aarch64__)
			voutput0 = vfmaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
			voutput1 = vfmaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
			voutput2 = vfmaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
			voutput3 = vfmaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
		#else
			voutput0 = vmlaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
			voutput1 = vmlaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
			voutput2 = vmlaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
			voutput3 = vmlaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
		#endif

		vst1q_f32(output0, voutput0);
		output0 += 4;
		vst1q_f32(output1, voutput1);
		output1 += 4;
		vst1q_f32(output2, voutput2);
		output2 += 4;
		vst1q_f32(output3, voutput3);
		output3 += 4;

		image_size -= 4;
	}
	if (image_size != 0) {
		const int32x4_t vmask = vld1q_s32(&mask_array[image_size]);

		output0 = output0 + image_size - 4;
		float32x4_t voutput0 = vld1q_f32(output0);
		output1 = output1 + image_size - 4;
		float32x4_t voutput1 = vld1q_f32(output1);
		output2 = output2 + image_size - 4;
		float32x4_t voutput2 = vld1q_f32(output2);
		output3 = output3 + image_size - 4;
		float32x4_t voutput3 = vld1q_f32(output3);

		const float32x4_t vinput0 = vreinterpretq_f32_s32(
			vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input0[image_size - 4]))));
		#if defined(__aarch64__)
			voutput0 = vfmaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vfmaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vfmaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vfmaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#else
			voutput0 = vmlaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vmlaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vmlaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vmlaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#endif

		const float32x4_t vinput1 = vreinterpretq_f32_s32(
			vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input1[image_size - 4]))));
		#if defined(__aarch64__)
			voutput0 = vfmaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
			voutput1 = vfmaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
			voutput2 = vfmaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
			voutput3 = vfmaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
		#else
			voutput0 = vmlaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
			voutput1 = vmlaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
			voutput2 = vmlaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
			voutput3 = vmlaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
		#endif

		vst1q_f32(output0, voutput0);
		vst1q_f32(output1, voutput1);
		vst1q_f32(output2, voutput2);
		vst1q_f32(output3, voutput3);
	}
}

void nnp_conv1x1_upto_2x4__neon(
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

	float32x2_t vkernel0x, vkernel1x, vkernel2x, vkernel3x;
	vkernel0x = vld1_dup_f32(&kernel[0]);
	if (input_channels_subblock_size > 1) {
		vkernel0x = vld1_lane_f32(&kernel[1], vkernel0x, 1);
	}
	if (output_channels_subblock_size > 1) {
		kernel += input_channels;
		vkernel1x = vld1_dup_f32(&kernel[0]);
		if (input_channels_subblock_size > 1) {
			vkernel1x = vld1_lane_f32(&kernel[1], vkernel1x, 1);
		}
		if (output_channels_subblock_size > 2) {
			kernel += input_channels;
			vkernel2x = vld1_dup_f32(&kernel[0]);
			if (input_channels_subblock_size > 1) {
				vkernel2x = vld1_lane_f32(&kernel[1], vkernel2x, 1);
			}
			if (output_channels_subblock_size > 3) {
				kernel += input_channels;
				vkernel3x = vld1_dup_f32(&kernel[0]);
				if (input_channels_subblock_size > 1) {
					vkernel3x = vld1_lane_f32(&kernel[1], vkernel3x, 1);
				}
			}
		}
	}

	float* output0 = output;
	float* output1 = output0 + image_size;
	float* output2 = output1 + image_size;
	float* output3 = output2 + image_size;
	while (image_size >= 4) {
		float32x4_t voutput0, voutput1, voutput2, voutput3;
		voutput0 = vld1q_f32(output0);
		if (output_channels_subblock_size > 1) {
			voutput1 = vld1q_f32(output1);
			if (output_channels_subblock_size > 2) {
				voutput2 = vld1q_f32(output2);
				if (output_channels_subblock_size > 3) {
					voutput3 = vld1q_f32(output3);
				}
			}
		}

		const float32x4_t vinput0 = vld1q_f32(input0);
		input0 += 4;
		#if defined(__aarch64__)
			voutput0 = vfmaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vfmaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vfmaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vfmaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#else
			voutput0 = vmlaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vmlaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vmlaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vmlaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#endif

		if (input_channels_subblock_size > 1) {
			const float32x4_t vinput1 = vld1q_f32(input1);
			input1 += 4;
			#if defined(__aarch64__)
				voutput0 = vfmaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
				voutput1 = vfmaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
				voutput2 = vfmaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
				voutput3 = vfmaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
			#else
				voutput0 = vmlaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
				voutput1 = vmlaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
				voutput2 = vmlaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
				voutput3 = vmlaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
			#endif
		}

		vst1q_f32(output0, voutput0);
		output0 += 4;
		if (output_channels_subblock_size > 1) {
			vst1q_f32(output1, voutput1);
			output1 += 4;
			if (output_channels_subblock_size > 2) {
				vst1q_f32(output2, voutput2);
				output2 += 4;
				if (output_channels_subblock_size > 3) {
					vst1q_f32(output3, voutput3);
					output3 += 4;
				}
			}
		}

		image_size -= 4;
	}
	if (image_size != 0) {
		const int32x4_t vmask = vld1q_s32(&mask_array[image_size]);

		float32x4_t voutput0, voutput1, voutput2, voutput3;
		output0 += image_size - 4;
		voutput0 = vld1q_f32(output0);
		if (output_channels_subblock_size > 1) {
			output1 += image_size - 4;
			voutput1 = vld1q_f32(output1);
			if (output_channels_subblock_size > 2) {
				output2 += image_size - 4;
				voutput2 = vld1q_f32(output2);
				if (output_channels_subblock_size > 3) {
					output3 += image_size - 4;
					voutput3 = vld1q_f32(output3);
				}
			}
		}

		const float32x4_t vinput0 = vreinterpretq_f32_s32(
			vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input0[image_size - 4]))));
		#if defined(__aarch64__)
			voutput0 = vfmaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vfmaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vfmaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vfmaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#else
			voutput0 = vmlaq_lane_f32(voutput0, vinput0, vkernel0x, 0);
			voutput1 = vmlaq_lane_f32(voutput1, vinput0, vkernel1x, 0);
			voutput2 = vmlaq_lane_f32(voutput2, vinput0, vkernel2x, 0);
			voutput3 = vmlaq_lane_f32(voutput3, vinput0, vkernel3x, 0);
		#endif

		if (input_channels_subblock_size > 1) {
			const float32x4_t vinput1 = vreinterpretq_f32_s32(
				vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input1[image_size - 4]))));
			#if defined(__aarch64__)
				voutput0 = vfmaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
				voutput1 = vfmaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
				voutput2 = vfmaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
				voutput3 = vfmaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
			#else
				voutput0 = vmlaq_lane_f32(voutput0, vinput1, vkernel0x, 1);
				voutput1 = vmlaq_lane_f32(voutput1, vinput1, vkernel1x, 1);
				voutput2 = vmlaq_lane_f32(voutput2, vinput1, vkernel2x, 1);
				voutput3 = vmlaq_lane_f32(voutput3, vinput1, vkernel3x, 1);
			#endif
		}

		vst1q_f32(output0, voutput0);
		if (output_channels_subblock_size > 1) {
			vst1q_f32(output1, voutput1);
			if (output_channels_subblock_size > 2) {
				vst1q_f32(output2, voutput2);
				if (output_channels_subblock_size > 3) {
					vst1q_f32(output3, voutput3);
				}
			}
		}
	}
}
