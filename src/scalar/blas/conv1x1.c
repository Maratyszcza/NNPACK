#include <stddef.h>
#include <stdint.h>


void nnp_conv1x1_only_2x4__scalar(
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float* pinput0 = input;
	const float* pinput1 = input + image_size;

	const float kernel00 = kernel[0];
	const float kernel01 = kernel[1];
	kernel += input_channels;
	const float kernel10 = kernel[0];
	const float kernel11 = kernel[1];
	kernel += input_channels;
	const float kernel20 = kernel[0];
	const float kernel21 = kernel[1];
	kernel += input_channels;
	const float kernel30 = kernel[0];
	const float kernel31 = kernel[1];

	float* poutput0 = output;
	float* poutput1 = poutput0 + image_size;
	float* poutput2 = poutput1 + image_size;
	float* poutput3 = poutput2 + image_size;
	do {
		float output0 = *poutput0;
		float output1 = *poutput1;
		float output2 = *poutput2;
		float output3 = *poutput3;

		const float input0 = *pinput0++;
		output0 += kernel00 * input0;
		output1 += kernel10 * input0;
		output2 += kernel20 * input0;
		output3 += kernel30 * input0;

		const float input1 = *pinput1++;
		output0 += kernel01 * input1;
		*poutput0++ = output0;
		output1 += kernel11 * input1;
		*poutput1++ = output1;
		output2 += kernel21 * input1;
		*poutput2++ = output2;
		output3 += kernel31 * input1;
		*poutput3++ = output3;
	} while (--image_size != 0);
}

void nnp_conv1x1_upto_2x4__scalar(
	uint32_t input_channels_subblock_size,
	uint32_t output_channels_subblock_size,
	size_t input_channels,
	size_t image_size,
	const float* input,
	const float* kernel,
	float* output)
{
	const float* pinput0 = input;
	const float* pinput1 = input + image_size;

	float kernel00, kernel01, kernel10, kernel11, kernel20, kernel21, kernel30, kernel31;
	kernel00 = kernel[0];
	if (input_channels_subblock_size > 1) {
		kernel01 = kernel[1];
	}
	if (output_channels_subblock_size > 1) {
		kernel += input_channels;
		kernel10 = kernel[0];
		if (input_channels_subblock_size > 1) {
			kernel11 = kernel[1];
		}
		if (output_channels_subblock_size > 2) {
			kernel += input_channels;
			kernel20 = kernel[0];
			if (input_channels_subblock_size > 1) {
				kernel21 = kernel[1];
			}
			if (output_channels_subblock_size > 3) {
				kernel += input_channels;
				kernel30 = kernel[0];
				if (input_channels_subblock_size > 1) {
					kernel31 = kernel[1];
				}
			}
		}
	}

	float* poutput0 = output;
	float* poutput1 = poutput0 + image_size;
	float* poutput2 = poutput1 + image_size;
	float* poutput3 = poutput2 + image_size;
	do {
		float output0, output1, output2, output3;
		output0 = *poutput0;
		if (output_channels_subblock_size > 1) {
			output1 = *poutput1;
			if (output_channels_subblock_size > 2) {
				output2 = *poutput2;
				if (output_channels_subblock_size > 3) {
					output3 = *poutput3;
				}
			}
		}

		const float input0 = *pinput0++;
		output0 += kernel00 * input0;
		output1 += kernel10 * input0;
		output2 += kernel20 * input0;
		output3 += kernel30 * input0;

		if (input_channels_subblock_size > 1) {
			const float input1 = *pinput1++;
			output0 += kernel01 * input1;
			output1 += kernel11 * input1;
			output2 += kernel21 * input1;
			output3 += kernel31 * input1;
		}

		*poutput0++ = output0;
		if (output_channels_subblock_size > 1) {
			*poutput1++ = output1;
			if (output_channels_subblock_size > 2) {
				*poutput2++ = output2;
				if (output_channels_subblock_size > 3) {
					*poutput3++ = output3;
				}
			}
		}
	} while (--image_size != 0);
}
