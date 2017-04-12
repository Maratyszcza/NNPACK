#include <stdint.h>
#include <stddef.h>

#include <nnpack/activations.h>


void nnp_relu__scalar(
	const float input[restrict static 1],
	float output[restrict static 1],
	size_t length,
	float negative_slope)
{
	while (length >= 4) {
		const float data0 = input[0];
		const float data1 = input[1];
		const float data2 = input[2];
		const float data3 = input[3];
		input += 4;

		output[0] = relu(data0, negative_slope);
		output[1] = relu(data1, negative_slope);
		output[2] = relu(data2, negative_slope);
		output[3] = relu(data3, negative_slope);
		output += 4;

		length -= 4;
	}
	while (length != 0) {
		*output++ = relu(*input++, negative_slope);
		length -= 1;
	}
}

void nnp_inplace_relu__scalar(
	float data[restrict static 1],
	size_t length,
	float negative_slope)
{
	while (length >= 4) {
		const float data0 = data[0];
		const float data1 = data[1];
		const float data2 = data[2];
		const float data3 = data[3];

		data[0] = relu(data0, negative_slope);
		data[1] = relu(data1, negative_slope);
		data[2] = relu(data2, negative_slope);
		data[3] = relu(data3, negative_slope);
		data += 4;

		length -= 4;
	}
	while (length != 0) {
		*data = relu(*data, negative_slope);

		data += 1;
		length -= 1;
	}
}

void nnp_grad_relu__scalar(
	const float output_gradient[restrict static 4],
	const float input[restrict static 4],
	float input_gradient[restrict static 4],
	size_t length,
	float negative_slope)
{
	while (length >= 4) {
		const float data0 = input[0];
		const float data1 = input[1];
		const float data2 = input[2];
		const float data3 = input[3];
		input += 4;

		const float grad0 = output_gradient[0];
		const float grad1 = output_gradient[1];
		const float grad2 = output_gradient[2];
		const float grad3 = output_gradient[3];
		output_gradient += 4;

		input_gradient[0] = grad_relu(grad0, data0, negative_slope);
		input_gradient[1] = grad_relu(grad1, data1, negative_slope);
		input_gradient[2] = grad_relu(grad2, data2, negative_slope);
		input_gradient[3] = grad_relu(grad3, data3, negative_slope);
		input_gradient += 4;

		length -= 4;
	}
	while (length != 0) {
		*input_gradient++ = grad_relu(*output_gradient++, *input++, negative_slope);
		length -= 1;
	}
}
