#pragma once

#include <nnpack.h>
#include <nnpack/utils.h>
#include <nnpack/hwinfo.h>

static inline enum nnp_status validate_convolution_arguments(
	size_t batch_size, size_t input_channels, size_t output_channels,
	struct nnp_size input_size, struct nnp_padding input_padding,
	struct nnp_size kernel_size, struct nnp_size output_subsampling)
{
	if (!nnp_hwinfo.initialized) {
		return nnp_status_uninitialized;
	}

	if (!nnp_hwinfo.supported) {
		return nnp_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return nnp_status_invalid_batch_size;
	}

	if (input_channels == 0) {
		return nnp_status_invalid_input_channels;
	}

	if (output_channels == 0) {
		return nnp_status_invalid_output_channels;
	}

	if (min(input_size.height, input_size.width) == 0) {
		return nnp_status_invalid_input_size;
	}

	if (max(input_padding.top, input_padding.bottom) >= kernel_size.height) {
		return nnp_status_invalid_input_padding;
	}

	if (max(input_padding.left, input_padding.right) >= kernel_size.width) {
		return nnp_status_invalid_input_padding;
	}

	if (min(kernel_size.height, kernel_size.width) == 0) {
		return nnp_status_invalid_kernel_size;
	}

	if (min(output_subsampling.height, output_subsampling.width) == 0) {
		return nnp_status_invalid_output_subsampling;
	}

	return nnp_status_success;
}

static inline enum nnp_status validate_fully_connected_arguments(
	size_t batch_size, size_t input_channels, size_t output_channels)
{
	if (!nnp_hwinfo.initialized) {
		return nnp_status_uninitialized;
	}

	if (!nnp_hwinfo.supported) {
		return nnp_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return nnp_status_invalid_batch_size;
	}

	if (input_channels == 0) {
		return nnp_status_invalid_input_channels;
	}

	if (output_channels == 0) {
		return nnp_status_invalid_output_channels;
	}

	return nnp_status_success;
}

static inline enum nnp_status validate_pooling_arguments(
	size_t batch_size, size_t channels,
	struct nnp_size input_size, struct nnp_padding input_padding,
	struct nnp_size pooling_size, struct nnp_size pooling_stride)
{
	if (!nnp_hwinfo.initialized) {
		return nnp_status_uninitialized;
	}

	if (!nnp_hwinfo.supported) {
		return nnp_status_unsupported_hardware;
	}

	if (min(input_size.height, input_size.width) == 0) {
		return nnp_status_invalid_input_size;
	}

	if (min(pooling_size.height, pooling_size.width) == 0) {
		return nnp_status_invalid_pooling_size;
	}

	if (min(pooling_stride.height, pooling_stride.width) == 0) {
		return nnp_status_invalid_pooling_stride;
	}

	if ((pooling_size.height < pooling_stride.height) || (pooling_size.width < pooling_size.width)) {
		return nnp_status_invalid_pooling_stride;
	}

	if (max(input_padding.top, input_padding.bottom) >= pooling_size.height) {
		return nnp_status_invalid_input_padding;
	}

	if (max(input_padding.left, input_padding.right) >= pooling_size.width) {
		return nnp_status_invalid_input_padding;
	}

	return nnp_status_success;
}

static inline enum nnp_status validate_relu_arguments(
	size_t batch_size, size_t channels)
{
	if (!nnp_hwinfo.initialized) {
		return nnp_status_uninitialized;
	}

	if (!nnp_hwinfo.supported) {
		return nnp_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return nnp_status_invalid_batch_size;
	}

	if (channels == 0) {
		return nnp_status_invalid_channels;
	}

	return nnp_status_success;
}

static inline enum nnp_status validate_softmax_arguments(
	size_t batch_size, size_t channels)
{
	if (!nnp_hwinfo.initialized) {
		return nnp_status_uninitialized;
	}

	if (!nnp_hwinfo.supported) {
		return nnp_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return nnp_status_invalid_batch_size;
	}

	if (channels == 0) {
		return nnp_status_invalid_channels;
	}

	return nnp_status_success;
}
