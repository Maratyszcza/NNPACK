#pragma once

#include <nnpack.h>

#include <testers/convolution.h>
#include <testers/fully-connected.h>
#include <testers/pooling.h>

namespace AlexNet {

	/*
	 * AlexNet conv2 layer:
	 *   input channels   = 64
	 *   output channels  = 192
	 *   input size       = 27x27
	 *   implicit padding = 2
	 *   kernel size      = 5x5
	 */
	inline ConvolutionTester conv2() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(64)
			.outputChannels(192)
			.inputSize(27, 27)
			.kernelSize(5, 5)
			.inputPadding(2, 2, 2, 2));
	}

	/*
	 * AlexNet conv3 layer:
	 *   input channels   = 192
	 *   output channels  = 384
	 *   input size       = 13x13
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv3() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(192)
			.outputChannels(384)
			.inputSize(13, 13)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * AlexNet conv4 layer:
	 *   input channels   = 384
	 *   output channels  = 256
	 *   input size       = 13x13
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv4() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(384)
			.outputChannels(256)
			.inputSize(13, 13)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * AlexNet conv5 layer:
	 *   input channels   = 256
	 *   output channels  = 256
	 *   input size       = 13x13
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv5() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(256)
			.outputChannels(256)
			.inputSize(13, 13)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * AlexNet fc6 layer:
	 *   input channels = 12544
	 *   output_channels = 4096
	 */
	inline FullyConnectedTester fc6() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(12544)
			.outputChannels(4096));
	}

	/*
	 * AlexNet fc7 layer:
	 *   input channels = 4096
	 *   output_channels = 4096
	 */
	inline FullyConnectedTester fc7() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(4096)
			.outputChannels(4096));
	}

	/*
	 * AlexNet fc8 layer:
	 *   input channels = 4096
	 *   output_channels = 1000
	 */
	inline FullyConnectedTester fc8() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(4096)
			.outputChannels(1000));
	}

}
