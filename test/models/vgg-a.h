#pragma once

#include <nnpack.h>

#include <testers/convolution.h>
#include <testers/fully-connected.h>
#include <testers/pooling.h>

namespace VGG_A {

	/*
	 * VGG model A conv1 layer:
	 *   input channels   = 3
	 *   output channels  = 64
	 *   input size       = 224x224
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv1() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(3)
			.outputChannels(64)
			.inputSize(224, 224)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A conv2 layer:
	 *   input channels   = 64
	 *   output channels  = 128
	 *   input size       = 112x112
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv2() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(64)
			.outputChannels(128)
			.inputSize(112, 112)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A conv3 layer:
	 *   input channels   = 128
	 *   output channels  = 256
	 *   input size       = 56x56
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv3() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(128)
			.outputChannels(256)
			.inputSize(56, 56)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A conv4 layer:
	 *   input channels   = 256
	 *   output channels  = 256
	 *   input size       = 56x56
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv4() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(256)
			.outputChannels(256)
			.inputSize(56, 56)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A conv5 layer:
	 *   input channels   = 256
	 *   output channels  = 512
	 *   input size       = 28x28
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv5() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(256)
			.outputChannels(512)
			.inputSize(28, 28)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A conv6 layer:
	 *   input channels   = 512
	 *   output channels  = 512
	 *   input size       = 28x28
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv6() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(512)
			.outputChannels(512)
			.inputSize(28, 28)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A conv8 layer:
	 *   input channels   = 512
	 *   output channels  = 512
	 *   input size       = 14x14
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv8() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(512)
			.outputChannels(512)
			.inputSize(14, 14)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * VGG model A fc6 layer:
	 *   input channels  = 25088
	 *   output channels = 4096
	 */
	inline FullyConnectedTester fc6() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(25088)
			.outputChannels(4096));
	}

	/*
	 * VGG model A fc7 layer:
	 *   input channels  = 4096
	 *   output channels = 4096
	 */
	inline FullyConnectedTester fc7() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(4096)
			.outputChannels(4096));
	}

	/*
	 * VGG model A fc8 layer:
	 *   input channels  = 4096
	 *   output channels = 1000
	 */
	inline FullyConnectedTester fc8() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(4096)
			.outputChannels(1000));
	}

	/*
	 * VGG model A pool1 layer:
	 *   channels         = 64
	 *   input size       = 224x224
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool1() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(64)
			.inputSize(224, 224)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

	/*
	 * VGG model A pool2 layer:
	 *   channels         = 128
	 *   input size       = 112x112
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool2() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(128)
			.inputSize(112, 112)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

	/*
	 * VGG model A pool3 layer:
	 *   channels         = 256
	 *   input size       = 56x56
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool3() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(256)
			.inputSize(56, 56)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

	/*
	 * VGG model A pool4 layer:
	 *   channels         = 512
	 *   input size       = 28x28
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool4() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(512)
			.inputSize(28, 28)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

	/*
	 * VGG model A pool5 layer:
	 *   channels         = 512
	 *   input size       = 14x14
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool5() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(512)
			.inputSize(14, 14)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

};
