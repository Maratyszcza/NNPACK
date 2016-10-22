#pragma once

#include <nnpack.h>

#include <testers/convolution.h>
#include <testers/fully-connected.h>
#include <testers/pooling.h>
#include <testers/relu.h>

namespace OverFeat_Fast {

	/*
	 * OverFeat (Fast model) conv1 layer:
	 *   input channels     = 3
	 *   output channels    = 96
	 *   input size         = 231x231
	 *   implicit padding   = 0
	 *   kernel size        = 11x11
	 *   output subsampling = 4x4
	 */
	inline ConvolutionTester conv1() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(3)
			.outputChannels(96)
			.inputSize(231, 231)
			.kernelSize(11, 11)
			.outputSubsampling(4, 4));
	}

	/*
	 * OverFeat (Fast model) conv1 ReLU layer
	 *   channels   = 96
	 *   image size = 56x56
	 */
	inline ReLUTester conv1_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(96)
			.imageSize(56, 56));
	}

	/*
	 * OverFeat (Fast model) conv2 layer:
	 *   input channels   = 96
	 *   output channels  = 256
	 *   input size       = 24x24
	 *   implicit padding = 0
	 *   kernel size      = 5x5
	 */
	inline ConvolutionTester conv2() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(96)
			.outputChannels(256)
			.inputSize(24, 24)
			.kernelSize(5, 5));
	}

	/*
	 * OverFeat (Fast model) conv2 ReLU layer
	 *   channels   = 256
	 *   image size = 24x24
	 */
	inline ReLUTester conv2_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(256)
			.imageSize(24, 24));
	}

	/*
	 * OverFeat (Fast model) conv3 layer:
	 *   input channels   = 256
	 *   output channels  = 512
	 *   input size       = 12x12
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv3() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(256)
			.outputChannels(512)
			.inputSize(12, 12)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * OverFeat (Fast model) conv3 ReLU layer
	 *   channels   = 512
	 *   image size = 12x12
	 */
	inline ReLUTester conv3_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(512)
			.imageSize(12, 12));
	}

	/*
	 * OverFeat (Fast model) conv4 layer:
	 *   input channels   = 512
	 *   output channels  = 1024
	 *   input size       = 12x12
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv4() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(512)
			.outputChannels(1024)
			.inputSize(12, 12)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * OverFeat (Fast model) conv4 ReLU layer
	 *   channels   = 1024
	 *   image size = 12x12
	 */
	inline ReLUTester conv4_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(1024)
			.imageSize(12, 12));
	}

	/*
	 * OverFeat (Fast model) conv5 layer:
	 *   input channels   = 1024
	 *   output channels  = 1024
	 *   input size       = 12x12
	 *   implicit padding = 1
	 *   kernel size      = 3x3
	 */
	inline ConvolutionTester conv5() {
		return std::move(ConvolutionTester()
			.multithreading(true)
			.inputChannels(1024)
			.outputChannels(1024)
			.inputSize(12, 12)
			.kernelSize(3, 3)
			.inputPadding(1, 1, 1, 1));
	}

	/*
	 * OverFeat (Fast model) fc6 layer:
	 *   input channels  = 36864
	 *   output channels = 3072
	 */
	inline FullyConnectedTester fc6() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(36864)
			.outputChannels(3072));
	}

	/*
	 * OverFeat (Fast model) fc6 ReLU layer
	 *   channels = 3072
	 */
	inline ReLUTester fc6_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(3072));
	}

	/*
	 * OverFeat (Fast model) fc7 layer:
	 *   input channels  = 3072
	 *   output channels = 4096
	 */
	inline FullyConnectedTester fc7() {
		return std::move(FullyConnectedTester()
			.multithreading(true)
			.inputChannels(3072)
			.outputChannels(4096));
	}

	/*
	 * OverFeat (Fast model) fc7 ReLU layer
	 *   channels = 4096
	 */
	inline ReLUTester fc7_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(4096));
	}

	/*
	 * OverFeat (Fast model) fc8 layer:
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
	 * OverFeat (Fast model) fc8 ReLU layer
	 *   channels = 1000
	 */
	inline ReLUTester fc8_relu() {
		return std::move(ReLUTester()
			.multithreading(true)
			.channels(1000));
	}

	/*
	 * OverFeat (Fast model) pool1 layer:
	 *   channels         = 96
	 *   input size       = 48x48
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool1() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(96)
			.inputSize(48, 48)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

	/*
	 * OverFeat (Fast model) pool2 layer:
	 *   channels         = 256
	 *   input size       = 24x24
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool2() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(256)
			.inputSize(24, 24)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

	/*
	 * OverFeat (Fast model) pool3 layer:
	 *   channels         = 1024
	 *   input size       = 12x12
	 *   implicit padding = 0
	 *   pooling size     = 2x2
	 *   pooling stride   = 2x2
	 */
	inline PoolingTester pool3() {
		return std::move(PoolingTester()
			.multithreading(true)
			.channels(1024)
			.inputSize(12, 12)
			.poolingSize(2, 2)
			.poolingStride(2, 2));
	}

};
