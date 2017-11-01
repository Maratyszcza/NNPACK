#pragma once

#include <cstddef>
#include <cstdlib>

#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#include <nnpack.h>
#include <nnpack/reference.h>

#include <testers/relu.h>

class ConvolutionTester {
public:
	ConvolutionTester() :
		iterations_(1),
		errorLimit_(1.0e-5),
		multithreading_(false),
		batchSize_(1),
		inputChannels_(1),
		outputChannels_(1)
	{
		inputSize(4, 4);
		kernelSize(3, 3);
		inputPadding(0, 0, 0, 0);
		outputSubsampling(1, 1);

		this->threadpool = nullptr;
	}

	ConvolutionTester(const ConvolutionTester&) = delete;

	inline ConvolutionTester(ConvolutionTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		multithreading_(tester.multithreading_),
		batchSize_(tester.batchSize_),
		inputChannels_(tester.inputChannels_),
		outputChannels_(tester.outputChannels_),
		inputSize_(tester.inputSize_),
		inputPadding_(tester.inputPadding_),
		kernelSize_(tester.kernelSize_),
		outputSubsampling_(tester.outputSubsampling_),
		threadpool(tester.threadpool)
	{
		tester.threadpool = nullptr;
	}

	ConvolutionTester& operator=(const ConvolutionTester&) = delete;

	~ConvolutionTester() {
		if (this->threadpool != nullptr) {
			pthreadpool_destroy(this->threadpool);
			this->threadpool = nullptr;
		}
	}

	inline ConvolutionTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline ConvolutionTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	inline ConvolutionTester& multithreading(bool multithreading) {
		this->multithreading_ = multithreading;
		if (multithreading && this->threadpool == nullptr) {
			this->threadpool = pthreadpool_create(0);
		} else if (!multithreading && this->threadpool != nullptr) {
			pthreadpool_destroy(this->threadpool);
			this->threadpool = nullptr;
		}
		return *this;
	}

	inline bool multithreading() const {
		return this->multithreading_;
	}

	inline ConvolutionTester& batchSize(size_t batchSize) {
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const {
		return this->batchSize_;
	}

	inline ConvolutionTester& inputChannels(size_t inputChannels) {
		this->inputChannels_ = inputChannels;
		return *this;
	}

	inline size_t inputChannels() const {
		return this->inputChannels_;
	}

	inline ConvolutionTester& outputChannels(size_t outputChannels) {
		this->outputChannels_ = outputChannels;
		return *this;
	}

	inline size_t outputChannels() const {
		return this->outputChannels_;
	}

	inline ConvolutionTester& inputSize(size_t height, size_t width) {
		this->inputSize_.height = height;
		this->inputSize_.width = width;
		return *this;
	}

	inline struct nnp_size inputSize() const {
		return this->inputSize_;
	}

	inline size_t inputHeight() const {
		return this->inputSize_.height;
	}

	inline size_t inputWidth() const {
		return this->inputSize_.width;
	}

	inline ConvolutionTester& kernelSize(size_t height, size_t width) {
		this->kernelSize_.height = height;
		this->kernelSize_.width = width;
		return *this;
	}

	inline struct nnp_size kernelSize() const {
		return this->kernelSize_;
	}

	inline size_t kernelHeight() const {
		return this->kernelSize_.height;
	}

	inline size_t kernelWidth() const {
		return this->kernelSize_.width;
	}

	inline struct nnp_size outputSize() const {
		struct nnp_size outputSize;
		outputSize.height = this->outputHeight();
		outputSize.width = this->outputWidth();
		return outputSize;
	}

	inline size_t outputHeight() const {
		return (this->inputPadding_.top + this->inputSize_.height + this->inputPadding_.bottom - this->kernelSize_.height) / this->outputSubsampling_.height + 1;
	}

	inline size_t outputWidth() const {
		return (this->inputPadding_.left + this->inputSize_.width + this->inputPadding_.right - this->kernelSize_.width) / this->outputSubsampling_.width + 1;
	}

	inline ConvolutionTester& outputSubsampling(size_t height, size_t width) {
		this->outputSubsampling_.height = height;
		this->outputSubsampling_.width = width;
		return *this;
	}

	inline struct nnp_size outputSubsampling() const {
		return this->outputSubsampling_;
	}

	inline ConvolutionTester& inputPadding(size_t top, size_t right, size_t bottom, size_t left) {
		this->inputPadding_.top = top;
		this->inputPadding_.right = right;
		this->inputPadding_.bottom = bottom;
		this->inputPadding_.left = left;
		return *this;
	}

	inline struct nnp_padding inputPadding() const {
		return this->inputPadding_;
	}

	void testOutput(enum nnp_convolution_algorithm algorithm, enum nnp_activation activation = nnp_activation_identity) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-0.1, 1.0), std::mt19937(seed));

		std::vector<float> input(batchSize() * inputChannels() * inputHeight() * inputWidth());
		std::vector<float> kernel(outputChannels() * inputChannels() * kernelHeight() * kernelWidth());

		std::vector<float> bias(outputChannels());

		std::vector<float> output(batchSize() * outputChannels() * outputHeight() * outputWidth());
		std::vector<float> referenceOutput(batchSize() * outputChannels() * outputHeight() * outputWidth());

		std::vector<float> maxErrors;
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::generate(bias.begin(), bias.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));

			nnp_convolution_output__reference(
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), kernel.data(), bias.data(), referenceOutput.data(),
				this->threadpool);

			switch (activation) {
				case nnp_activation_identity:
					break;
				case nnp_activation_relu:
					nnp_relu_output__reference(
						batchSize(), outputChannels() * outputHeight() * outputWidth(),
						referenceOutput.data(), referenceOutput.data(), 0.0,
						this->threadpool);
					break;
				default:
					FAIL() << "Unexpected activation value: " << activation;
			}

			enum nnp_status status = nnp_convolution_output(
				algorithm,
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(),
				input.data(), kernel.data(), bias.data(), output.data(), NULL, NULL,
				activation, NULL,
				this->threadpool, nullptr);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		EXPECT_LT(median(maxErrors), errorLimit());
	}

	void testInputGradient(enum nnp_convolution_algorithm algorithm, enum nnp_activation activation = nnp_activation_identity) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float> outputGradient(batchSize() * outputChannels() * outputHeight() * outputWidth());
		std::vector<float> kernel(outputChannels() * inputChannels() * kernelHeight() * kernelWidth());

		std::vector<float> inputGradient(batchSize() * inputChannels() * inputHeight() * inputWidth());

		std::vector<float> referenceInputGradient(batchSize() * inputChannels() * inputHeight() * inputWidth());

		std::vector<float> maxErrors;
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(outputGradient.begin(), outputGradient.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::fill(inputGradient.begin(), inputGradient.end(), nanf(""));

			nnp_convolution_input_gradient__reference(
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(),
				outputGradient.data(), kernel.data(), referenceInputGradient.data(),
				this->threadpool);

			enum nnp_status status = nnp_convolution_input_gradient(
				algorithm,
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(),
				outputGradient.data(), kernel.data(), inputGradient.data(), NULL, NULL,
				nnp_activation_identity, NULL,
				this->threadpool, nullptr);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceInputGradient.cbegin(), referenceInputGradient.cend(), inputGradient.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		EXPECT_LT(median(maxErrors), errorLimit());
	}

	void testKernelGradient(enum nnp_convolution_algorithm algorithm, enum nnp_activation activation = nnp_activation_identity) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float> input(batchSize() * inputChannels() * inputHeight() * inputWidth());
		std::vector<float> outputGradient(batchSize() * outputChannels() * outputHeight() * outputWidth());
		std::vector<float> kernelGradient(outputChannels() * inputChannels() * kernelHeight() * kernelWidth());

		std::vector<float> referenceKernelGradient(outputChannels() * inputChannels() * kernelHeight() * kernelWidth());

		std::vector<float> maxErrors;
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(outputGradient.begin(), outputGradient.end(), std::ref(rng));
			std::fill(kernelGradient.begin(), kernelGradient.end(), nanf(""));

			nnp_convolution_kernel_gradient__reference(
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(),
				input.data(), outputGradient.data(), referenceKernelGradient.data(),
				this->threadpool);

			enum nnp_status status = nnp_convolution_kernel_gradient(
				algorithm,
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(),
				input.data(), outputGradient.data(), kernelGradient.data(), NULL, NULL,
				nnp_activation_identity, NULL,
				this->threadpool,
				NULL);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceKernelGradient.cbegin(), referenceKernelGradient.cend(), kernelGradient.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		EXPECT_LT(median(maxErrors), errorLimit());
	}

	void testInference(enum nnp_convolution_algorithm algorithm, enum nnp_activation activation = nnp_activation_identity) const {
		ASSERT_EQ(1, batchSize());

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-0.1, 1.0), std::mt19937(seed));

		std::vector<float> input(inputChannels() * inputHeight() * inputWidth());
		std::vector<float> kernel(outputChannels() * inputChannels() * kernelHeight() * kernelWidth());

		std::vector<float> bias(outputChannels());

		std::vector<float> output(outputChannels() * outputHeight() * outputWidth());
		std::vector<float> referenceOutput(outputChannels() * outputHeight() * outputWidth());

		std::vector<float> maxErrors;
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::generate(bias.begin(), bias.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));

			nnp_convolution_output__reference(
				1, inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), kernel.data(), bias.data(), referenceOutput.data(),
				this->threadpool);

			switch (activation) {
				case nnp_activation_identity:
					break;
				case nnp_activation_relu:
					nnp_relu_output__reference(
						batchSize(), outputChannels() * outputHeight() * outputWidth(),
						referenceOutput.data(), referenceOutput.data(), 0.0,
						this->threadpool);
					break;
				default:
					FAIL() << "Unexpected activation value: " << activation;
			}

			enum nnp_status status = nnp_convolution_inference(
				algorithm, nnp_convolution_transform_strategy_compute,
				inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), kernel.data(), bias.data(), output.data(), NULL, NULL,
				activation, NULL,
				this->threadpool, nullptr);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		EXPECT_LT(median(maxErrors), errorLimit());
	}

protected:
	pthreadpool_t threadpool;

private:
	inline static float relativeError(float reference, float actual) {
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

	inline static float median(std::vector<float>& array) {
		std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
		return array[array.size() / 2];
	}

	size_t iterations_;
	float errorLimit_;
	bool multithreading_;

	size_t batchSize_;
	size_t inputChannels_;
	size_t outputChannels_;
	struct nnp_size inputSize_;
	struct nnp_padding inputPadding_;
	struct nnp_size kernelSize_;
	struct nnp_size outputSubsampling_;
};
