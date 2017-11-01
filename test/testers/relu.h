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
#include <nnpack/utils.h>

class ReLUTester {
public:
	ReLUTester() :
		iterations_(1),
		errorLimit_(1.0e-7),
		multithreading_(false),
		batchSize_(1),
		channels_(1)
	{
		imageSize(1, 1);

		this->threadpool = nullptr;
	}

	ReLUTester(const ReLUTester&) = delete;

	inline ReLUTester(ReLUTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		multithreading_(tester.multithreading_),
		batchSize_(tester.batchSize_),
		channels_(tester.channels_),
		imageHeight_(tester.imageHeight_),
		imageWidth_(tester.imageWidth_),
		threadpool(tester.threadpool)
	{
		tester.threadpool = nullptr;
	}

	ReLUTester& operator=(const ReLUTester&) = delete;

	~ReLUTester() {
		if (this->threadpool != nullptr) {
			pthreadpool_destroy(this->threadpool);
			this->threadpool = nullptr;
		}
	}

	inline ReLUTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline ReLUTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	inline ReLUTester& multithreading(bool multithreading) {
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

	inline ReLUTester& batchSize(size_t batchSize) {
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const {
		return this->batchSize_;
	}

	inline ReLUTester& channels(size_t channels) {
		this->channels_ = channels;
		return *this;
	}

	inline size_t channels() const {
		return this->channels_;
	}

	inline ReLUTester& imageSize(size_t height, size_t width) {
		this->imageHeight_ = height;
		this->imageWidth_ = width;
		return *this;
	}

	inline size_t imageHeight() const {
		return this->imageHeight_;
	}

	inline size_t imageWidth() const {
		return this->imageWidth_;
	}

	void testOutput() const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::mt19937(seed));

		std::vector<float> input(batchSize() * channels() * imageHeight() * imageWidth());
		std::vector<float> output(batchSize() * channels() * imageHeight() * imageWidth());
		std::vector<float> referenceOutput(batchSize() * channels() * imageHeight() * imageWidth());
		const float negativeSlope = 0.2f;

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));

			nnp_relu_output__reference(
				batchSize(), channels() * imageHeight() * imageWidth(),
				input.data(), referenceOutput.data(), negativeSlope,
				this->threadpool);

			enum nnp_status status = nnp_relu_output(
				batchSize(), channels() * imageHeight() * imageWidth(),
				input.data(), output.data(), negativeSlope,
				this->threadpool);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			EXPECT_LT(maxError, errorLimit());
		}
	}

	void testOutputInplace() const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::mt19937(seed));

		std::vector<float> data(batchSize() * channels() * imageHeight() * imageWidth());
		std::vector<float> referenceData(batchSize() * channels() * imageHeight() * imageWidth());
		const float negativeSlope = 0.2f;

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(data.begin(), data.end(), std::ref(rng));
			std::copy(data.cbegin(), data.cend(), referenceData.begin());

			nnp_relu_output__reference(
				batchSize(), channels() * imageHeight() * imageWidth(),
				referenceData.data(), referenceData.data(), negativeSlope,
				this->threadpool);

			enum nnp_status status = nnp_relu_output(
				batchSize(), channels() * imageHeight() * imageWidth(),
				data.data(), data.data(), negativeSlope,
				this->threadpool);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceData.cbegin(), referenceData.cend(), data.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			EXPECT_LT(maxError, errorLimit());
		}
	}

	void testInputGradient() const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::mt19937(seed));

		std::vector<float> outputGradient(batchSize() * channels() * imageHeight() * imageWidth());
		std::vector<float> input(batchSize() * channels() * imageHeight() * imageWidth());
		std::vector<float> inputGradient(batchSize() * channels() * imageHeight() * imageWidth());
		std::vector<float> referenceInputGradient(batchSize() * channels() * imageHeight() * imageWidth());
		const float negativeSlope = 0.2f;

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(outputGradient.begin(), outputGradient.end(), std::ref(rng));
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(inputGradient.begin(), inputGradient.end(), nanf(""));
			std::fill(referenceInputGradient.begin(), referenceInputGradient.end(), nanf(""));

			nnp_relu_input_gradient__reference(
				batchSize(), channels() * imageHeight() * imageWidth(),
				outputGradient.data(), input.data(), referenceInputGradient.data(), negativeSlope,
				this->threadpool);

			enum nnp_status status = nnp_relu_input_gradient(
				batchSize(), channels() * imageHeight() * imageWidth(),
				outputGradient.data(), input.data(), inputGradient.data(), negativeSlope,
				this->threadpool);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceInputGradient.cbegin(), referenceInputGradient.cend(), inputGradient.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			EXPECT_LT(maxError, errorLimit());
		}
	}

protected:
	pthreadpool_t threadpool;

private:
	inline static float relativeError(float reference, float actual) {
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

	size_t iterations_;
	float errorLimit_;
	bool multithreading_;

	size_t batchSize_;
	size_t channels_;
	size_t imageHeight_;
	size_t imageWidth_;
};
