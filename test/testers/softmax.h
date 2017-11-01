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

class SoftmaxTester {
public:
	SoftmaxTester() :
		iterations_(1),
		errorLimit_(1.0e-5),
		multithreading_(false),
		batchSize_(1),
		channels_(1)
	{
		this->threadpool = nullptr;
	}

	SoftmaxTester(const SoftmaxTester&) = delete;

	inline SoftmaxTester(SoftmaxTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		multithreading_(tester.multithreading_),
		batchSize_(tester.batchSize_),
		channels_(tester.channels_),
		threadpool(tester.threadpool)
	{
		tester.threadpool = nullptr;
	}

	SoftmaxTester& operator=(const SoftmaxTester&) = delete;

	~SoftmaxTester() {
		if (this->threadpool != nullptr) {
			pthreadpool_destroy(this->threadpool);
			this->threadpool = nullptr;
		}
	}

	inline SoftmaxTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline SoftmaxTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	inline SoftmaxTester& multithreading(bool multithreading) {
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

	inline SoftmaxTester& batchSize(size_t batchSize) {
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const {
		return this->batchSize_;
	}

	inline SoftmaxTester& channels(size_t channels) {
		this->channels_ = channels;
		return *this;
	}

	inline size_t channels() const {
		return this->channels_;
	}

	void testOutput() const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::mt19937(seed));

		std::vector<float> input(batchSize() * channels());
		std::vector<float> output(batchSize() * channels());
		std::vector<float> referenceOutput(batchSize() * channels());

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));

			nnp_softmax_output__reference(
				batchSize(), channels(),
				input.data(), referenceOutput.data(),
				this->threadpool);

			enum nnp_status status = nnp_softmax_output(
				batchSize(), channels(),
				input.data(), output.data(),
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

		std::vector<float> data(batchSize() * channels());
		std::vector<float> referenceData(batchSize() * channels());
		const float negativeSlope = 0.2f;

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(data.begin(), data.end(), std::ref(rng));
			std::copy(data.cbegin(), data.cend(), referenceData.begin());

			nnp_softmax_output__reference(
				batchSize(), channels(),
				referenceData.data(), referenceData.data(),
				this->threadpool);

			enum nnp_status status = nnp_softmax_output(
				batchSize(), channels(),
				data.data(), data.data(),
				this->threadpool);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceData.cbegin(), referenceData.cend(), data.cbegin(), 0.0f,
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
};
