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

class PoolingTester {
public:
	PoolingTester() :
		iterations_(1),
		errorLimit_(1.0e-7),
		multithreading_(false),
		batchSize_(1),
		channels_(1)
	{
		inputSize(4, 4);
		inputPadding(0, 0, 0, 0);
		poolingSize(2, 2);
		poolingStride(2, 2);

		this->threadpool = nullptr;
	}

	PoolingTester(const PoolingTester&) = delete;

	inline PoolingTester(PoolingTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		multithreading_(tester.multithreading_),
		batchSize_(tester.batchSize_),
		channels_(tester.channels_),
		inputSize_(tester.inputSize_),
		inputPadding_(tester.inputPadding_),
		poolingSize_(tester.poolingSize_),
		poolingStride_(tester.poolingStride_),
		threadpool(tester.threadpool)
	{
		tester.threadpool = nullptr;
	}

	PoolingTester& operator=(const PoolingTester&) = delete;

	~PoolingTester() {
		if (this->threadpool != nullptr) {
			pthreadpool_destroy(this->threadpool);
			this->threadpool = nullptr;
		}
	}

	inline PoolingTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline PoolingTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	inline PoolingTester& multithreading(bool multithreading) {
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

	inline PoolingTester& batchSize(size_t batchSize) {
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const {
		return this->batchSize_;
	}

	inline PoolingTester& channels(size_t channels) {
		this->channels_ = channels;
		return *this;
	}

	inline size_t channels() const {
		return this->channels_;
	}

	inline PoolingTester& inputSize(size_t height, size_t width) {
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

	inline PoolingTester& poolingSize(size_t height, size_t width) {
		this->poolingSize_.height = height;
		this->poolingSize_.width = width;
		return *this;
	}

	inline struct nnp_size poolingSize() const {
		return this->poolingSize_;
	}

	inline size_t poolingHeight() const {
		return this->poolingSize_.height;
	}

	inline size_t poolingWidth() const {
		return this->poolingSize_.width;
	}

	inline PoolingTester& poolingStride(size_t heightStride, size_t widthStride) {
		this->poolingStride_.height = heightStride;
		this->poolingStride_.width = widthStride;
		return *this;
	}

	inline struct nnp_size poolingStride() const {
		return this->poolingStride_;
	}

	inline struct nnp_size outputSize() const {
		struct nnp_size outputSize;
		outputSize.height = this->outputHeight();
		outputSize.width = this->outputWidth();
		return outputSize;
	}

	inline size_t outputHeight() const {
		return 1 + divide_round_up(
			std::max(this->inputPadding_.top + this->inputSize_.height + this->inputPadding_.bottom, this->poolingSize_.height) -
			this->poolingSize_.height, this->poolingStride_.height);
	}

	inline size_t outputWidth() const {
		return 1 + divide_round_up(
			std::max(this->inputPadding_.left + this->inputSize_.width + this->inputPadding_.right, this->poolingSize_.width) -
			this->poolingSize_.width, this->poolingStride_.width);
	}

	inline PoolingTester& inputPadding(size_t top, size_t right, size_t bottom, size_t left) {
		this->inputPadding_.top = top;
		this->inputPadding_.right = right;
		this->inputPadding_.bottom = bottom;
		this->inputPadding_.left = left;
		return *this;
	}

	inline struct nnp_padding inputPadding() const {
		return this->inputPadding_;
	}

	void testOutput() const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float> input(batchSize() * channels() * inputHeight() * inputWidth());
		std::vector<float> output(batchSize() * channels() * outputHeight() * outputWidth());
		std::vector<float> referenceOutput(batchSize() * channels() * outputHeight() * outputWidth());

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));

			nnp_max_pooling_output__reference(
				batchSize(), channels(),
				inputSize(), inputPadding(), poolingSize(), poolingStride(),
				input.data(), referenceOutput.data(),
				this->threadpool);

			enum nnp_status status = nnp_max_pooling_output(
				batchSize(), channels(),
				inputSize(), inputPadding(), poolingSize(), poolingStride(),
				input.data(), output.data(),
				this->threadpool);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
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
	struct nnp_size inputSize_;
	struct nnp_padding inputPadding_;
	struct nnp_size poolingSize_;
	struct nnp_size poolingStride_;
};
