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

#include <nnpack/winograd.h>
#include <nnpack/transform.h>
#include <nnpack/hwinfo.h>
#include <nnpack/AlignedAllocator.h>

class WinogradTransformTester {
public:
	inline WinogradTransformTester() :
		kernelSize_(0),
		outputSize_(0),
		simdWidth_(1),
		iterations_(1000),
		errorLimit_(1.0e-5)
	{
	}

	inline WinogradTransformTester& kernelSize(size_t kernelSize) {
		this->kernelSize_ = kernelSize;
		return *this;
	}

	inline size_t kernelSize() const {
		return this->kernelSize_;
	}

	inline WinogradTransformTester& outputSize(size_t outputSize) {
		this->outputSize_ = outputSize;
		return *this;
	}

	inline size_t outputSize() const {
		return this->outputSize_;
	}

	inline size_t inputSize() const {
		return this->outputSize_ + this->kernelSize_ - 1;
	}

	inline size_t transformSize() const {
		return this->outputSize_ + this->kernelSize_ - 1;
	}

	inline WinogradTransformTester& simdWidth(size_t simdWidth) {
		this->simdWidth_ = simdWidth;
		return *this;
	}

	inline size_t simdWidth() const {
		return this->simdWidth_;
	}

	inline WinogradTransformTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline WinogradTransformTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	/**
	 * Validates that 1D winograd transform for the input matches the linear transformation defined by a matrix.
	 */
	void testInputTransform(nnp_wt_function inputTransform, const float transformationMatrix[]) const {
		testTransform(inputTransform, inputSize(), transformSize(), transformationMatrix);
	}

	/**
	 * Validates that 2D winograd transform for the input matches the linear transformation defined by a matrix.
	 */
	void testInputTransform2D(nnp_transform_2d_with_offset inputTransform, const float transformationMatrix[]) const {
		testTransform2D(inputTransform, Type::input, inputSize(), transformSize(), inputSize(), simdWidth() * sizeof(float), transformationMatrix);
	}

	/**
	 * Validates that 1D winograd transform for the kernel matches the linear transformation defined by a matrix.
	 */
	void testKernelTransform(nnp_wt_function kernelTransform, const float transformationMatrix[]) const {
		testTransform(kernelTransform, kernelSize(), transformSize(), transformationMatrix);
	}

	/**
	 * Validates that 2D winograd transform for the kernel matches the linear transformation defined by a matrix.
	 */
	void testKernelTransform2D(nnp_transform_2d_with_offset kernelTransform, const float transformationMatrix[]) const {
		testTransform2D(kernelTransform, Type::kernel, kernelSize(), transformSize(), kernelSize(), simdWidth() * sizeof(float), transformationMatrix);
	}

	/**
	 * Validates that 1D winograd transform for the output matches the linear transformation defined by a matrix.
	 */
	void testOutputTransform(nnp_wt_function outputTransform, const float transformationMatrix[]) const {
		testTransform(outputTransform, transformSize(), outputSize(), transformationMatrix);
	}

	/**
	 * Validates that 2D winograd transform for the output matches the linear transformation defined by a matrix.
	 */
	void testOutputTransform2D(nnp_transform_2d_with_offset outputTransform, const float transformationMatrix[]) const {
		testTransform2D(outputTransform, Type::output, transformSize(), outputSize(), simdWidth() * sizeof(float), outputSize(), transformationMatrix);
	}

private:
	enum class Type {
		input,
		kernel,
		output,
	};

	void testTransform(nnp_wt_function transform, size_t inputSize, size_t outputSize,
		const float transformationMatrix[]) const
	{
		ASSERT_NE(this->kernelSize(), 0);
		ASSERT_NE(this->outputSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float> input(inputSize * simdWidth());
		std::vector<float> output(outputSize * simdWidth());
		std::vector<float> referenceOutput(outputSize * simdWidth());

		std::vector<std::vector<float>> errors(outputSize * simdWidth());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), std::nanf(""));
			std::fill(referenceOutput.begin(), referenceOutput.end(), 0.0f);

			transform(input.data(), output.data());

			for (size_t simdLane = 0; simdLane < simdWidth(); simdLane += 1) {
				/* Transformation matrix is outputSize x inputSize */
				for (size_t i = 0; i < outputSize; i += 1) {
					for (size_t j = 0; j < inputSize; j += 1) {
						referenceOutput[i * simdWidth() + simdLane] +=
							transformationMatrix[i * inputSize + j] * input[j * simdWidth() + simdLane];
					}
				}
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(output[i], referenceOutput[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	void testTransform2D(nnp_transform_2d_with_offset transform2d, Type type, size_t inputSize, size_t outputSize, size_t inputStride, size_t outputStride,
		const float transformationMatrix[]) const
	{
		ASSERT_NE(this->kernelSize(), 0);
		ASSERT_NE(this->outputSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float, AlignedAllocator<float, 64>> simdInput(inputSize * inputSize);
		std::vector<float, AlignedAllocator<float, 64>> simdOutput(outputSize * outputSize);
		std::vector<float> input(inputSize * inputSize);
		std::vector<float> output(outputSize * outputSize);
		std::vector<float> referenceOutput(outputSize * outputSize);

		std::vector<std::vector<float>> errors(outputSize * outputSize);
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), std::nanf(""));
			std::fill(referenceOutput.begin(), referenceOutput.end(), 0.0f);

			if (type == Type::output) {
				for (size_t row = 0; row < inputSize; row += 1) {
					for (size_t column = 0; column < inputSize; column += simdWidth()) {
						std::copy(input.cbegin() + row * inputSize + column,
							input.cbegin() + row * inputSize + column + simdWidth(),
							simdInput.begin() + column * inputSize + row * simdWidth());
					}
				}
			} else {
				std::copy(input.cbegin(), input.cend(), simdInput.begin());
			}
			transform2d(simdInput.data(), simdOutput.data(), inputStride, outputStride,
				type == Type::output ? outputSize : inputSize, type == Type::output ? outputSize : inputSize, 0, 0);
			if (type != Type::output) {
				for (size_t row = 0; row < inputSize; row += 1) {
					for (size_t column = 0; column < inputSize; column += simdWidth()) {
						std::copy(simdOutput.cbegin() + column * outputSize + row * simdWidth(),
							simdOutput.cbegin() + column * outputSize + (row + 1) * simdWidth(),
							output.begin() + row * outputSize + column);
					}
				}
			} else {
				std::copy(simdOutput.cbegin(), simdOutput.cend(), output.begin());
			}

			for (size_t p = 0; p < outputSize; p += 1) {
				for (size_t q = 0; q < outputSize; q += 1) {
					/* Transformation matrix is outputSize x inputSize */
					for (size_t r = 0; r < inputSize; r += 1) {
						for (size_t s = 0; s < inputSize; s += 1) {
							referenceOutput[p * outputSize + q] +=
								transformationMatrix[p * inputSize + r] *
								transformationMatrix[q * inputSize + s] *
								input[s * inputSize + r];
						}
					}
				}
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(output[i], referenceOutput[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	inline static float relativeError(float reference, float actual) {
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

	inline static std::vector<float> median(std::vector<std::vector<float>>& matrix) {
		for (auto& row : matrix) {
			std::nth_element(row.begin(), row.begin() + row.size() / 2, row.end());
		}

		std::vector<float> medians(matrix.size());
		std::transform(matrix.cbegin(), matrix.cend(), medians.begin(),
			[](const std::vector<float>& elementErrors) { return elementErrors[elementErrors.size() / 2]; });

		return std::move(medians);
	}

	size_t kernelSize_;
	size_t outputSize_;
	size_t simdWidth_;
	size_t iterations_;
	float errorLimit_;
};
