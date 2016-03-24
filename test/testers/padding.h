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

#include <nnpack/simd.h>

class Padded2DTransformTester {
public:
	Padded2DTransformTester(size_t fftSize) :
		fftSize(fftSize)
	{
	}

	template<class FFTFunctionT>
	void testForwardTransform(FFTFunctionT forwardTransform) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		float rawInput[this->fftSize * this->fftSize];
		std::generate(rawInput, rawInput + this->fftSize * this->fftSize, std::ref(rng));
		for (size_t rowOffset = 0; rowOffset < this->fftSize; rowOffset++) {
			for (size_t rowCount = 1; rowCount <= this->fftSize - rowOffset; rowCount++) {
				for (size_t colOffset = 0; colOffset < this->fftSize; colOffset++) {
					for (size_t colCount = 1; colCount <= this->fftSize - colOffset; colCount++) {
						float packedInput[rowCount * colCount];
						for (size_t i = 0; i < rowCount; i++) {
							for (size_t j = 0; j < colCount; j++) {
								packedInput[i * colCount + j] = rawInput[(rowOffset + i) * this->fftSize + (colOffset + j)];
							}
						}
						NNP_SIMD_ALIGN float packedResult[this->fftSize * this->fftSize];
						memset(packedResult, -1, sizeof(packedResult));
						forwardTransform(packedInput, packedResult,
							colCount, 64,
							rowCount, colCount, rowOffset, colOffset);

						float paddedInput[this->fftSize * this->fftSize];
						for (size_t i = 0; i < this->fftSize; i++) {
							for (size_t j = 0; j < this->fftSize; j++) {
								if ((i >= rowOffset) && (i < rowOffset + rowCount) && (j >= colOffset) && (j < colOffset + colCount)) {
									paddedInput[i * this->fftSize + j] = rawInput[i * this->fftSize + j];
								} else {
									paddedInput[i * this->fftSize + j] = 0.0f;
								}
							}
						}
						NNP_SIMD_ALIGN float paddedResult[this->fftSize * this->fftSize];
						memset(paddedResult, -1, sizeof(paddedResult));
						forwardTransform(paddedInput, paddedResult,
							this->fftSize, 64,
							this->fftSize, this->fftSize, 0, 0);

						ASSERT_EQ(memcmp(packedResult, paddedResult, this->fftSize * this->fftSize * sizeof(float)), 0) << "ROWS: " << rowCount << " (offset " << rowOffset << ") COLUMNS: " << colCount << " (offset " << colOffset << ")";
					}
				}
			}
		}
	}

	template<class IFFTFunctionT>
	void testInverseTransform(IFFTFunctionT inverseTransform) {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		NNP_SIMD_ALIGN float input[this->fftSize * this->fftSize];
		std::generate(input, input + this->fftSize * this->fftSize, std::ref(rng));

		NNP_SIMD_ALIGN float inputCopy[this->fftSize * this->fftSize];
		memcpy(inputCopy, input, sizeof(input));
		float output[this->fftSize * this->fftSize];
		inverseTransform(inputCopy, output, 64, this->fftSize, this->fftSize, this->fftSize, 0, 0);

		for (size_t rowOffset = 0; rowOffset < this->fftSize; rowOffset++) {
			for (size_t rowCount = 1; rowCount <= this->fftSize - rowOffset; rowCount++) {
				for (size_t colOffset = 0; colOffset < this->fftSize; colOffset++) {
					for (size_t colCount = 1; colCount <= this->fftSize - colOffset; colCount++) {
						float packedOutput[rowCount * colCount];
						memset(packedOutput, -1, sizeof(packedOutput));
						memcpy(inputCopy, input, sizeof(input));
						inverseTransform(inputCopy, packedOutput, 64, colCount, rowCount, colCount, rowOffset, colOffset);

						for (size_t i = 0; i < rowCount; i++) {
							ASSERT_EQ(memcmp(&output[(rowOffset + i) * this->fftSize + colOffset], &packedOutput[i * colCount], colCount * sizeof(float)), 0) << "ROWS: " << rowCount << " (offset " << rowOffset << ") COLUMNS: " << colCount << " (offset " << colOffset << ")";
						}
					}
				}
			}
		}
	}

protected:
	size_t fftSize;
};
