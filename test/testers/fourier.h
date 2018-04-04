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

#include <nnpack/fft.h>
#include <nnpack/AlignedAllocator.h>

class FFTTester {
public:
	inline FFTTester() :
		fftSize_(0),
		simdWidth_(1),
		iterations_(1000),
		errorLimit_(1.0e-5)
	{
	}

	inline FFTTester& fftSize(size_t fftSize) {
		this->fftSize_ = fftSize;
		return *this;
	}

	inline size_t fftSize() const {
		return this->fftSize_;
	}

	inline FFTTester& simdWidth(size_t simdWidth) {
		this->simdWidth_ = simdWidth;
		return *this;
	}

	inline size_t simdWidth() const {
		return this->simdWidth_;
	}

	inline FFTTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline FFTTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	/**
	 * Validates that output of forward complex 1D FFT with array-of-structure layout matches reference samples.
	 */
	void testForwardAosSamples(nnp_strided_fft_function forward_fft, const float input[], const float expectedOutput[]) const
	{
		ASSERT_NE(fftSize(), 0);

		std::vector<float> output(2 * fftSize());
		forward_fft(input, 1, output.data(), 1);

		const float maxError = std::inner_product(output.cbegin(), output.cend(), expectedOutput, 0.0f,
			[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
		ASSERT_LT(maxError, errorLimit());
	}

	/**
	 * Validates that complex 1D forward FFT + inverse FFT with array-of-structure layout is an identity transformation.
	 */
	void testForwardAndInverseAos(nnp_strided_fft_function forward_fft, nnp_strided_fft_function inverse_fft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Original sequence */
		std::vector<float> data(2 * fftSize());
		/* Original sequence after forward Fourier transform */
		std::vector<float> freqData(2 * fftSize());
		/* Original sequence after forward + inverse Fourier transform */
		std::vector<float> timeData(2 * fftSize());

		std::vector<std::vector<float>> errors(2 * fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(data.begin(), data.end(), std::ref(rng));
			std::fill(freqData.begin(), freqData.end(), std::nanf(""));
			std::fill(timeData.begin(), timeData.end(), std::nanf(""));

			forward_fft(data.data(), 1, freqData.data(), 1);
			inverse_fft(freqData.data(), 1, timeData.data(), 1);

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(data[i], timeData[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	/**
	 * Validates that complex 1D FFT with structure-of-arrays layout produces the same output
	 * as FFT with array-of-structures layout. This function works for both forward and inverse FFT.
	 */
	void testSoa(nnp_strided_fft_function soa_fft, nnp_strided_fft_function aos_fft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Input sequence with SOA layout */
		std::vector<float> inputSOA(2 * fftSize());
		/* The same input sequence reshuffled into AOS layout */
		std::vector<float> inputAOS(2 * fftSize());
		/* Output of complex 1D FFT with SOA layout */
		std::vector<float> outputSOA(2 * fftSize());
		/* The same output sequence reshuffled into AOS layout */
		std::vector<float> outputAOS(2 * fftSize());
		/* Ouptut of complex 1D FFT with AOS layout */
		std::vector<float> referenceOutputAOS(2 * fftSize());

		std::vector<std::vector<float>> errors(2 * fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(inputSOA.begin(), inputSOA.end(), std::ref(rng));
			std::fill(outputSOA.begin(), outputSOA.end(), std::nanf(""));
			std::fill(referenceOutputAOS.begin(), referenceOutputAOS.end(), std::nanf(""));

			soa_fft(inputSOA.data(), 1, outputSOA.data(), 1);

			/* Repack input and output from SOA to AOS */
			for (size_t i = 0; i < fftSize(); i++) {
				/* Real component */
				inputAOS[2 * i + 0] = inputSOA[i];
				outputAOS[2 * i + 0] = outputSOA[i];
				/* Imaginary component */
				inputAOS[2 * i + 1] = inputSOA[fftSize() + i];
				outputAOS[2 * i + 1] = outputSOA[fftSize() + i];
			}

			aos_fft(inputAOS.data(), 1, referenceOutputAOS.data(), 1);

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceOutputAOS[i], outputAOS[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	/**
	 * Validates that real-to-complex 1D FFT produces the same output as complex FFT with array-of-structures layout.
	 */
	void testRealToComplex(nnp_strided_fft_function real_fft, nnp_strided_fft_function aos_fft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Input for real-to-complex 1D FFT */
		std::vector<float> realInput(fftSize());
		/* Output of real-to-complex 1D FFT in MKL PERM format */
		std::vector<float> realOutput(fftSize());
		/* Input for complex 1D FFT */
		std::vector<float> complexInput(2 * fftSize());
		/* Output of complex 1D FFT */
		std::vector<float> complexOutput(2 * fftSize());
		/* Output of complex 1D FFT repacked into MKL PERM format */
		std::vector<float> referenceRealOutput(fftSize());

		std::vector<std::vector<float>> errors(fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(realInput.begin(), realInput.end(), std::ref(rng));
			std::fill(realOutput.begin(), realOutput.end(), std::nanf(""));
			std::fill(complexOutput.begin(), complexOutput.end(), std::nanf(""));

			/* Repack real input sequence into complex AOS sequence */
			for (size_t i = 0; i < fftSize(); i++) {
				/* Real component */
				complexInput[2 * i + 0] = realInput[i];
				/* Imaginary component */
				complexInput[2 * i + 1] = 0.0f;
			}

			real_fft(realInput.data(), 1, realOutput.data(), 1);
			aos_fft(complexInput.data(), 1, complexOutput.data(), 1);

			/* Repack complex AOS output into MKL PERM format */
			referenceRealOutput[0] = complexOutput[0];
			referenceRealOutput[1] = complexOutput[fftSize()];
			std::copy(complexOutput.cbegin() + 2, complexOutput.cbegin() + fftSize(), referenceRealOutput.begin() + 2);

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceRealOutput[i], realOutput[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	/**
	 * Validates that complex-to-real 1D FFT produces the same output as complex FFT with array-of-structures layout.
	 */
	void testComplexToReal(nnp_strided_fft_function real_ifft, nnp_strided_fft_function aos_ifft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Input for complex-to-real 1D IFFT in MKL PERM format */
		std::vector<float> realInput(fftSize());
		/* Real output of complex-to-real 1D IFFT */
		std::vector<float> realOutput(fftSize());
		/* Input for complex 1D IFFT in array-of-structures layout */
		std::vector<float> complexInput(2 * fftSize());
		/* Output of complex 1D IFFT in array-of-structures layout */
		std::vector<float> complexOutput(2 * fftSize());
		/* Real output of complex 1D IFFT */
		std::vector<float> referenceRealOutput(fftSize());

		std::vector<std::vector<float>> errors(fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(realInput.begin(), realInput.end(), std::ref(rng));
			std::fill(realOutput.begin(), realOutput.end(), std::nanf(""));
			std::fill(complexOutput.begin(), complexOutput.end(), std::nanf(""));

			/* Repack input sequence in MKL PERM format into complex AOS sequence */
			complexInput[0] = realInput[0];
			complexInput[fftSize()] = realInput[1];
			complexInput[1] = 0.0f;
			complexInput[fftSize() + 1] = 0.0f;
			for (size_t i = 2; i < fftSize(); i++) {
				complexInput[i] = realInput[i];
				if (i % 2 == 0) {
					/* Real component */
					complexInput[2 * fftSize() - i] = realInput[i];
				} else {
					/* Imaginary component */
					complexInput[2 * fftSize() - i + 2] = -realInput[i];
				}
			}

			real_ifft(realInput.data(), 1, realOutput.data(), 1);
			aos_ifft(complexInput.data(), 1, complexOutput.data(), 1);

			/* Repack complex AOS output into MKL PERM format */
			for (size_t i = 0; i < fftSize(); i++) {
				referenceRealOutput[i] = complexOutput[i * 2];
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceRealOutput[i], realOutput[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	/**
	 * Validates that dual-sequence real-to-complex 1D FFT produces the same output as two real-to-complex FFTs.
	 */
	void testDualRealToComplex(nnp_fft_function dual_real_fft, nnp_strided_fft_function real_fft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Two sequences as input of dual-sequence real-to-complex 1D FFT */
		std::vector<float> input(2 * fftSize());
		/* Two interleaved sequences in MKL PERM SOA format as output of dual-sequence real-to-complex 1D FFT */
		std::vector<float> output(2 * fftSize());
		/* Two deinterleaved sequences in MKL PERM AOS format as outputs of two real-to-complex 1D FFTs */
		std::vector<float> deinterleavedOutput(2 * fftSize());
		/* Two interleaved sequences as outputs of two real-to-complex 1D FFTs */
		std::vector<float> referenceOutput(2 * fftSize());

		std::vector<std::vector<float>> errors(fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), std::nanf(""));
			std::fill(deinterleavedOutput.begin(), deinterleavedOutput.end(), std::nanf(""));

			dual_real_fft(input.data(), output.data());
			real_fft(input.data(), 1, deinterleavedOutput.data(), 1);
			real_fft(input.data() + fftSize(), 1, deinterleavedOutput.data() + fftSize(), 1);

			/* Repack and interleave two real output sequences */
			for (size_t i = 0; i < fftSize() / 2; i++) {
				/* Sequence X, real component */
				referenceOutput[i * 2 + 0] = deinterleavedOutput[i * 2 + 0];
				/* Sequence X, imaginary component */
				referenceOutput[fftSize() + i * 2 + 0] = deinterleavedOutput[i * 2 + 1];
				/* Sequence H, real component */
				referenceOutput[i * 2 + 1] = deinterleavedOutput[fftSize() + i * 2 + 0];
				/* Sequence H, imaginary component */
				referenceOutput[fftSize() + i * 2 + 1] = deinterleavedOutput[fftSize() + i * 2 + 1];
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceOutput[i], output[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	/**
	 * Validates that dual-sequence complex-to-real 1D FFT produces the same output as two complex-to-real FFTs.
	 */
	void testDualComplexToReal(nnp_fft_function dual_real_ifft, nnp_strided_fft_function real_ifft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Two interleaved sequences in MKL PERM SOA format as input of dual-sequence complex-to-real 1D IFFT */
		std::vector<float> input(2 * fftSize());
		/* Two real sequences as output of dual-sequence complex-to-real 1D IFFT */
		std::vector<float> output(2 * fftSize());
		/* Two deinterleaved sequences in MKL PERM AOS format as inputs of two complex-to-real 1D IFFTs */
		std::vector<float> deinterleavedInput(2 * fftSize());
		/* Two real sequences as outputs of two complex-to-real 1D IFFTs */
		std::vector<float> referenceOutput(2 * fftSize());

		std::vector<std::vector<float>> errors(fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), std::nanf(""));
			std::fill(referenceOutput.begin(), referenceOutput.end(), std::nanf(""));

			/* Unpack and deinterleave two real input sequences */
			for (size_t i = 0; i < fftSize() / 2; i++) {
				/* Sequence X, real component */
				deinterleavedInput[i * 2 + 0] = input[i * 2 + 0];
				/* Sequence X, imaginary component */
				deinterleavedInput[i * 2 + 1] = input[fftSize() + i * 2 + 0];
				/* Sequence H, real component */
				deinterleavedInput[fftSize() + i * 2 + 0] = input[i * 2 + 1];
				/* Sequence H, imaginary component */
				deinterleavedInput[fftSize() + i * 2 + 1] = input[fftSize() + i * 2 + 1];
			}

			dual_real_ifft(input.data(), output.data());
			real_ifft(deinterleavedInput.data(), 1, referenceOutput.data(), 1);
			real_ifft(deinterleavedInput.data() + fftSize(), 1, referenceOutput.data() + fftSize(), 1);

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceOutput[i], output[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

	/**
	 * Validates that optimized complex 1D FFT produces the same output as reference implementation.
	 */
	void testOptimizedComplex(nnp_fft_function optimized_fft, nnp_strided_fft_function reference_fft) const {
		testOptimized(2, optimized_fft, reference_fft);
	}

	/**
	 * Validates that optimized real 1D FFT produces the same output as reference implementation.
	 */
	void testOptimizedReal(nnp_fft_function optimized_fft, nnp_strided_fft_function reference_fft) const {
		testOptimized(1, optimized_fft, reference_fft);
	}

	/**
	 * Validates that optimized dual-sequence real 1D FFT produces the same output as reference implementation.
	 */
	void testOptimizedDualReal(nnp_fft_function optimized_fft, nnp_fft_function reference_fft) const {
		ASSERT_EQ(1, simdWidth());
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Original sequence */
		std::vector<float, AlignedAllocator<float, 32>> data(2 * fftSize());
		/* Original sequence after forward Fourier transform */
		std::vector<float, AlignedAllocator<float, 32>> transformedData(2 * fftSize());
		/* Original sequence after forward + inverse Fourier transform */
		std::vector<float> referenceTransformedData(2 * fftSize());

		std::vector<std::vector<float>> errors(2 * fftSize());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(data.begin(), data.end(), std::ref(rng));
			std::fill(transformedData.begin(), transformedData.end(), std::nanf(""));
			std::fill(referenceTransformedData.begin(), referenceTransformedData.end(), std::nanf(""));

			optimized_fft(data.data(), transformedData.data());
			reference_fft(data.data(), referenceTransformedData.data());

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceTransformedData[i], transformedData[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

protected:
	void testOptimized(size_t elements, nnp_fft_function optimized_fft, nnp_strided_fft_function reference_fft) const {
		ASSERT_NE(fftSize(), 0);

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		/* Original sequence */
		std::vector<float, AlignedAllocator<float, 32>> data(elements * fftSize() * simdWidth());
		/* Original sequence after forward Fourier transform */
		std::vector<float, AlignedAllocator<float, 32>> transformedData(elements * fftSize() * simdWidth());
		/* Original sequence after forward + inverse Fourier transform */
		std::vector<float> referenceTransformedData(elements * fftSize() * simdWidth());

		std::vector<std::vector<float>> errors(elements * fftSize() * simdWidth());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(data.begin(), data.end(), std::ref(rng));
			std::fill(transformedData.begin(), transformedData.end(), std::nanf(""));
			std::fill(referenceTransformedData.begin(), referenceTransformedData.end(), std::nanf(""));

			optimized_fft(data.data(), transformedData.data());
			for (size_t simdOffset = 0; simdOffset < simdWidth(); simdOffset++) {
				reference_fft(data.data() + simdOffset, simdWidth(),
					referenceTransformedData.data() + simdOffset, simdWidth());
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(referenceTransformedData[i], transformedData[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit());
	}

private:
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

	size_t fftSize_;
	size_t simdWidth_;
	size_t iterations_;
	float errorLimit_;
};
