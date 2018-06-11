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

#include <fp16.h>

#include <nnpack/hwinfo.h>
#include <nnpack/AlignedAllocator.h>

class GemmMicroKernelTester {
public:
	inline GemmMicroKernelTester() :
		mr_(1),
		nr_(1),
		kc_(1),
		simdWidth_(1),
		accumulateC_(false),
		iterations_(1000),
		errorLimit_(1.0e-5)
	{
	}

	inline GemmMicroKernelTester& mr(size_t mr) {
		this->mr_ = mr;
		return *this;
	}

	inline size_t mr() const {
		return this->mr_;
	}

	inline GemmMicroKernelTester& nr(size_t nr) {
		this->nr_ = nr;
		return *this;
	}

	inline size_t nr() const {
		return this->nr_;
	}

	inline GemmMicroKernelTester& kc(size_t kc) {
		this->kc_ = kc;
		return *this;
	}

	inline size_t kc() const {
		return this->kc_;
	}

	inline GemmMicroKernelTester& simdWidth(size_t simdWidth) {
		this->simdWidth_ = simdWidth;
		return *this;
	}

	inline size_t simdWidth() const {
		return this->simdWidth_;
	}

	/* NR stride is NR rounded up to SIMD width */
	inline size_t nrStride(size_t nr) const {
		return (nr + simdWidth() - 1) / simdWidth() * simdWidth();
	}

	inline GemmMicroKernelTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline GemmMicroKernelTester& accumulateC(bool value) {
		this->accumulateC_ = value;
		return *this;
	}

	inline bool accumulateC() const {
		return this->accumulateC_;
	}

	inline GemmMicroKernelTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	void testSXGEMM(nnp_fast_tuple_gemm_function fast_sxgemm) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float, AlignedAllocator<float, 32>> a(simdWidth() * mr() * kc());
		std::vector<float, AlignedAllocator<float, 32>> b(simdWidth() * nr() * kc());
		std::vector<float, AlignedAllocator<float, 32>> c(simdWidth() * mr() * nr());
		std::vector<float> cReference(simdWidth() * mr() * nr());

		std::vector<std::vector<float>> errors(simdWidth() * mr() * nr());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(a.begin(), a.end(), std::ref(rng));
			std::generate(b.begin(), b.end(), std::ref(rng));
			if (accumulateC()) {
				std::generate(c.begin(), c.end(), std::ref(rng));
				std::copy(c.cbegin(), c.cend(), cReference.begin());
			} else {
				std::fill(c.begin(), c.end(), std::nanf(""));
				std::fill(cReference.begin(), cReference.end(), 0.0f);
			}

			fast_sxgemm(kc(), accumulateC(), a.data(), b.data(), c.data(), simdWidth() * nr());

			for (size_t k = 0; k < kc(); k++) {
				for (size_t m = 0; m < mr(); m++) {
					for (size_t n = 0; n < nr(); n++) {
						for (size_t i = 0; i < simdWidth(); i++) {
							cReference[(m * nr() + n) * simdWidth() + i] +=
								a[(k * mr() + m) * simdWidth() + i] * b[(k * nr() + n) * simdWidth() + i];
						}
					}
				}
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(cReference[i], c[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit()) <<
			"Mr x Nr = " << mr() << " x " << nr() << ", Kc = " << kc() << ", SIMD width = " << simdWidth();
	}

	void testSXGEMM(nnp_full_tuple_gemm_function full_sxgemm) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		for (uint32_t mr = 1; mr <= this->mr(); mr++) {
			for (uint32_t nr = 1; nr <= this->nr(); nr++) {
				std::vector<float, AlignedAllocator<float, 32>> a(simdWidth() * mr * kc());
				std::vector<float, AlignedAllocator<float, 32>> b(simdWidth() * nr * kc());
				std::vector<float, AlignedAllocator<float, 32>> c(simdWidth() * mr * nr);
				std::vector<float> cReference(simdWidth() * mr * nr);

				std::vector<std::vector<float>> errors(simdWidth() * mr * nr);
				for (size_t iteration = 0; iteration < iterations(); iteration++) {
					std::generate(a.begin(), a.end(), std::ref(rng));
					std::generate(b.begin(), b.end(), std::ref(rng));
					if (accumulateC()) {
						std::generate(c.begin(), c.end(), std::ref(rng));
						std::copy(c.cbegin(), c.cend(), cReference.begin());
					} else {
						std::fill(c.begin(), c.end(), std::nanf(""));
						std::fill(cReference.begin(), cReference.end(), 0.0f);
					}

					full_sxgemm(mr, nr, kc(), accumulateC(), a.data(), b.data(), c.data(), simdWidth() * nr);

					for (size_t k = 0; k < kc(); k++) {
						for (size_t m = 0; m < mr; m++) {
							for (size_t n = 0; n < nr; n++) {
								for (size_t i = 0; i < simdWidth(); i++) {
									cReference[(m * nr + n) * simdWidth() + i] +=
										a[(k * mr + m) * simdWidth() + i] * b[(k * nr + n) * simdWidth() + i];
								}
							}
						}
					}

					for (size_t i = 0; i < errors.size(); i++) {
						errors[i].push_back(relativeError(cReference[i], c[i]));
					}
				}

				const std::vector<float> medianErrors = median(errors);
				const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
				ASSERT_LT(maxMedianError, errorLimit()) <<
					"Mr x Nr = " << mr << " x " << nr << ", Kc = " << kc() << ", SIMD width = " << simdWidth();
			}
		}
	}

	void testHXGEMM(nnp_fast_tuple_gemm_function fast_hxgemm) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(fp16_ieee_from_fp32_value, std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed)));

		std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> a(simdWidth() * mr() * kc());
		std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> b(simdWidth() * nr() * kc());
		std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> c(simdWidth() * mr() * nr());
		std::vector<float> cReference(simdWidth() * mr() * nr());

		std::vector<std::vector<float>> errors(simdWidth() * mr() * nr());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(a.begin(), a.end(), std::ref(rng));
			std::generate(b.begin(), b.end(), std::ref(rng));
			if (accumulateC()) {
				std::generate(c.begin(), c.end(), std::ref(rng));
				std::transform(c.cbegin(), c.cend(), cReference.begin(), fp16_ieee_to_fp32_value);
			} else {
				std::fill(c.begin(), c.end(), 0x7C00);
				std::fill(cReference.begin(), cReference.end(), 0.0f);
			}

			fast_hxgemm(kc(), accumulateC(), a.data(), b.data(), c.data(), simdWidth() * nr());

			for (size_t k = 0; k < kc(); k++) {
				for (size_t m = 0; m < mr(); m++) {
					for (size_t n = 0; n < nr(); n++) {
						for (size_t i = 0; i < simdWidth(); i++) {
							cReference[(m * nr() + n) * simdWidth() + i] +=
								fp16_ieee_to_fp32_value(a[(k * mr() + m) * simdWidth() + i]) *
								fp16_ieee_to_fp32_value(b[(k * nr() + n) * simdWidth() + i]);
						}
					}
				}
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(cReference[i], fp16_ieee_to_fp32_value(c[i])));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit()) <<
			"Mr x Nr = " << mr() << " x " << nr() << ", Kc = " << kc() << ", SIMD width = " << simdWidth();
	}

	void testHXGEMM(nnp_full_tuple_gemm_function full_hxgemm) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(fp16_ieee_from_fp32_value, std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed)));

		for (uint32_t mr = 1; mr <= this->mr(); mr++) {
			for (uint32_t nr = 1; nr <= this->nr(); nr++) {
				std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> a(simdWidth() * mr * kc());
				std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> b(simdWidth() * nr * kc());
				std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> c(simdWidth() * mr * nr);
				std::vector<float> cReference(simdWidth() * mr * nr);

				std::vector<std::vector<float>> errors(simdWidth() * mr * nr);
				for (size_t iteration = 0; iteration < iterations(); iteration++) {
					std::generate(a.begin(), a.end(), std::ref(rng));
					std::generate(b.begin(), b.end(), std::ref(rng));
					if (accumulateC()) {
						std::generate(c.begin(), c.end(), std::ref(rng));
						std::transform(c.cbegin(), c.cend(), cReference.begin(), fp16_ieee_to_fp32_value);
					} else {
						std::fill(c.begin(), c.end(), 0x7C00);
						std::fill(cReference.begin(), cReference.end(), 0.0f);
					}

					full_hxgemm(mr, nr, kc(), accumulateC(), a.data(), b.data(), c.data(), simdWidth() * nr);

					for (size_t k = 0; k < kc(); k++) {
						for (size_t m = 0; m < mr; m++) {
							for (size_t n = 0; n < nr; n++) {
								for (size_t i = 0; i < simdWidth(); i++) {
									cReference[(m * nr + n) * simdWidth() + i] +=
										fp16_ieee_to_fp32_value(a[(k * mr + m) * simdWidth() + i]) *
										fp16_ieee_to_fp32_value(b[(k * nr + n) * simdWidth() + i]);
								}
							}
						}
					}

					for (size_t i = 0; i < errors.size(); i++) {
						errors[i].push_back(relativeError(cReference[i], fp16_ieee_to_fp32_value(c[i])));
					}
				}

				const std::vector<float> medianErrors = median(errors);
				const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
				ASSERT_LT(maxMedianError, errorLimit()) <<
					"Mr x Nr = " << mr << " x " << nr << ", Kc = " << kc() << ", SIMD width = " << simdWidth();
			}
		}
	}

	void testSGEMM(nnp_fast_sgemm_function fast_sgemm) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float, AlignedAllocator<float, 32>> a(mr() * kc());
		std::vector<float, AlignedAllocator<float, 32>> b(nr() * kc());
		std::vector<float> c(mr() * nr());
		std::vector<float> cReference(mr() * nr());

		std::vector<std::vector<float>> errors(mr() * nr());
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(a.begin(), a.end(), std::ref(rng));
			std::generate(b.begin(), b.end(), std::ref(rng));
			if (accumulateC()) {
				std::generate(c.begin(), c.end(), std::ref(rng));
				std::copy(c.cbegin(), c.cend(), cReference.begin());
			} else {
				std::fill(c.begin(), c.end(), std::nanf(""));
				std::fill(cReference.begin(), cReference.end(), 0.0f);
			}

			fast_sgemm(kc(), accumulateC(), a.data(), b.data(), c.data(), nr());

			for (size_t k = 0; k < kc(); k++) {
				for (size_t m = 0; m < mr(); m++) {
					for (size_t n = 0; n < nr(); n++) {
						cReference[m * nr() + n] += a[k * mr() + m] * b[k * nr() + n];
					}
				}
			}

			for (size_t i = 0; i < errors.size(); i++) {
				errors[i].push_back(relativeError(cReference[i], c[i]));
			}
		}

		const std::vector<float> medianErrors = median(errors);
		const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
		ASSERT_LT(maxMedianError, errorLimit()) <<
			"Mr x Nr = " << mr() << " x " << nr() << ", Kc = " << kc();
	}

	void testSGEMM(nnp_full_sgemm_function full_sgemm) const {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		for (uint32_t mr = 1; mr < this->mr(); mr++) {
			for (uint32_t nr = 1; nr < this->nr(); nr++) {
				std::vector<float, AlignedAllocator<float, 32>> a(mr * kc());
				std::vector<float, AlignedAllocator<float, 32>> b(nrStride(nr) * kc());
				std::vector<float> c(mr * nr);
				std::vector<float> cReference(mr * nr);

				std::vector<std::vector<float>> errors(mr * nr);
				for (size_t iteration = 0; iteration < iterations(); iteration++) {
					std::generate(a.begin(), a.end(), std::ref(rng));
					std::generate(b.begin(), b.end(), std::ref(rng));
					if (accumulateC()) {
						std::generate(c.begin(), c.end(), std::ref(rng));
						std::copy(c.cbegin(), c.cend(), cReference.begin());
					} else {
						std::fill(c.begin(), c.end(), std::nanf(""));
						std::fill(cReference.begin(), cReference.end(), 0.0f);
					}

					full_sgemm(mr, nr, kc(), accumulateC(), a.data(), b.data(), c.data(), nr);

					for (size_t k = 0; k < kc(); k++) {
						for (size_t m = 0; m < mr; m++) {
							for (size_t n = 0; n < nr; n++) {
								cReference[m * nr + n] += a[k * mr + m] * b[k * nrStride(nr) + n];
							}
						}
					}

					for (size_t i = 0; i < errors.size(); i++) {
						errors[i].push_back(relativeError(cReference[i], c[i]));
					}
				}

				const std::vector<float> medianErrors = median(errors);
				const float maxMedianError = *std::max_element(medianErrors.cbegin(), medianErrors.cend());
				ASSERT_LT(maxMedianError, errorLimit()) <<
					"Mr x Nr = " << mr << " x " << nr << ", Kc = " << kc();
			}
		}
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

	size_t mr_;
	size_t nr_;
	size_t kc_;
	size_t simdWidth_;
	bool accumulateC_;
	size_t iterations_;
	float errorLimit_;
};
