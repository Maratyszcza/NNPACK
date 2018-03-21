#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#include <cpuinfo.h>
#include <nnpack/macros.h>
#include <nnpack/blas.h>
#include <AlignedAllocator.h>

#include <benchmark/benchmark.h>


template<uint32_t mr_, uint32_t nr_>
class SGEMM : public benchmark::Fixture {
public:
	inline SGEMM() {
		cpuinfo_initialize();
		const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size - 512;
		kc_ = (l1d_size / sizeof(float) - mr() * nr()) / (mr() + nr());
	}

	virtual void SetUp(const benchmark::State&) override {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		a_.resize(mr() * kc());
		std::generate(a_.begin(), a_.end(), std::ref(rng));
		b_.resize(nr() * kc());
		std::generate(b_.begin(), b_.end(), std::ref(rng));
		c_.resize(mr() * nr());
		std::fill(c_.begin(), c_.end(), std::nanf(""));
	}

	virtual void TearDown(benchmark::State& state) override {
		state.SetItemsProcessed(uint64_t(state.iterations()) * 2 * mr() * nr() * kc());
		a_.clear();
		b_.clear();
		c_.clear();
	}

	inline const float* a() const {
		return a_.data();
	}

	inline const float* b() const {
		return b_.data();
	}

	inline float* c() {
		return c_.data();
	}

	inline uint32_t mr() const {
		return mr_;
	}

	inline uint32_t nr() const {
		return nr_;
	}

	inline uint32_t kc() const {
		return kc_;
	}

private:
	std::vector<float, AlignedAllocator<float, 32>> a_;
	std::vector<float, AlignedAllocator<float, 32>> b_;
	std::vector<float> c_;
	uint32_t kc_;
};

#if NNP_BACKEND_X86_64
	BENCHMARK_TEMPLATE_F(SGEMM, fma3, 4, 24)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_sgemm_only_4x24__fma3(kc(), 0, a(), b(), c(), nr());
		}
	}
#endif

#if NNP_BACKEND_ARM && CPUINFO_ARCH_ARM
	BENCHMARK_TEMPLATE_F(SGEMM, aarch32_neon, 6, 8)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_sgemm_only_6x8__aarch32_neon(kc(), 0, a(), b(), c(), nr());
		}
	}
#endif

#if NNP_BACKEND_ARM
	BENCHMARK_TEMPLATE_F(SGEMM, neon, 6, 8)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_sgemm_only_6x8__neon(kc(), 0, a(), b(), c(), nr());
		}
	}
#endif

#if NNP_BACKEND_PSIMD
	BENCHMARK_TEMPLATE_F(SGEMM, psimd, 4, 8)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_sgemm_only_4x8__psimd(kc(), 0, a(), b(), c(), nr());
		}
	}
#endif

#if NNP_BACKEND_SCALAR
	BENCHMARK_TEMPLATE_F(SGEMM, scalar, 4, 3)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_sgemm_only_4x3__scalar(kc(), 0, a(), b(), c(), nr());
		}
	}
#endif

BENCHMARK_MAIN();
