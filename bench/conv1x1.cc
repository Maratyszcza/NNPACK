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
#include <nnpack/AlignedAllocator.h>

#include <benchmark/benchmark.h>


template<uint32_t mr_, uint32_t nr_>
class CONV1x1 : public benchmark::Fixture {
public:
	inline CONV1x1() {
		cpuinfo_initialize();
		const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
		const size_t l1d_reserve = 512;
		kc_ = ((l1d_size - l1d_reserve) / sizeof(float) - mr() * nr()) / (mr() + nr());
	}

	virtual void SetUp(const benchmark::State&) override {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		i_.resize(mr() * kc());
		std::generate(i_.begin(), i_.end(), std::ref(rng));
		k_.resize(mr() * kc() + nr());
		std::fill(k_.begin(), k_.end(), std::nanf(""));
		o_.resize(nr() * kc());
		std::generate(o_.begin(), o_.end(), std::ref(rng));
	}

	virtual void TearDown(benchmark::State& state) override {
		state.SetItemsProcessed(uint64_t(state.iterations()) * 2 * mr() * nr() * kc());
		i_.clear();
		k_.clear();
		o_.clear();
	}

	inline const float* i() const {
		return i_.data();
	}

	inline const float* k() const {
		return k_.data();
	}

	inline float* o() {
		return o_.data();
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
	std::vector<float> i_;
	std::vector<float> k_;
	std::vector<float> o_;
	uint32_t kc_;
};

#if NNP_BACKEND_X86_64
	BENCHMARK_TEMPLATE_F(CONV1x1, fast__neon, 2, 4)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_conv1x1_only_2x4__fma3(mr(), kc(), i(), k(), o());
		}
	}
#endif

#if NNP_BACKEND_ARM
	BENCHMARK_TEMPLATE_F(CONV1x1, fast__neon, 4, 4)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_conv1x1_only_4x4__neon(mr(), kc(), i(), k(), o());
		}
	}
#endif

#if NNP_BACKEND_PSIMD
	BENCHMARK_TEMPLATE_F(CONV1x1, psimd, 2, 8)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_conv1x1_only_2x4__psimd(mr(), kc(), i(), k(), o());
		}
	}
#endif

#if NNP_BACKEND_SCALAR
	BENCHMARK_TEMPLATE_F(CONV1x1, scalar, 2, 4)(benchmark::State& state) {
		for (auto _ : state) {
			nnp_conv1x1_only_2x4__scalar(mr(), kc(), i(), k(), o());
		}
	}
#endif

BENCHMARK_MAIN();
