#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#include <cpuinfo.h>
#include <fp16.h>
#include <nnpack/macros.h>
#include <nnpack/blas.h>
#include <nnpack/AlignedAllocator.h>

#include <benchmark/benchmark.h>


template<uint32_t xr_, uint32_t mr_, uint32_t nr_>
class HXGEMM : public benchmark::Fixture {
public:
	inline HXGEMM() {
		cpuinfo_initialize();
		const size_t l1_size = cpuinfo_get_l1d_cache(0)->size;
		const size_t l2_size = cpuinfo_get_l2_cache(0)->size;
		const size_t l1_reserve = 512;
		const size_t l2_reserve = 2048;
		kc_ = ((l1_size - l1_reserve) / sizeof(float) - xr() * mr() * nr()) / (xr() * mr() + xr() * nr());
		mc_ = ((l2_size - l2_reserve) / sizeof(float) - xr() * nr() * kc()) / (xr() * nr() + xr() * kc());
		mc_ = mc_ / mr() * mr();
	}

	virtual void SetUp(const benchmark::State&) override {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(fp16_ieee_from_fp32_value, std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed)));

		a_.resize(xr() * mc() * kc());
		std::generate(a_.begin(), a_.end(), std::ref(rng));
		b_.resize(xr() * nr() * kc());
		std::generate(b_.begin(), b_.end(), std::ref(rng));
		c_.resize(xr() * mc() * nr());
		std::fill(c_.begin(), c_.end(), std::nanf(""));
	}

	virtual void TearDown(benchmark::State& state) override {
		state.SetItemsProcessed(uint64_t(state.iterations()) * 2 * xr() * mc() * nr() * kc());
		a_.clear();
		b_.clear();
		c_.clear();
	}

	inline const uint16_t* a() const {
		return a_.data();
	}

	inline const uint16_t* b() const {
		return b_.data();
	}

	inline uint16_t* c() {
		return c_.data();
	}

	inline uint32_t xr() const {
		return xr_;
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

	inline uint32_t mc() const {
		return mc_;
	}

private:
	std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> a_;
	std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> b_;
	std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> c_;
	uint32_t kc_;
	uint32_t mc_;
};

#if NNP_BACKEND_ARM
	#if CPUINFO_ARCH_ARM
		BENCHMARK_TEMPLATE_F(HXGEMM, fast__aarch32_neonhparith, 4, 3, 3)(benchmark::State& state) {
			if (!cpuinfo_has_arm_neon_fp16_arith()) {
				state.SkipWithError("NEON-FP16 arithmetics are not supported");
			}
			for (auto _ : state) {
				for (uint32_t m = 0; m < mc(); m += mr()) {
					nnp_h4gemm_only_3x3__aarch32_neonhparith(
						kc(),
						0,
						a() + xr() * m * kc(),
						b(),
						c() + xr() * m * nr(),
						xr() * nr());
				}
			}
		}

		BENCHMARK_TEMPLATE_F(HXGEMM, fast__aarch32_neon2, 4, 3, 3)(benchmark::State& state) {
			if (!cpuinfo_has_arm_neon_fma()) {
				state.SkipWithError("NEONv2 (NEON-FMA) is not supported");
			}
			for (auto _ : state) {
				for (uint32_t m = 0; m < mc(); m += mr()) {
					nnp_h4gemm_only_3x3__aarch32_neon2(
						kc(),
						0,
						a() + xr() * m * kc(),
						b(),
						c() + xr() * m * nr(),
						xr() * nr());
				}
			}
		}


		BENCHMARK_TEMPLATE_F(HXGEMM, full__aarch32_neon2, 4, 3, 3)(benchmark::State& state) {
			if (!cpuinfo_has_arm_neon_fma()) {
				state.SkipWithError("NEONv2 (NEON-FMA) is not supported");
			}
			for (auto _ : state) {
				for (uint32_t m = 0; m < mc(); m += mr()) {
					nnp_h4gemm_upto_3x3__aarch32_neon2(
						3, 3,
						kc(),
						0,
						a() + xr() * m * kc(),
						b(),
						c() + xr() * m * nr(),
						xr() * nr());
				}
			}
		}

		BENCHMARK_TEMPLATE_F(HXGEMM, fast__aarch32_neonhp, 4, 3, 3)(benchmark::State& state) {
			if (!cpuinfo_has_arm_neon_fp16()) {
				state.SkipWithError("NEONHP (NEON-FP16) is not supported");
			}
			for (auto _ : state) {
				for (uint32_t m = 0; m < mc(); m += mr()) {
					nnp_h4gemm_only_3x3__aarch32_neonhp(
						kc(),
						0,
						a() + xr() * m * kc(),
						b(),
						c() + xr() * m * nr(),
						xr() * nr());
				}
			}
		}
	#endif /* CPUINFO_ARCH_ARM */

	BENCHMARK_TEMPLATE_F(HXGEMM, fast__neon, 4, 3, 3)(benchmark::State& state) {
		if (!cpuinfo_has_arm_neon_fp16()) {
			state.SkipWithError("NEONHP (NEON-FP16) is not supported");
		}
		for (auto _ : state) {
			for (uint32_t m = 0; m < mc(); m += mr()) {
				nnp_h4gemm_only_3x3__neonhp(
					kc(),
					0,
					a() + xr() * m * kc(),
					b(),
					c() + xr() * m * nr(),
					xr() * nr());
			}
		}
	}

	BENCHMARK_TEMPLATE_F(HXGEMM, full__neon, 4, 3, 3)(benchmark::State& state) {
		if (!cpuinfo_has_arm_neon_fp16()) {
			state.SkipWithError("NEONHP (NEON-FP16) is not supported");
		}
		for (auto _ : state) {
			for (uint32_t m = 0; m < mc(); m += mr()) {
				nnp_h4gemm_upto_3x3__neonhp(
					3, 3,
					kc(),
					0,
					a() + xr() * m * kc(),
					b(),
					c() + xr() * m * nr(),
					xr() * nr());
			}
		}
	}
#endif /* NNP_BACKEND_ARM */

BENCHMARK_MAIN();
