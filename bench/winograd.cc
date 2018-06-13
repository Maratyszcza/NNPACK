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
#include <nnpack/transform.h>
#include <nnpack/AlignedAllocator.h>

#include <benchmark/benchmark.h>


template<class TransformedT>
class InputWinogradTransform : public benchmark::Fixture {
public:
	inline InputWinogradTransform() {
		cpuinfo_initialize();
		const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
		const size_t l1d_reserve = 1024;

		tiles_count_ = (l1d_size - l1d_reserve) / (tileElements() * (sizeof(float) + sizeof(TransformedT)));
	}

	virtual void SetUp(const benchmark::State&) override {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		input_.resize(tilesCount() * tileElements());
		std::generate(input_.begin(), input_.end(), std::ref(rng));
		transformedInput_.resize(tilesCount() * tileElements());
		std::fill(transformedInput_.begin(), transformedInput_.end(), std::nanf(""));
	}

	virtual void TearDown(benchmark::State& state) override {
		state.SetItemsProcessed(int64_t(state.iterations()) * tilesCount());
		state.SetBytesProcessed(int64_t(state.iterations()) * tilesCount() * (tileElements() * (sizeof(float) + sizeof(TransformedT))));
		input_.clear();
		transformedInput_.clear();
	}

	inline const float* input() const {
		return input_.data();
	}

	inline TransformedT* transformedInput() {
		return transformedInput_.data();
	}

	inline uint32_t tilesCount() const {
		return tiles_count_;
	}

	inline uint32_t tileSize() const {
		return 8;
	}

	inline uint32_t tileElements() const {
		return tileSize() * tileSize();
	}

private:
	std::vector<float> input_;
	std::vector<TransformedT, AlignedAllocator<TransformedT, 32>> transformedInput_;
	uint32_t tiles_count_;
};

template<class TransformedT>
class OutputWinogradTransform : public benchmark::Fixture {
public:
	inline OutputWinogradTransform() {
		cpuinfo_initialize();
		const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
		const size_t l1d_reserve = 1024;

		tiles_count_ = (l1d_size - l1d_reserve) / (tileElements() * (sizeof(float) + sizeof(TransformedT)));
	}

	virtual void SetUp(const benchmark::State&) override {
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(fp16_ieee_from_fp32_value, 
			std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed)));

		output_.resize(tilesCount() * tileElements());
		std::generate(output_.begin(), output_.end(), std::ref(rng));
		transformedOutput_.resize(tilesCount() * tileElements());
		std::fill(transformedOutput_.begin(), transformedOutput_.end(), std::nanf(""));
	}

	virtual void TearDown(benchmark::State& state) override {
		state.SetItemsProcessed(int64_t(state.iterations()) * tilesCount());
		state.SetBytesProcessed(int64_t(state.iterations()) * tilesCount() * (tileElements() * (sizeof(float) + sizeof(TransformedT))));
		transformedOutput_.clear();
		output_.clear();
	}

	inline const TransformedT* transformedOutput() const {
		return transformedOutput_.data();
	}

	inline float* output() {
		return output_.data();
	}

	inline uint32_t tilesCount() const {
		return tiles_count_;
	}

	inline uint32_t tileSize() const {
		return 8;
	}

	inline uint32_t tileElements() const {
		return tileSize() * tileSize();
	}

private:
	std::vector<TransformedT, AlignedAllocator<TransformedT, 32>> transformedOutput_;
	std::vector<float> output_;
	uint32_t tiles_count_;
};

#if NNP_BACKEND_ARM
	BENCHMARK_TEMPLATE_F(InputWinogradTransform, neon, float)(benchmark::State& state) {
		for (auto _ : state) {
			for (uint32_t i = 0; i < tilesCount(); i++) {
				nnp_iwt8x8_3x3_with_offset__neon(
					input() + i * tileElements(),
					transformedInput() + i * tileElements(),
					tileSize(), tileSize() * sizeof(float),
					tileSize(), tileSize(), 0, 0);
			}
		}
	}

	BENCHMARK_TEMPLATE_F(InputWinogradTransform, neonhp, uint16_t)(benchmark::State& state) {
		for (auto _ : state) {
			for (uint32_t i = 0; i < tilesCount(); i++) {
				nnp_iwt8x8_3x3_fp16_with_offset__neonhp(
					input() + i * tileElements(),
					transformedInput() + i * tileElements(),
					tileSize(), tileSize() * sizeof(uint16_t),
					tileSize(), tileSize(), 0, 0);
			}
		}
	}

	BENCHMARK_TEMPLATE_F(OutputWinogradTransform, neon, float)(benchmark::State& state) {
		for (auto _ : state) {
			for (uint32_t i = 0; i < tilesCount(); i++) {
				nnp_owt8x8_3x3__neon(
					transformedOutput() + i * tileElements(),
					output() + i * tileElements(),
					tileSize() * sizeof(float), tileSize(),
					tileSize(), tileSize(), 0, 0);
			}
		}
	}

	BENCHMARK_TEMPLATE_F(OutputWinogradTransform, neonhp, uint16_t)(benchmark::State& state) {
		for (auto _ : state) {
			for (uint32_t i = 0; i < tilesCount(); i++) {
				nnp_owt8x8_3x3_fp16__neonhp(
					transformedOutput() + i * tileElements(),
					output() + i * tileElements(),
					tileSize() * sizeof(uint16_t), tileSize(),
					tileSize(), tileSize(), 0, 0);
			}
		}
	}
#endif

BENCHMARK_MAIN();
