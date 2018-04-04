#include <vector>

#include <nnpack.h>
#include <nnpack/AlignedAllocator.h>

#include <benchmark/benchmark.h>


static void ConvolutionSetup(benchmark::internal::Benchmark* benchmark) {
	benchmark->Unit(benchmark::kMicrosecond)->ArgNames({"Cin", "Cout", "ImageSize"});
}

class NNPACK : public benchmark::Fixture {
	virtual void SetUp(const benchmark::State&) override {
		const auto status = nnp_initialize();
		assert(status == nnp_status_success);
	}

	virtual void TearDown(const benchmark::State&) override {
		const auto status = nnp_deinitialize();
		assert(status == nnp_status_success);
	}
};

BENCHMARK_DEFINE_F(NNPACK, conv1x1)(benchmark::State& state) {
	const size_t inputChannels  = static_cast<size_t>(state.range(0));
	const size_t outputChannels = static_cast<size_t>(state.range(1));
	const size_t imageSize      = static_cast<size_t>(state.range(2));

	std::vector<float> input, kernel, output, bias;
	std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> transformedKernel, workspaceBuffer;
	input.resize(inputChannels * imageSize * imageSize);
	kernel.resize(outputChannels * inputChannels);
	bias.resize(outputChannels);
	output.resize(outputChannels * imageSize * imageSize);

	nnp_convolution_transform_strategy strategy = nnp_convolution_transform_strategy_precompute;
	const nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_implicit_gemm;
	const nnp_size imageSize2D = { imageSize, imageSize };
	const nnp_size kernelSize2D = { 1, 1 };
	const nnp_size outputStride2D = { 1, 1 };
	const nnp_padding imagePadding = { 0, 0, 0, 0 };

	if (strategy == nnp_convolution_transform_strategy_precompute) {
		size_t transformedKernelSize = 0;
		nnp_status status = nnp_convolution_inference(
			algorithm, nnp_convolution_transform_strategy_precompute,
			inputChannels, outputChannels,
			imageSize2D, imagePadding, kernelSize2D, outputStride2D,
			NULL, NULL, NULL, NULL, NULL, &transformedKernelSize,
			nnp_activation_identity, NULL,
			NULL, NULL);
		if (status == nnp_status_success) {
			transformedKernel.resize(transformedKernelSize);
			status = nnp_convolution_inference(
				algorithm, nnp_convolution_transform_strategy_precompute,
				inputChannels, outputChannels,
				imageSize2D, imagePadding, kernelSize2D, outputStride2D,
				NULL, kernel.data(), NULL, NULL, transformedKernel.data(), &transformedKernelSize,
				nnp_activation_identity, NULL,
				NULL, NULL);
			assert(status == nnp_status_success);
			strategy = nnp_convolution_transform_strategy_reuse;
		} else {
			assert(status == nnp_status_unsupported_transform_strategy);
			strategy = nnp_convolution_transform_strategy_compute;
		}
	}

	size_t workspaceSize = 0;
	nnp_status status = nnp_convolution_inference(
		algorithm, strategy,
		inputChannels, outputChannels,
		imageSize2D, imagePadding, kernelSize2D, outputStride2D,
		NULL, NULL, NULL, NULL, NULL, &workspaceSize,
		nnp_activation_identity, NULL,
		NULL, NULL);
	assert(status == nnp_status_success);
	workspaceBuffer.resize(workspaceSize);

	double input_transform_share = 0.0, kernel_transform_share = 0.0, output_transform_share = 0.0, matmul_share = 0.0;
	for (auto _ : state) {
		nnp_profile profile;
		status = nnp_convolution_inference(
			algorithm, strategy,
			inputChannels, outputChannels,
			imageSize2D, imagePadding, kernelSize2D, outputStride2D,
			input.data(),
			transformedKernel.empty() ? kernel.data() : static_cast<float*>(static_cast<void*>(transformedKernel.data())),
			bias.data(), output.data(),
			workspaceBuffer.data(), &workspaceSize,
			nnp_activation_identity, NULL,
			NULL, &profile);
		assert(status == nnp_status_success);

		input_transform_share += profile.input_transform;
		kernel_transform_share += profile.kernel_transform;
		output_transform_share += profile.output_transform;
		matmul_share += profile.block_multiplication;
	}
	state.counters["Ti"] = benchmark::Counter(input_transform_share, benchmark::Counter::kIsRate);
	state.counters["Tk"] = benchmark::Counter(kernel_transform_share, benchmark::Counter::kIsRate);
	state.counters["To"] = benchmark::Counter(output_transform_share, benchmark::Counter::kIsRate);
	state.counters["MM"] = benchmark::Counter(matmul_share, benchmark::Counter::kIsRate);

	state.SetItemsProcessed(state.iterations() * imageSize * imageSize * inputChannels * outputChannels);
}

BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024, 1024, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024,  512, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024,  256, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512, 1024, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512,  512, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512,  256, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256, 1024, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256,  512, 16});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256,  256, 16});

BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024, 1024, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024,  512, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024,  256, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512, 1024, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512,  512, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512,  256, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256, 1024, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256,  512, 26});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256,  256, 26});

BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024, 1024, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024,  512, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({1024,  256, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512, 1024, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512,  512, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 512,  256, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256, 1024, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256,  512, 52});
BENCHMARK_REGISTER_F(NNPACK, conv1x1)->Apply(ConvolutionSetup)->Args({ 256,  256, 52});

BENCHMARK_MAIN();
