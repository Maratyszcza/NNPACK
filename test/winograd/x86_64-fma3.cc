#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/winograd.h>

/**
 * Test 1D input Winograd gransform F(6, 3).
 */

TEST(F6k3, input) {
	auto tester = WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	const float inputTransformMatrix[8 * 8] = {
		-36.0f,   0.0f, 49.0f,   0.0f, -14.0f,   0.0f, 1.0f, 0.0f,
		  0.0f,  36.0f, 36.0f, -13.0f, -13.0f,   1.0f, 1.0f, 0.0f,
		  0.0f, -36.0f, 36.0f,  13.0f, -13.0f,  -1.0f, 1.0f, 0.0f,
		  0.0f,  18.0f,  9.0f, -20.0f, -10.0f,   2.0f, 1.0f, 0.0f,
		  0.0f, -18.0f,  9.0f,  20.0f, -10.0f,  -2.0f, 1.0f, 0.0f,
		  0.0f,  12.0f,  4.0f, -15.0f,  -5.0f,   3.0f, 1.0f, 0.0f,
		  0.0f, -12.0f,  4.0f,  15.0f,  -5.0f,  -3.0f, 1.0f, 0.0f,
		  0.0f, -36.0f,  0.0f,  49.0f,   0.0f, -14.0f, 0.0f, 1.0f
	};
	tester.testInputTransform(nnp_iwt_f6k3__fma3, inputTransformMatrix);
}

/**
 * Test 1D kernel Winograd gransform F(6, 3).
 */

TEST(F6k3, kernel) {
	auto tester = WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	const float kernelTransformMatrix[8 * 3] = {
		-1.f/36,   0.0f,     0.f,
		 1.f/48,   1.f/48,   1.f/48,
		 1.f/48,  -1.f/48,   1.f/48,
		-1.f/120, -1.f/60,  -1.f/30,
		-1.f/120,  1.f/60,  -1.f/30,
		 1.f/720,  1.f/240,  1.f/80,
		 1.f/720, -1.f/240,  1.f/80,
		 0.f,      0.f,      1.f
	};
	tester.testKernelTransform(nnp_kwt_f6k3__fma3, kernelTransformMatrix);
}

/**
 * Test 1D output Winograd gransform F(6, 3).
 */

TEST(F6k3, output) {
	const float outputTransformMatrix[6 * 8] = {
		1.0f, 1.0f,  1.0f,  1.0f,   1.0f,   1.0f,    1.0f, 0.0f,
		0.0f, 1.0f, -1.0f,  2.0f,  -2.0f,   3.0f,   -3.0f, 0.0f,
		0.0f, 1.0f,  1.0f,  4.0f,   4.0f,   9.0f,    9.0f, 0.0f,
		0.0f, 1.0f, -1.0f,  8.0f,  -8.0f,  27.0f,  -27.0f, 0.0f,
		0.0f, 1.0f,  1.0f, 16.0f,  16.0f,  81.0f,   81.0f, 0.0f,
		0.0f, 1.0f, -1.0f, 32.0f, -32.0f, 243.0f, -243.0f, 1.0f
	};
	WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testOutputTransform(nnp_owt_f6k3__fma3, outputTransformMatrix);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
