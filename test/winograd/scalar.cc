#include <gtest/gtest.h>

#include <testers/winograd.h>

/**
 * Test 1D input Winograd gransform F(6, 3).
 */

TEST(F6k3, input) {
	auto tester = WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(1)
		.errorLimit(1.0e-6f);
	const float inputTransformMatrix[8 * 8] = {
		1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f,
		0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f,
		0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f,
		0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f,
		0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f,
		0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f,
		0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f,
		0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f
	};
	tester.testInputTransform(nnp_iwt_f6k3__scalar, inputTransformMatrix);
}

/**
 * Test 2D input Winograd gransform F(6x6, 3x3).
 */

TEST(F6x6_3x3, DISABLED_input) {
	auto tester = WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(1)
		.errorLimit(1.0e-6f);
	const float inputTransformMatrix[8 * 8] = {
		1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f,
		0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f,
		0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f,
		0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f,
		0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f,
		0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f,
		0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f,
		0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f
	};
	tester.testInputTransform2D(nnp_transform_2d_with_offset(nnp_iwt8x8_3x3_with_offset__scalar), inputTransformMatrix);
}

/**
 * Test 1D kernel Winograd gransform F(6, 3).
 */

TEST(F6k3, kernel) {
	auto tester = WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(1)
		.errorLimit(1.0e-6f);
	const float kernelTransformMatrix[8 * 3] = {
		 1.0f,     0.0f,     0.0f,
		-2.0f/9,  -2.0f/9,  -2.0f/9,
		-2.0f/9,   2.0f/9,  -2.0f/9,
		 1.0f/90,  1.0f/45,  2.0f/45,
		 1.0f/90, -1.0f/45,  2.0f/45,
		 1.0f/45,  1.0f/90,  1.0f/180,
		 1.0f/45, -1.0f/90,  1.0f/180,
		 0.0f,     0.0f,     1.0f
	};
	tester.testKernelTransform(nnp_kwt_f6k3__scalar, kernelTransformMatrix);
}

/**
 * Test 2D kernel Winograd gransform F(6x6, 3x3).
 */

TEST(F6x6_3x3, DISABLED_kernel) {
	auto tester = WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(1)
		.errorLimit(1.0e-6f);
	const float kernelTransformMatrix[8 * 3] = {
		 1.0f,     0.0f,     0.0f,
		-2.0f/9,  -2.0f/9,  -2.0f/9,
		-2.0f/9,   2.0f/9,  -2.0f/9,
		 1.0f/90,  1.0f/45,  2.0f/45,
		 1.0f/90, -1.0f/45,  2.0f/45,
		 1.0f/45,  1.0f/90,  1.0f/180,
		 1.0f/45, -1.0f/90,  1.0f/180,
		 0.0f,     0.0f,     1.0f
	};
	tester.testKernelTransform2D(nnp_transform_2d_with_offset(nnp_kwt8x8_3x3__scalar), kernelTransformMatrix);
}

/**
 * Test 1D output Winograd gransform F(6, 3).
 */

TEST(F6k3, output) {
	const float outputTransformMatrix[6 * 8] = {
		1.0f, 1.0f,  1.0f,  1.0f,   1.0f,  32.0f,   32.0f, 0.0f,
		0.0f, 1.0f, -1.0f,  2.0f,  -2.0f,  16.0f,  -16.0f, 0.0f,
		0.0f, 1.0f,  1.0f,  4.0f,   4.0f,   8.0f,    8.0f, 0.0f,
		0.0f, 1.0f, -1.0f,  8.0f,  -8.0f,   4.0f,   -4.0f, 0.0f,
		0.0f, 1.0f,  1.0f, 16.0f,  16.0f,   2.0f,    2.0f, 0.0f,
		0.0f, 1.0f, -1.0f, 32.0f, -32.0f,   1.0f,   -1.0f, 1.0f
	};
	WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testOutputTransform(nnp_owt_f6k3__scalar, outputTransformMatrix);
}

/**
 * Test 2D output Winograd gransform F(6x6, 3x3).
 */

TEST(F6x6_3x3, DISABLED_output) {
	const float outputTransformMatrix[6 * 8] = {
		1.0f, 1.0f,  1.0f,  1.0f,   1.0f,  32.0f,   32.0f, 0.0f,
		0.0f, 1.0f, -1.0f,  2.0f,  -2.0f,  16.0f,  -16.0f, 0.0f,
		0.0f, 1.0f,  1.0f,  4.0f,   4.0f,   8.0f,    8.0f, 0.0f,
		0.0f, 1.0f, -1.0f,  8.0f,  -8.0f,   4.0f,   -4.0f, 0.0f,
		0.0f, 1.0f,  1.0f, 16.0f,  16.0f,   2.0f,    2.0f, 0.0f,
		0.0f, 1.0f, -1.0f, 32.0f, -32.0f,   1.0f,   -1.0f, 1.0f
	};
	WinogradTransformTester()
		.kernelSize(3)
		.outputSize(6)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testOutputTransform2D(nnp_transform_2d_with_offset(nnp_owt8x8_3x3__scalar), outputTransformMatrix);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
