#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/vgg-a.h>

/*
 * VGG model A conv1 layer
 */

TEST(FT8x8, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

/*
 * VGG model A conv2 layer
 */

TEST(FT8x8, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

/*
 * VGG model A conv3 layer
 */

TEST(FT8x8, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

/*
 * VGG model A conv4 layer
 */

TEST(FT8x8, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

/*
 * VGG model A conv5 layer
 */

TEST(FT8x8, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

/*
 * VGG model A conv6 layer
 */

TEST(FT8x8, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

/*
 * VGG model A conv8 layer
 */

TEST(FT8x8, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
