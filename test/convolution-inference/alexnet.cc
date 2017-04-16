#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/alexnet.h>

/*
 * AlexNet conv1 layer
 */

TEST(IMPLICIT_GEMM, conv1) {
	AlexNet::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm);
}

/*
 * AlexNet conv2 layer
 */

TEST(FT8x8, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16);
}

TEST(IMPLICIT_GEMM, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm);
}

/*
 * AlexNet conv3 layer
 */

TEST(FT8x8, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8);
}

TEST(IMPLICIT_GEMM, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm);
}

/*
 * AlexNet conv4 layer
 */

TEST(FT8x8, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8);
}

TEST(IMPLICIT_GEMM, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm);
}

/*
 * AlexNet conv5 layer
 */

TEST(FT8x8, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8);
}

TEST(IMPLICIT_GEMM, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
