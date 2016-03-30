#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/alexnet.h>

/*
 * AlexNet conv2 layer
 */

TEST(FT8x8_RECOMPUTE, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * AlexNet conv3 layer
 */

TEST(FT8x8_RECOMPUTE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * AlexNet conv4 layer
 */

TEST(FT8x8_RECOMPUTE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * AlexNet conv5 layer
 */

TEST(FT8x8_RECOMPUTE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
