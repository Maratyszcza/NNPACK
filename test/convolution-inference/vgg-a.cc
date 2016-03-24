#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/vgg-a.h>

/*
 * VGG model A conv1 layer
 */

TEST(FT8x8_RECOMPUTE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * VGG model A conv2 layer
 */

TEST(FT8x8_RECOMPUTE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * VGG model A conv3 layer
 */

TEST(FT8x8_RECOMPUTE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * VGG model A conv4 layer
 */

TEST(FT8x8_RECOMPUTE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * VGG model A conv5 layer
 */

TEST(FT8x8_RECOMPUTE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * VGG model A conv6 layer
 */

TEST(FT8x8_RECOMPUTE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

/*
 * VGG model A conv8 layer
 */

TEST(FT8x8_RECOMPUTE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT8x8_REUSE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(FT16x16_RECOMPUTE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(FT16x16_REUSE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_kernel_transform_strategy_reuse);
}

TEST(WT8x8_RECOMPUTE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_recompute);
}

TEST(WT8x8_REUSE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-3)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_kernel_transform_strategy_reuse);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
