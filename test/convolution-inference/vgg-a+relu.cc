#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/vgg-a.h>

/*
 * VGG model A conv1 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv1) {
	VGG_A::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * VGG model A conv2 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv2) {
	VGG_A::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * VGG model A conv3 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv3) {
	VGG_A::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * VGG model A conv4 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv4) {
	VGG_A::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * VGG model A conv5 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv5) {
	VGG_A::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * VGG model A conv6 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv6) {
	VGG_A::conv6()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * VGG model A conv8 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv8) {
	VGG_A::conv8()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
