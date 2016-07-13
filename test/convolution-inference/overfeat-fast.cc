#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/overfeat-fast.h>

/*
 * OverFeat (Fast model) conv1 layer
 */

TEST(IMPLICIT_GEMM, conv1) {
	OverFeat_Fast::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based);
}

/*
 * OverFeat (Fast model) conv2 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(FT8x8_TUPLE, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(FT16x16_BLOCK, DISABLED_conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based);
}

TEST(FT16x16_TUPLE, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based);
}

TEST(IMPLICIT_GEMM, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based);
}

/*
 * OverFeat (Fast model) conv3 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(FT8x8_TUPLE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(FT16x16_BLOCK, DISABLED_conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based);
}

TEST(FT16x16_TUPLE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based);
}

TEST(WT8x8_BLOCK, DISABLED_conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(WT8x8_TUPLE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(IMPLICIT_GEMM, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based);
}

/*
 * OverFeat (Fast model) conv4 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(FT8x8_TUPLE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(FT16x16_BLOCK, DISABLED_conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based);
}

TEST(FT16x16_TUPLE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based);
}

TEST(WT8x8_BLOCK, DISABLED_conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(WT8x8_TUPLE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(IMPLICIT_GEMM, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based);
}

/*
 * OverFeat (Fast model) conv5 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(FT8x8_TUPLE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(FT16x16_BLOCK, DISABLED_conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based);
}

TEST(FT16x16_TUPLE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based);
}

TEST(WT8x8_BLOCK, DISABLED_conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based);
}

TEST(WT8x8_TUPLE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based);
}

TEST(IMPLICIT_GEMM, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
