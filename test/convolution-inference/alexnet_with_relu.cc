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
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * AlexNet conv2 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv2) {
	AlexNet::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * AlexNet conv3 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv3) {
	AlexNet::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * AlexNet conv4 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv4) {
	AlexNet::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

/*
 * AlexNet conv5 layer
 */

TEST(FT8x8_BLOCK, DISABLED_conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT8x8_TUPLE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(FT16x16_BLOCK, DISABLED_conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(FT16x16_TUPLE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(WT8x8_BLOCK, DISABLED_conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_block_based,
			nnp_activation_relu);
}

TEST(WT8x8_TUPLE, conv5) {
	AlexNet::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_convolution_transform_strategy_tuple_based,
			nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv5) {
	AlexNet::conv5()
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
