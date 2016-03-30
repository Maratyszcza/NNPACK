#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/overfeat-fast.h>

/*
 * OverFeat (Fast model) conv2 layer
 */

TEST(FT8x8, conv2) {
	OverFeat_Fast::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv2) {
	OverFeat_Fast::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft16x16);
}

/*
 * OverFeat (Fast model) conv3 layer
 */

TEST(FT8x8, conv3) {
	OverFeat_Fast::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv3) {
	OverFeat_Fast::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, conv3) {
	OverFeat_Fast::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_wt8x8);
}

/*
 * OverFeat (Fast model) conv4 layer
 */

TEST(FT8x8, conv4) {
	OverFeat_Fast::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv4) {
	OverFeat_Fast::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, conv4) {
	OverFeat_Fast::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_wt8x8);
}

/*
 * OverFeat (Fast model) conv5 layer
 */

TEST(FT8x8, conv5) {
	OverFeat_Fast::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv5) {
	OverFeat_Fast::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, conv5) {
	OverFeat_Fast::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testInputGradient(nnp_convolution_algorithm_wt8x8);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
