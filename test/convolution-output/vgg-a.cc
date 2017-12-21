#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/vgg-a.h>

/*
 * VGG model A conv1 layer
 */

TEST(FT8x8, conv1) {
	VGG_A::conv1()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv1_with_relu) {
	VGG_A::conv1()
		.batchSize(64)
		.errorLimit(3.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv1) {
	VGG_A::conv1()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv1_with_relu) {
	VGG_A::conv1()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv1) {
	VGG_A::conv1()
		.batchSize(64)
		.errorLimit(3.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv1_with_relu) {
	VGG_A::conv1()
		.batchSize(64)
		.errorLimit(3.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * VGG model A conv2 layer
 */

TEST(FT8x8, conv2) {
	VGG_A::conv2()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv2_with_relu) {
	VGG_A::conv2()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv2) {
	VGG_A::conv2()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv2_with_relu) {
	VGG_A::conv2()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv2) {
	VGG_A::conv2()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv2_with_relu) {
	VGG_A::conv2()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * VGG model A conv3 layer
 */

TEST(FT8x8, conv3) {
	VGG_A::conv3()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv3_with_relu) {
	VGG_A::conv3()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv3) {
	VGG_A::conv3()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv3_with_relu) {
	VGG_A::conv3()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv3) {
	VGG_A::conv3()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv3_with_relu) {
	VGG_A::conv3()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * VGG model A conv4 layer
 */

TEST(FT8x8, conv4) {
	VGG_A::conv4()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv4_with_relu) {
	VGG_A::conv4()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv4) {
	VGG_A::conv4()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv4_with_relu) {
	VGG_A::conv4()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv4) {
	VGG_A::conv4()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv4_with_relu) {
	VGG_A::conv4()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * VGG model A conv5 layer
 */

TEST(FT8x8, conv5) {
	VGG_A::conv5()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv5_with_relu) {
	VGG_A::conv5()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv5) {
	VGG_A::conv5()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv5_with_relu) {
	VGG_A::conv5()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv5) {
	VGG_A::conv5()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv5_with_relu) {
	VGG_A::conv5()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * VGG model A conv6 layer
 */

TEST(FT8x8, conv6) {
	VGG_A::conv6()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv6_with_relu) {
	VGG_A::conv6()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv6) {
	VGG_A::conv6()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv6_with_relu) {
	VGG_A::conv6()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv6) {
	VGG_A::conv6()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv6_with_relu) {
	VGG_A::conv6()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * VGG model A conv8 layer
 */

TEST(FT8x8, conv8) {
	VGG_A::conv8()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv8_with_relu) {
	VGG_A::conv8()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv8) {
	VGG_A::conv8()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv8_with_relu) {
	VGG_A::conv8()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv8) {
	VGG_A::conv8()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv8_with_relu) {
	VGG_A::conv8()
		.batchSize(64)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
