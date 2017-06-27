#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/alexnet.h>

/*
 * AlexNet conv2 layer
 */

TEST(FT8x8, conv2) {
	AlexNet::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv2_with_relu) {
	AlexNet::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv2) {
	AlexNet::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv2_with_relu) {
	AlexNet::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

/*
 * AlexNet conv3 layer
 */

TEST(FT8x8, conv3) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv3_with_relu) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv3) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv3_with_relu) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv3) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv3_with_relu) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * AlexNet conv4 layer
 */

TEST(FT8x8, conv4) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv4_with_relu) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv4) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv4_with_relu) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv4) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv4_with_relu) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

/*
 * AlexNet conv5 layer
 */

TEST(FT8x8, conv5) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8);
}

TEST(FT8x8, conv5_with_relu) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv5) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16);
}

TEST(FT16x16, conv5_with_relu) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv5) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testOutput(nnp_convolution_algorithm_wt8x8);
}

TEST(WT8x8, conv5_with_relu) {
	AlexNet::conv5()
		.batchSize(128)
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
