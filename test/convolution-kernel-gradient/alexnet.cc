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
		.errorLimit(1.0e-6)
		.testKernelGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv2) {
	AlexNet::conv2()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testKernelGradient(nnp_convolution_algorithm_ft16x16);
}

/*
 * AlexNet conv3 layer
 */

TEST(FT8x8, conv3) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-6)
		.testKernelGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv3) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testKernelGradient(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, DISABLED_conv3) {
	AlexNet::conv3()
		.batchSize(128)
		.errorLimit(1.0e-3)
		.testKernelGradient(nnp_convolution_algorithm_wt8x8);
}

/*
 * AlexNet conv4 layer
 */

TEST(FT8x8, conv4) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-6)
		.testKernelGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv4) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testKernelGradient(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, DISABLED_conv4) {
	AlexNet::conv4()
		.batchSize(128)
		.errorLimit(1.0e-3)
		.testKernelGradient(nnp_convolution_algorithm_wt8x8);
}

/*
 * AlexNet conv5 layer
 */

TEST(FT8x8, conv5) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-6)
		.testKernelGradient(nnp_convolution_algorithm_ft8x8);
}

TEST(FT16x16, conv5) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-5)
		.testKernelGradient(nnp_convolution_algorithm_ft16x16);
}

TEST(WT8x8, DISABLED_conv5) {
	AlexNet::conv5()
		.batchSize(128)
		.errorLimit(1.0e-3)
		.testKernelGradient(nnp_convolution_algorithm_wt8x8);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
