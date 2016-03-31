#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>
#include <models/vgg-a.h>

/*
 * VGG model A conv1 ReLU layer
 */

TEST(OUT_OF_PLACE, conv1_relu) {
	VGG_A::conv1_relu()
		.batchSize(64)
		.testInputGradient();
}

/*
 * VGG model A conv1 ReLU layer
 */

TEST(OUT_OF_PLACE, conv2_relu) {
	VGG_A::conv2_relu()
		.batchSize(64)
		.testInputGradient();
}

/*
 * VGG model A conv3 ReLU layer
 */

TEST(OUT_OF_PLACE, conv3_relu) {
	VGG_A::conv3_relu()
		.batchSize(64)
		.testInputGradient();
}

/*
 * VGG model A conv5 ReLU layer
 */

TEST(OUT_OF_PLACE, conv5_relu) {
	VGG_A::conv5_relu()
		.batchSize(64)
		.testInputGradient();
}

/*
 * VGG model A conv8 ReLU layer
 */

TEST(OUT_OF_PLACE, conv8_relu) {
	VGG_A::conv8_relu()
		.batchSize(64)
		.testInputGradient();
}

/*
 * VGG model A fc6 ReLU layer
 */

TEST(OUT_OF_PLACE, fc6_relu) {
	VGG_A::fc6_relu()
		.batchSize(64)
		.testInputGradient();
}

/*
 * VGG model A fc8 ReLU layer
 */

TEST(OUT_OF_PLACE, fc8_relu) {
	VGG_A::fc8_relu()
		.batchSize(64)
		.testInputGradient();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
