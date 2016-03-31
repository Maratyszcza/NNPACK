#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>
#include <models/overfeat-fast.h>

/*
 * OverFeat (Fast model) conv1 ReLU layer
 */

TEST(OUT_OF_PLACE, conv1_relu) {
	OverFeat_Fast::conv1_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * OverFeat (Fast model) conv1 ReLU layer
 */

TEST(OUT_OF_PLACE, conv2_relu) {
	OverFeat_Fast::conv2_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * OverFeat (Fast model) conv3 ReLU layer
 */

TEST(OUT_OF_PLACE, conv3_relu) {
	OverFeat_Fast::conv3_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * OverFeat (Fast model) conv4 ReLU layer
 */

TEST(OUT_OF_PLACE, conv4_relu) {
	OverFeat_Fast::conv4_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * OverFeat (Fast model) fc6 ReLU layer
 */

TEST(OUT_OF_PLACE, fc6_relu) {
	OverFeat_Fast::fc6_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * OverFeat (Fast model) fc7 ReLU layer
 */

TEST(OUT_OF_PLACE, fc7_relu) {
	OverFeat_Fast::fc7_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * OverFeat (Fast model) fc8 ReLU layer
 */

TEST(OUT_OF_PLACE, fc8_relu) {
	OverFeat_Fast::fc8_relu()
		.batchSize(128)
		.testInputGradient();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
