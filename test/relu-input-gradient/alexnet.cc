#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>
#include <models/alexnet.h>

/*
 * AlexNet conv1 ReLU layer
 */

TEST(OUT_OF_PLACE, conv1_relu) {
	AlexNet::conv1_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * AlexNet conv1 ReLU layer
 */

TEST(OUT_OF_PLACE, conv2_relu) {
	AlexNet::conv2_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * AlexNet conv3 ReLU layer
 */

TEST(OUT_OF_PLACE, conv3_relu) {
	AlexNet::conv3_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * AlexNet conv4 ReLU layer
 */

TEST(OUT_OF_PLACE, conv4_relu) {
	AlexNet::conv4_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * AlexNet fc6 ReLU layer
 */

TEST(OUT_OF_PLACE, fc6_relu) {
	AlexNet::fc6_relu()
		.batchSize(128)
		.testInputGradient();
}

/*
 * AlexNet fc8 ReLU layer
 */

TEST(OUT_OF_PLACE, fc8_relu) {
	AlexNet::fc8_relu()
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
