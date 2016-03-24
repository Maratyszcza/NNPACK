#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/fully-connected.h>
#include <models/alexnet.h>

/*
 * AlexNet fc6 layer
 */

TEST(FC, fc6) {
	AlexNet::fc6()
		.errorLimit(2.0e-5)
		.testInference();
}

/*
 * AlexNet fc7 layer
 */

TEST(FC, fc7) {
	AlexNet::fc7()
		.errorLimit(1.0e-5)
		.testInference();
}

/*
 * AlexNet fc8 layer
 */

TEST(FC, fc8) {
	AlexNet::fc8()
		.errorLimit(1.0e-5)
		.testInference();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
