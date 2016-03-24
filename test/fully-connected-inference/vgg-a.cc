#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/fully-connected.h>
#include <models/vgg-a.h>

/*
 * VGG model A fc6 layer
 */

TEST(FC, fc6) {
	VGG_A::fc6()
		.errorLimit(2.0e-5)
		.testInference();
}

/*
 * VGG model A fc7 layer
 */

TEST(FC, fc7) {
	VGG_A::fc7()
		.errorLimit(1.0e-5)
		.testInference();
}

/*
 * VGG model A fc8 layer
 */

TEST(FC, fc8) {
	VGG_A::fc8()
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
