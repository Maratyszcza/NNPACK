#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>
#include <models/vgg-a.h>

/*
 * VGG model A pool1 layer
 */

TEST(MaxPooling2x2, pool1) {
	VGG_A::pool1()
		.batchSize(64)
		.testOutput();
}

/*
 * VGG model A pool2 layer
 */

TEST(MaxPooling2x2, pool2) {
	VGG_A::pool2()
		.batchSize(64)
		.testOutput();
}

/*
 * VGG model A pool3 layer
 */

TEST(MaxPooling2x2, pool3) {
	VGG_A::pool3()
		.batchSize(64)
		.testOutput();
}

/*
 * VGG model A pool4 layer
 */

TEST(MaxPooling2x2, pool4) {
	VGG_A::pool4()
		.batchSize(64)
		.testOutput();
}

/*
 * VGG model A pool5 layer
 */

TEST(MaxPooling2x2, pool5) {
	VGG_A::pool5()
		.batchSize(64)
		.testOutput();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
