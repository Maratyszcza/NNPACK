#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>
#include <models/overfeat-fast.h>

/*
 * OverFeat (Fast model) pool1 layer
 */

TEST(MaxPooling2x2, pool1) {
	OverFeat_Fast::pool1()
		.batchSize(128)
		.testOutput();
}

/*
 * OverFeat (Fast model) pool2 layer
 */

TEST(MaxPooling2x2, pool2) {
	OverFeat_Fast::pool2()
		.batchSize(128)
		.testOutput();
}

/*
 * OverFeat (Fast model) pool3 layer
 */

TEST(MaxPooling2x2, pool3) {
	OverFeat_Fast::pool3()
		.batchSize(128)
		.testOutput();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
