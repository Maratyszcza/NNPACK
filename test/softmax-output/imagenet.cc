#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/softmax.h>

/*
 * ImageNet (1000 categories) with batch size = 1
 */

TEST(OUT_OF_PLACE, batch1) {
	SoftmaxTester()
		.channels(1000)
		.testOutput();
}

TEST(IN_PLACE, batch1) {
	SoftmaxTester()
		.channels(1000)
		.testOutputInplace();
}

/*
 * ImageNet (1000 categories) with batch size = 2
 */

TEST(OUT_OF_PLACE, batch2) {
	SoftmaxTester()
		.batchSize(2)
		.channels(1000)
		.testOutput();
}

TEST(IN_PLACE, batch2) {
	SoftmaxTester()
		.batchSize(2)
		.channels(1000)
		.testOutputInplace();
}

/*
 * ImageNet (1000 categories) with batch size = 16
 */

TEST(OUT_OF_PLACE, batch16) {
	SoftmaxTester()
		.batchSize(16)
		.channels(1000)
		.testOutput();
}

TEST(IN_PLACE, batch16) {
	SoftmaxTester()
		.batchSize(16)
		.channels(1000)
		.testOutputInplace();
}

/*
 * ImageNet (1000 categories) with batch size = 64
 */

TEST(OUT_OF_PLACE, batch64) {
	SoftmaxTester()
		.batchSize(64)
		.channels(1000)
		.testOutput();
}

TEST(IN_PLACE, batch64) {
	SoftmaxTester()
		.batchSize(64)
		.channels(1000)
		.testOutputInplace();
}

/*
 * ImageNet (1000 categories) with batch size = 128
 */

TEST(OUT_OF_PLACE, batch128) {
	SoftmaxTester()
		.multithreading(true)
		.batchSize(128)
		.channels(1000)
		.testOutput();
}

TEST(IN_PLACE, batch128) {
	SoftmaxTester()
		.multithreading(true)
		.batchSize(128)
		.channels(1000)
		.testOutputInplace();
}

/*
 * ImageNet (1000 categories) with batch size = 256
 */

TEST(OUT_OF_PLACE, batch256) {
	SoftmaxTester()
		.multithreading(true)
		.batchSize(256)
		.channels(1000)
		.testOutput();
}

TEST(IN_PLACE, batch256) {
	SoftmaxTester()
		.multithreading(true)
		.batchSize(256)
		.channels(1000)
		.testOutputInplace();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
