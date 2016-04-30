#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/softmax.h>

/*
 * Test that implementation works for a small number of channels
 */

TEST(OUT_OF_PLACE, few_channels) {
	auto tester = SoftmaxTester();
	for (size_t channels = 1; channels <= 96; channels += 1) {
		tester.channels(1000)
			.testOutput();
	}
}

TEST(IN_PLACE, few_channels) {
	auto tester = SoftmaxTester();
	for (size_t channels = 1; channels <= 96; channels += 1) {
		tester.channels(1000)
			.testOutputInplace();
	}
}

/*
 * Test that implementation works for a moderate number of channels with small batch
 */

TEST(OUT_OF_PLACE, small_batch) {
	auto tester = SoftmaxTester();
	for (size_t channels = 100; channels <= 115; channels += 1) {
		for (size_t batch = 2; batch <= 5; batch += 1) {
			tester.channels(1000)
				.batchSize(batch)
				.testOutput();
		}
	}
}

TEST(IN_PLACE, small_batch) {
	auto tester = SoftmaxTester();
	for (size_t channels = 100; channels <= 115; channels += 1) {
		for (size_t batch = 2; batch <= 5; batch += 1) {
			tester.channels(1000)
				.batchSize(batch)
				.testOutputInplace();
		}
	}
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
