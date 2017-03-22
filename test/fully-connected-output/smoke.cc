#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/fully-connected.h>

/*
 * Test that implementation works for a single input channel
 */

TEST(MRxNR_4x24, single_input_channel) {
	FullyConnectedTester()
		.batchSize(4)
		.outputChannels(24)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

/*
 * Test that implementation works for few input channels (1 block of L1 cache)
 */

TEST(MRxNR_4x24, few_input_channels) {
	FullyConnectedTester()
		.batchSize(4)
		.inputChannels(13)
		.outputChannels(24)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

/*
 * Test that implementation works for many input channels (multiple blocks of L1 cache)
 */

TEST(MRxNR_4x24, many_input_channels) {
	FullyConnectedTester()
		.batchSize(4)
		.inputChannels(1024)
		.outputChannels(24)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

/*
 * Test that implementation works for batch subblocks (less than a single register block of minibatch)
 */

TEST(MRxNR_4x24, batch_subblock) {
	FullyConnectedTester tester;
	tester.outputChannels(24)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t batchSize = 1; batchSize < 4; batchSize += 1) {
		tester.batchSize(batchSize)
			.testOutput();
	}
}

/*
 * Test that implementation works for few batch blocks (single cache block, but multiple register blocks of minibatch)
 */

TEST(MRxNR_4x24, small_batch_size) {
	FullyConnectedTester()
		.batchSize(12)
		.outputChannels(24)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

/*
 * Test that implementation works when batch is not divisible by subblock
 * (single cache block, multiple register blocks of minibatch, with remainder subblock)
 */

TEST(MRxNR_4x24, batch_remainder_subblock) {
	for (size_t batchSize = 12; batchSize < 16; batchSize += 1) {
		FullyConnectedTester()
			.batchSize(batchSize)
			.outputChannels(24)
			.iterations(100)
			.errorLimit(1.0e-5)
			.testOutput();
	}
}

/*
 * Test that implementation works for many batch blocks (multiple cache blocks of minibatch)
 */

TEST(MRxNR_4x24, large_batch_size) {
	FullyConnectedTester()
		.batchSize(1024)
		.outputChannels(24)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

/*
 * Test that implementation works for a subblock of output channels (less a single register block of output channels)
 */

TEST(MRxNR_4x24, output_channels_subblock) {
	FullyConnectedTester tester;
	tester.batchSize(4)
		.iterations(100)
		.errorLimit(1.0e-5);
	for (size_t outputChannels = 1; outputChannels < 24; outputChannels += 1) {
		tester.outputChannels(outputChannels)
			.testOutput();
	}
}

/*
 * Test that implementation works for few output channels (single cache block, but multiple register blocks of output channels)
 */

TEST(MRxNR_4x24, few_output_channels) {
	FullyConnectedTester()
		.batchSize(4)
		.outputChannels(72)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

/*
 * Test that implementation works when output channels count is not divisible by subblock
 * (single cache block, multiple register blocks of output channels, with remainder subblock)
 */

TEST(MRxNR_4x24, output_channels_remainder_subblock) {
	for (size_t outputChannels = 3 * 24 + 1; outputChannels < 4 * 24; outputChannels += 1) {
		FullyConnectedTester()
			.batchSize(4)
			.outputChannels(outputChannels)
			.iterations(100)
			.errorLimit(1.0e-5)
			.testOutput();
	}
}

/*
 * Test that implementation works for many output channels (multiple cache blocks of output channels)
 */

TEST(MRxNR_4x24, many_output_channels) {
	FullyConnectedTester()
		.batchSize(4)
		.outputChannels(1200)
		.iterations(100)
		.errorLimit(1.0e-5)
		.testOutput();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
