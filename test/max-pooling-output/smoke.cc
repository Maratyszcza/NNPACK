#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/pooling.h>

/*
 * Test that implementation works for a single-channel image with a single-pool
 */

TEST(MAX_POOLING_2x2, single_pool) {
	PoolingTester()
		.inputSize(2, 2)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, single_pool) {
	PoolingTester()
		.inputSize(2, 2)
		.poolingSize(3, 3)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_1x2, single_pool) {
	PoolingTester()
		.inputSize(1, 2)
		.poolingSize(1, 2)
		.poolingStride(1, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_2x1, single_pool) {
	PoolingTester()
		.inputSize(2, 1)
		.poolingSize(2, 1)
		.poolingStride(2, 1)
		.iterations(100)
		.testOutput();
}

/*
 * Test that implementation works for a single-channel image with few horizontal pools
 */

TEST(MAX_POOLING_2x2, few_horizontal_pools) {
	for (size_t imageWidth = 4; imageWidth <= 50; imageWidth += 2) {
		PoolingTester()
			.inputSize(2, imageWidth)
			.poolingSize(2, 2)
			.poolingStride(2, 2)
			.iterations(100)
			.testOutput();
	}
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, few_horizontal_pools) {
	for (size_t imageWidth = 4; imageWidth <= 50; imageWidth += 2) {
		PoolingTester()
			.inputSize(2, imageWidth)
			.poolingSize(3, 3)
			.poolingStride(2, 2)
			.iterations(100)
			.testOutput();
	}
}

TEST(MAX_POOLING_1x2, few_horizontal_pools) {
	for (size_t imageWidth = 4; imageWidth <= 50; imageWidth += 2) {
		PoolingTester()
			.inputSize(1, imageWidth)
			.poolingSize(1, 2)
			.poolingStride(1, 2)
			.iterations(100)
			.testOutput();
	}
}

TEST(MAX_POOLING_2x1, few_horizontal_pools) {
	for (size_t imageWidth = 2; imageWidth <= 50; imageWidth += 1) {
		PoolingTester()
			.inputSize(2, imageWidth)
			.poolingSize(2, 2)
			.poolingStride(2, 1)
			.iterations(100)
			.testOutput();
	}
}

/*
 * Test that implementation works for a single-channel image with few vertical pools
 */

TEST(MAX_POOLING_2x2, few_vertical_pools) {
	for (size_t imageHeight = 4; imageHeight <= 50; imageHeight += 2) {
		PoolingTester()
			.inputSize(imageHeight, 2)
			.poolingSize(2, 2)
			.poolingStride(2, 2)
			.iterations(100)
			.testOutput();
	}
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, few_vertical_pools) {
	for (size_t imageHeight = 4; imageHeight <= 50; imageHeight += 2) {
		PoolingTester()
			.inputSize(imageHeight, 2)
			.poolingSize(3, 3)
			.poolingStride(2, 2)
			.iterations(100)
			.testOutput();
	}
}

TEST(MAX_POOLING_1x2, few_vertical_pools) {
	for (size_t imageHeight = 2; imageHeight <= 50; imageHeight += 1) {
		PoolingTester()
			.inputSize(imageHeight, 2)
			.poolingSize(1, 2)
			.poolingStride(1, 2)
			.iterations(100)
			.testOutput();
	}
}

TEST(MAX_POOLING_2x1, few_vertical_pools) {
	for (size_t imageHeight = 4; imageHeight <= 50; imageHeight += 2) {
		PoolingTester()
			.inputSize(imageHeight, 1)
			.poolingSize(2, 1)
			.poolingStride(2, 1)
			.iterations(100)
			.testOutput();
	}
}

/*
 * Test that implementation works for a single-channel image with multiple horizontal and vertical pools
 */

TEST(MAX_POOLING_2x2, large_image) {
	PoolingTester()
		.inputSize(128, 128)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, large_image) {
	PoolingTester()
		.inputSize(129, 129)
		.poolingSize(3, 3)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_1x2, large_image) {
	PoolingTester()
		.inputSize(128, 128)
		.poolingSize(1, 2)
		.poolingStride(1, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_2x1, large_image) {
	PoolingTester()
		.inputSize(128, 128)
		.poolingSize(2, 1)
		.poolingStride(2, 1)
		.iterations(100)
		.testOutput();
}

/*
 * Test that implementation works for a single-channel image with size which is not perfectly divisible by the pool size
 */

TEST(MAX_POOLING_2x2, indivisible_size) {
	PoolingTester()
		.inputSize(5, 5)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, indivisible_size) {
	PoolingTester()
		.inputSize(6, 6)
		.poolingSize(3, 3)
		.poolingStride(2, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_1x2, indivisible_size) {
	PoolingTester()
		.inputSize(1, 5)
		.poolingSize(1, 2)
		.poolingStride(1, 2)
		.iterations(100)
		.testOutput();
}

TEST(MAX_POOLING_2x1, indivisible_size) {
	PoolingTester()
		.inputSize(5, 1)
		.poolingSize(2, 1)
		.poolingStride(2, 1)
		.iterations(100)
		.testOutput();
}

/*
 * Test that implementation works for a single-channel image with implicit padding
 */

TEST(MAX_POOLING_2x2, implicit_padding) {
	PoolingTester tester;
	tester.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100);
	const size_t inputHeight = 24;
	const size_t inputWidth = 24;
	for (size_t paddingTop = 0; paddingTop < tester.poolingHeight(); paddingTop++) {
		for (size_t paddingLeft = 0; paddingLeft < tester.poolingWidth(); paddingLeft++) {
			for (size_t paddingBottom = 0; paddingBottom < tester.poolingHeight(); paddingBottom++) {
				for (size_t paddingRight = 0; paddingRight < tester.poolingWidth(); paddingRight++) {
					tester.inputSize(
							inputHeight - paddingTop - paddingBottom,
							inputWidth - paddingLeft - paddingRight)
						.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput();
				}
			}
		}
	}
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, implicit_padding) {
	PoolingTester tester;
	tester.poolingSize(3, 3)
		.poolingStride(2, 2)
		.iterations(100);
	const size_t inputHeight = 23;
	const size_t inputWidth = 23;
	for (size_t paddingTop = 0; paddingTop < tester.poolingHeight(); paddingTop++) {
		for (size_t paddingLeft = 0; paddingLeft < tester.poolingWidth(); paddingLeft++) {
			for (size_t paddingBottom = 0; paddingBottom < tester.poolingHeight(); paddingBottom++) {
				for (size_t paddingRight = 0; paddingRight < tester.poolingWidth(); paddingRight++) {
					tester.inputSize(
							inputHeight - paddingTop - paddingBottom,
							inputWidth - paddingLeft - paddingRight)
						.inputPadding(paddingTop, paddingRight, paddingBottom, paddingLeft)
						.testOutput();
				}
			}
		}
	}
}

TEST(MAX_POOLING_1x2, implicit_padding) {
	PoolingTester tester;
	tester.poolingSize(1, 2)
		.poolingStride(1, 2)
		.iterations(100);
	const size_t inputHeight = 24;
	const size_t inputWidth = 24;
	for (size_t paddingLeft = 0; paddingLeft < tester.poolingWidth(); paddingLeft++) {
		for (size_t paddingRight = 0; paddingRight < tester.poolingWidth(); paddingRight++) {
			tester.inputSize(inputHeight, inputWidth - paddingLeft - paddingRight)
				.inputPadding(0, paddingRight, 0, paddingLeft)
				.testOutput();
		}
	}
}

TEST(MAX_POOLING_2x1, implicit_padding) {
	PoolingTester tester;
	tester.poolingSize(2, 1)
		.poolingStride(2, 1)
		.iterations(100);
	const size_t inputHeight = 24;
	const size_t inputWidth = 24;
	for (size_t paddingTop = 0; paddingTop < tester.poolingHeight(); paddingTop++) {
		for (size_t paddingBottom = 0; paddingBottom < tester.poolingHeight(); paddingBottom++) {
			tester.inputSize(inputHeight - paddingTop - paddingBottom, inputWidth)
				.inputPadding(paddingTop, 0, paddingBottom, 0)
				.testOutput();
		}
	}
}

/*
 * Test that implementation can handle small non-unit batch_size
 */

TEST(MAX_POOLING_2x2, small_batch) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput();
	}
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, small_batch) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(3, 3)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput();
	}
}

TEST(MAX_POOLING_1x2, small_batch) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(1, 2)
		.poolingStride(1, 2)
		.iterations(100);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput();
	}
}

TEST(MAX_POOLING_2x1, small_batch) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(2, 1)
		.poolingStride(2, 1)
		.iterations(100);
	for (size_t batchSize = 2; batchSize <= 5; batchSize++) {
		tester.batchSize(batchSize)
			.testOutput();
	}
}

/*
 * Test that implementation can handle small non-unit number of channels
 */

TEST(MAX_POOLING_2x2, few_channels) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(2, 2)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t channels = 2; channels <= 5; channels++) {
		tester.channels(channels)
			.testOutput();
	}
}

TEST(MAX_POOLING_3x3_STRIDE_2x2, few_channels) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(3, 3)
		.poolingStride(2, 2)
		.iterations(100);
	for (size_t channels = 2; channels <= 5; channels++) {
		tester.channels(channels)
			.testOutput();
	}
}

TEST(MAX_POOLING_1x2, few_channels) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(1, 2)
		.poolingStride(1, 2)
		.iterations(100);
	for (size_t channels = 2; channels <= 5; channels++) {
		tester.channels(channels)
			.testOutput();
	}
}

TEST(MAX_POOLING_2x1, few_channels) {
	PoolingTester tester;
	tester.inputSize(12, 12)
		.poolingSize(2, 1)
		.poolingStride(2, 1)
		.iterations(100);
	for (size_t channels = 2; channels <= 5; channels++) {
		tester.channels(channels)
			.testOutput();
	}
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	std::cout << init_status << std::endl;
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
