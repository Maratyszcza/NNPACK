#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/fully-connected.h>
#include <models/alexnet.h>

/*
 * AlexNet fc6 layer
 */

TEST(F32, fc6) {
	AlexNet::fc6()
		.errorLimit(2.0e-5)
		.testInferenceF32();
}

TEST(F16F32, fc6) {
	AlexNet::fc6()
		.errorLimit(2.0e-5)
		.testInferenceF16F32();
}

/*
 * AlexNet fc7 layer
 */

TEST(F32, fc7) {
	AlexNet::fc7()
		.errorLimit(1.0e-5)
		.testInferenceF32();
}

TEST(F16F32, fc7) {
	AlexNet::fc7()
		.errorLimit(1.0e-5)
		.testInferenceF16F32();
}

/*
 * AlexNet fc8 layer
 */

TEST(F32, fc8) {
	AlexNet::fc8()
		.errorLimit(1.0e-5)
		.testInferenceF32();
}

TEST(F16F32, fc8) {
	AlexNet::fc8()
		.errorLimit(1.0e-5)
		.testInferenceF16F32();
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
