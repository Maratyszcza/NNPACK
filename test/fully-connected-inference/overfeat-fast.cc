#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/fully-connected.h>
#include <models/overfeat-fast.h>

/*
 * OverFeat (Fast model) fc6 layer
 */

TEST(F32, fc6) {
	OverFeat_Fast::fc6()
		.errorLimit(2.0e-5)
		.testInferenceF32();
}

TEST(F16F32, fc6) {
	OverFeat_Fast::fc6()
		.errorLimit(2.0e-5)
		.testInferenceF16F32();
}

/*
 * OverFeat (Fast model) fc7 layer
 */

TEST(F32, fc7) {
	OverFeat_Fast::fc7()
		.errorLimit(1.0e-5)
		.testInferenceF32();
}

TEST(F16F32, fc7) {
	OverFeat_Fast::fc7()
		.errorLimit(1.0e-5)
		.testInferenceF16F32();
}

/*
 * OverFeat (Fast model) fc8 layer
 */

TEST(F32, fc8) {
	OverFeat_Fast::fc8()
		.errorLimit(1.0e-5)
		.testInferenceF32();
}

TEST(F16F32, fc8) {
	OverFeat_Fast::fc8()
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
