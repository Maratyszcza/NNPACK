#include <gtest/gtest.h>

#include <nnpack.h>

#include <testers/convolution.h>
#include <models/overfeat-fast.h>

/*
 * OverFeat (Fast model) conv1 layer
 */

TEST(IMPLICIT_GEMM, conv1) {
	OverFeat_Fast::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(IMPLICIT_GEMM, conv1_with_relu) {
	OverFeat_Fast::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM_PREPACK, conv1) {
	OverFeat_Fast::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv1_with_relu) {
	OverFeat_Fast::conv1()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu, true);
}

/*
 * OverFeat (Fast model) conv2 layer
 */

TEST(FT8x8, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity);
}

TEST(FT8x8, conv2_with_relu) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity);
}

TEST(FT16x16, conv2_with_relu) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(IMPLICIT_GEMM, conv2_with_relu) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

TEST(FT8x8_PRECOMPUTE, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity, true);
}

TEST(FT8x8_PRECOMPUTE, conv2_with_relu) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu, true);
}

TEST(FT16x16_PRECOMPUTE, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity, true);
}

TEST(FT16x16_PRECOMPUTE, conv2_with_relu) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv2) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv2_with_relu) {
	OverFeat_Fast::conv2()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu, true);
}

/*
 * OverFeat (Fast model) conv3 layer
 */

TEST(FT8x8, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity);
}

TEST(FT8x8, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity);
}

TEST(FT16x16, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_identity);
}

TEST(WT8x8, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(WT8x8_FP16, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(5.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_identity);
}

TEST(WT8x8_FP16, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(5.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(IMPLICIT_GEMM, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

TEST(FT8x8_PRECOMPUTE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity, true);
}

TEST(FT8x8_PRECOMPUTE, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu, true);
}

TEST(FT16x16_PRECOMPUTE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity, true);
}

TEST(FT16x16_PRECOMPUTE, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu, true);
}

TEST(WT8x8_PRECOMPUTE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_identity, true);
}

TEST(WT8x8_PRECOMPUTE, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu, true);
}

TEST(WT8x8_FP16_PRECOMPUTE, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(5.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_identity, true);
}

TEST(WT8x8_FP16_PRECOMPUTE, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(5.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_relu, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv3) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv3_with_relu) {
	OverFeat_Fast::conv3()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu, true);
}

/*
 * OverFeat (Fast model) conv4 layer
 */

TEST(FT8x8, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity);
}

TEST(FT8x8, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity);
}

TEST(FT16x16, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_identity);
}

TEST(WT8x8, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(WT8x8_FP16, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_identity);
}

TEST(WT8x8_FP16, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(IMPLICIT_GEMM, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

TEST(FT8x8_PRECOMPUTE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity, true);
}

TEST(FT8x8_PRECOMPUTE, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu, true);
}

TEST(FT16x16_PRECOMPUTE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity, true);
}

TEST(FT16x16_PRECOMPUTE, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu, true);
}

TEST(WT8x8_PRECOMPUTE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_identity, true);
}

TEST(WT8x8_PRECOMPUTE, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu, true);
}

TEST(WT8x8_FP16_PRECOMPUTE, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_identity, true);
}

TEST(WT8x8_FP16_PRECOMPUTE, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_relu, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv4) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv4_with_relu) {
	OverFeat_Fast::conv4()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu, true);
}

/*
 * OverFeat (Fast model) conv5 layer
 */

TEST(FT8x8, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity);
}

TEST(FT8x8, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu);
}

TEST(FT16x16, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity);
}

TEST(FT16x16, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu);
}

TEST(WT8x8, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_identity);
}

TEST(WT8x8, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu);
}

TEST(WT8x8_FP16, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_identity);
}

TEST(WT8x8_FP16, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_relu);
}

TEST(IMPLICIT_GEMM, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(IMPLICIT_GEMM, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu);
}

TEST(FT8x8_PRECOMPUTE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_identity, true);
}

TEST(FT8x8_PRECOMPUTE, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft8x8, nnp_activation_relu, true);
}

TEST(FT16x16_PRECOMPUTE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity, true);
}

TEST(FT16x16_PRECOMPUTE, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_relu, true);
}

TEST(WT8x8_PRECOMPUTE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_identity, true);
}

TEST(WT8x8_PRECOMPUTE, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_wt8x8, nnp_activation_relu, true);
}

TEST(WT8x8_FP16_PRECOMPUTE, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_identity, true);
}

TEST(WT8x8_FP16_PRECOMPUTE, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(3.0e-2)
		.testInference(nnp_convolution_algorithm_wt8x8_fp16, nnp_activation_relu, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv5) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity, true);
}

TEST(IMPLICIT_GEMM_PREPACK, conv5_with_relu) {
	OverFeat_Fast::conv5()
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_implicit_gemm, nnp_activation_relu, true);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
