#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>

TEST(FAST_4x3, kc1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x3__scalar);
}

TEST(FAST_4x3, kc2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x3__scalar);
}

TEST(FAST_4x3, kc10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x3__scalar);
}

TEST(FULL_4x3, 1x1) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 1x2) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 1x3) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(3)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 2x1) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 2x2) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 2x3) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(3)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 3x1) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 3x2) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 3x3) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(3)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 4x1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 4x2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL_4x3, 4x3) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
