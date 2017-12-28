#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST_6x8, kc1) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_6x8__neon);
}

TEST(FAST_6x8, kc2) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_6x8__neon);
}

TEST(FAST_6x8, kc10) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_6x8__neon);
}

TEST(FULL_6x8, 1x1) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x2) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x3) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x4) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x5) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x6) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x7) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 1x8) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x1) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x2) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x3) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x4) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x5) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x6) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x7) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 2x8) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x1) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x2) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x3) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x4) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x5) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x6) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x7) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 3x8) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x3) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x4) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x5) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x6) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x7) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 4x8) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x1) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x2) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x3) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x4) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x5) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x6) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x7) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 5x8) {
	GemmMicroKernelTester()
		.mr(5)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x1) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x2) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x3) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x4) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x5) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x6) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x7) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL_6x8, 6x8) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
