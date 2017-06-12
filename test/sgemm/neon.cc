#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST_4x12, kc1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(12)
		.kc(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x12__neon);
}

TEST(FAST_4x12, kc2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(12)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x12__neon);
}

TEST(FAST_4x12, kc10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(12)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x12__neon);
}

TEST(FULL_4x12, 1x1) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x2) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x3) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x4) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x5) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x6) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x7) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x8) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x9) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(9)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x10) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x11) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(11)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 1x12) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(12)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x1) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x2) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x3) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x4) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x5) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x6) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x7) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x8) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x9) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(9)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x10) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x11) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(11)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 2x12) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(12)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x1) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x2) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x3) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x4) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x5) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x6) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x7) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x8) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x9) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(9)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x10) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x11) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(11)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 3x12) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(12)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x3) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x4) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(4)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x5) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(5)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x6) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(6)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x7) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(7)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x8) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x9) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(9)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x11) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(11)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

TEST(FULL_4x12, 4x12) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(12)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x12__neon);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
