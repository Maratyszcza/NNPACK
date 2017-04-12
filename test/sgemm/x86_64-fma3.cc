#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST_4x24, kc1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
}

TEST(FAST_4x24, kc2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
}

TEST(FAST_4x24, kc10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
}

TEST(FULL_4x24, 1x1) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x2) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x3) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(3)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x4) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(4)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x5) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(5)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x6) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x7) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(7)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x8) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(8)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x9) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(9)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x10) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(10)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x11) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(11)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x12) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(12)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x13) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(13)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x14) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(14)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x15) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(15)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x16) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(16)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x17) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(17)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x18) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(18)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x19) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(19)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x20) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(20)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x21) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(21)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x22) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(22)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x23) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(23)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 1x24) {
	GemmMicroKernelTester()
		.mr(1)
		.nr(24)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x1) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x2) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x3) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(3)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x4) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(4)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x5) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(5)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x6) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x7) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(7)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x8) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(8)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x9) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(9)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x10) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(10)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x11) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(11)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x12) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(12)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x13) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(13)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x14) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(14)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x15) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(15)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x16) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(16)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x17) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(17)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x18) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(18)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x19) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(19)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x20) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(20)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x21) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(21)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x22) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(22)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x23) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(23)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 2x24) {
	GemmMicroKernelTester()
		.mr(2)
		.nr(24)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x1) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x2) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x3) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(3)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x4) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(4)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x5) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(5)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x6) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x7) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(7)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x8) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(8)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x9) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(9)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x10) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(10)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x11) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(11)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x12) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(12)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x13) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(13)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x14) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(14)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x15) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(15)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x16) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(16)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x17) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(17)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x18) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(18)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x19) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(19)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x20) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(20)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x21) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(21)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x22) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(22)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x23) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(23)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 3x24) {
	GemmMicroKernelTester()
		.mr(3)
		.nr(24)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x3) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x4) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(4)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x5) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(5)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x6) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(6)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x7) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(7)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x8) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x9) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(9)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(10)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x11) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(11)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x12) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(12)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x13) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(13)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x14) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(14)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x15) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(15)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x16) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(16)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x17) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(17)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x18) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(18)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x19) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(19)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x20) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(20)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x21) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(21)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x22) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(22)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x23) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(23)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL_4x24, 4x24) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.simdWidth(8)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
