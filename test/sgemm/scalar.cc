#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST4x3, kc1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x3__scalar);
}

TEST(FAST4x3, kc2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x3__scalar);
}

TEST(FAST4x3, kc10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(10)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_4x3__scalar);
}

TEST(FULL4x3, kc1) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(1)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL4x3, kc2) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(2)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}

TEST(FULL4x3, kc10) {
	GemmMicroKernelTester()
		.mr(4)
		.nr(3)
		.kc(10)
		.simdWidth(1)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_4x3__scalar);
}
