#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST4x8, kc1) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.kc(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_only_4x8__psimd);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_only_4x8__psimd);
}

TEST(FAST4x8, kc2) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_only_4x8__psimd);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_only_4x8__psimd);
}

TEST(FAST4x8, kc10) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.kc(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_only_4x8__psimd);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_only_4x8__psimd);
}

TEST(FULL4x8, kc1) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.kc(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_upto_4x8__psimd);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_upto_4x8__psimd);
}

TEST(FULL4x8, kc2) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_upto_4x8__psimd);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_upto_4x8__psimd);
}

TEST(FULL4x8, kc10) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(8)
		.kc(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_upto_4x8__psimd);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_upto_4x8__psimd);
}
