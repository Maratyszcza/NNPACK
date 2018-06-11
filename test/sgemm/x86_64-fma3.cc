#include <gtest/gtest.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST4x24, kc1) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
}

TEST(FAST4x24, kc2) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
}

TEST(FAST4x24, kc10) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(10)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_only_4x24__fma3);
}

TEST(FULL4x24, kc1) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(1)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL4x24, kc2) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(2)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}

TEST(FULL4x24, kc10) {
	auto tester = GemmMicroKernelTester()
		.mr(4)
		.nr(24)
		.kc(10)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester
		.accumulateC(true)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
	tester
		.accumulateC(false)
		.testSGEMM(nnp_sgemm_upto_4x24__fma3);
}
