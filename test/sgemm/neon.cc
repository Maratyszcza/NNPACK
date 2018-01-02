#include <gtest/gtest.h>

#include <cpuinfo.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

TEST(FAST6x8_NEON, kc1) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_6x8__neon);
}

TEST(FAST6x8_NEON, kc2) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_6x8__neon);
}

TEST(FAST6x8_NEON, kc10) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_only_6x8__neon);
}

#if CPUINFO_ARCH_ARM
	TEST(FAST6x8_AARCH32_NEON, kc1) {
		GemmMicroKernelTester()
			.mr(6)
			.nr(8)
			.kc(1)
			.simdWidth(4)
			.errorLimit(1.0e-6f)
			.testSGEMM(nnp_sgemm_only_6x8__aarch32_neon);
	}

	TEST(FAST6x8_AARCH32_NEON, kc2) {
		GemmMicroKernelTester()
			.mr(6)
			.nr(8)
			.kc(2)
			.simdWidth(4)
			.errorLimit(1.0e-6f)
			.testSGEMM(nnp_sgemm_only_6x8__aarch32_neon);
	}

	TEST(FAST6x8_AARCH32_NEON, kc10) {
		GemmMicroKernelTester()
			.mr(6)
			.nr(8)
			.kc(10)
			.simdWidth(4)
			.errorLimit(1.0e-6f)
			.testSGEMM(nnp_sgemm_only_6x8__aarch32_neon);
	}
#endif

TEST(FULL6x8_NEON, kc1) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(1)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL6x8_NEON, kc2) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(2)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}

TEST(FULL6x8_NEON, kc10) {
	GemmMicroKernelTester()
		.mr(6)
		.nr(8)
		.kc(10)
		.simdWidth(4)
		.errorLimit(1.0e-6f)
		.testSGEMM(nnp_sgemm_upto_6x8__neon);
}
