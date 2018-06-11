#include <gtest/gtest.h>

#include <cpuinfo.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(FAST_S4GEMM_3x3, neon) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-6f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testSXGEMM(nnp_fast_tuple_gemm_function(nnp_s4gemm_only_3x3__neon));
			tester
				.accumulateC(false)
				.testSXGEMM(nnp_fast_tuple_gemm_function(nnp_s4gemm_only_3x3__neon));
		}
	}
}

TEST(FULL_S4GEMM_3x3, neon) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-6f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testSXGEMM(nnp_full_tuple_gemm_function(nnp_s4gemm_upto_3x3__neon));
			tester
				.accumulateC(false)
				.testSXGEMM(nnp_full_tuple_gemm_function(nnp_s4gemm_upto_3x3__neon));
		}
	}
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_ARM
TEST(FAST_S4GEMM_3x3, aarch32_neon) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-6f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testSXGEMM(nnp_fast_tuple_gemm_function(nnp_s4gemm_only_3x3__aarch32_neon));
			tester
				.accumulateC(false)
				.testSXGEMM(nnp_fast_tuple_gemm_function(nnp_s4gemm_only_3x3__aarch32_neon));
		}
	}
}

TEST(FAST_S4GEMM_3x3, aarch32_neon2) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-6f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testSXGEMM(nnp_fast_tuple_gemm_function(nnp_s4gemm_only_3x3__aarch32_neon2));
			tester
				.accumulateC(false)
				.testSXGEMM(nnp_fast_tuple_gemm_function(nnp_s4gemm_only_3x3__aarch32_neon2));
		}
	}
}
#endif /* CPUINFO_ARCH_ARM */
