#include <gtest/gtest.h>

#include <cpuinfo.h>

#include <testers/gemm-ukernel.h>
#include <nnpack/blas.h>

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(FAST_H4GEMM_3x3, neonhp) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-3f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__neonhp));
			tester
				.accumulateC(false)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__neonhp));
		}
	}
}

TEST(FULL_H4GEMM_3x3, neon) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-3f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testHXGEMM(nnp_full_tuple_gemm_function(nnp_h4gemm_upto_3x3__neonhp));
			tester
				.accumulateC(false)
				.testHXGEMM(nnp_full_tuple_gemm_function(nnp_h4gemm_upto_3x3__neonhp));
		}
	}
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_ARM
TEST(FAST_H4GEMM_3x3, aarch32_neonhp) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-3f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__aarch32_neonhp));
			tester
				.accumulateC(false)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__aarch32_neonhp));
		}
	}
}

TEST(FAST_H4GEMM_3x3, aarch32_neon2) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-3f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__aarch32_neon2));
			tester
				.accumulateC(false)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__aarch32_neon2));
		}
	}
}

TEST(FULL_H4GEMM_3x3, aarch32_neon2) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fma()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-3f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testHXGEMM(nnp_full_tuple_gemm_function(nnp_h4gemm_upto_3x3__aarch32_neon2));
			tester
				.accumulateC(false)
				.testHXGEMM(nnp_full_tuple_gemm_function(nnp_h4gemm_upto_3x3__aarch32_neon2));
		}
	}
}

TEST(FAST_H4GEMM_3x3, aarch32_neonhparith) {
	ASSERT_TRUE(cpuinfo_initialize());
	if (cpuinfo_has_arm_neon_fp16_arith()) {
		GemmMicroKernelTester tester = GemmMicroKernelTester()
			.simdWidth(4)
			.mr(3)
			.nr(3)
			.errorLimit(1.0e-3f);

		for (uint32_t kc = 1; kc < 10; kc++) {
			tester
				.kc(kc)
				.accumulateC(true)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__aarch32_neonhparith));
			tester
				.accumulateC(false)
				.testHXGEMM(nnp_fast_tuple_gemm_function(nnp_h4gemm_only_3x3__aarch32_neonhparith));
		}
	}
}
#endif /* CPUINFO_ARCH_ARM */
