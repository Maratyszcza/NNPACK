#include <gtest/gtest.h>

#include <testers/fourier.h>

/**
 * Test output of complex 1D forward FFT within rows against reference implementation with SOA layout.
 */

TEST(FFT8_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-8f);
	tester.testOptimizedComplex(nnp_fft8_soa__avx2, nnp_fft8_soa__ref);
}

TEST(FFT16_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-8f);
	tester.testOptimizedComplex(nnp_fft16_soa__avx2, nnp_fft16_soa__ref);
}

/**
 * Test output of complex 1D inverse FFT within rows against reference implementation with SOA layout.
 */

TEST(IFFT8_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-8f);
	tester.testOptimizedComplex(nnp_ifft8_soa__avx2, nnp_ifft8_soa__ref);
}

TEST(IFFT16_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-8f);
	tester.testOptimizedComplex(nnp_ifft16_soa__avx2, nnp_ifft16_soa__ref);
}

/**
 * Test output of dual-sequence real 1D forward FFT within rows against reference implementation.
 */

TEST(FFT8_DUAL_REAL_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-7f);
	tester.testOptimizedDualReal(nnp_fft8_dualreal__avx2, nnp_fft8_dualreal__ref);
}

TEST(FFT16_DUAL_REAL_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-7f);
	tester.testOptimizedDualReal(nnp_fft16_dualreal__avx2, nnp_fft16_dualreal__ref);
}

/**
 * Test output of dual-sequence real 1D inverse FFT within rows against reference implementation.
 */

TEST(IFFT8_DUAL_REAL_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-7f);
	tester.testOptimizedDualReal(nnp_ifft8_dualreal__avx2, nnp_ifft8_dualreal__ref);
}

TEST(IFFT16_DUAL_REAL_WITHIN_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-7f);
	tester.testOptimizedDualReal(nnp_ifft16_dualreal__avx2, nnp_ifft16_dualreal__ref);
}

/**
 * Test output of complex 1D forward FFT across rows against reference implementation with AOS layout.
 */

TEST(FFT4_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(4)
		.simdWidth(8)
		.errorLimit(1.0e-8f);
	tester.testOptimizedComplex(nnp_fft4_8aos__fma3, nnp_fft4_aos__ref);
}

TEST(FFT8_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.simdWidth(8)
		.errorLimit(1.0e-7f);
	tester.testOptimizedComplex(nnp_fft8_8aos__fma3, nnp_fft8_aos__ref);
}

/**
 * Test output of complex 1D forward FFT across rows against reference implementation with AOS layout.
 */

TEST(IFFT8_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.simdWidth(8)
		.errorLimit(1.0e-7f);
	tester.testOptimizedComplex(nnp_ifft8_8aos__fma3, nnp_ifft8_aos__ref);
}

/**
 * Test output of real 1D forward FFT across rows against reference implementation.
 */

TEST(FFT8_REAL_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.simdWidth(8)
		.errorLimit(1.0e-7f);
	tester.testOptimizedReal(nnp_fft8_8real__fma3, nnp_fft8_real__ref);
}

TEST(FFT16_REAL_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(16)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester.testOptimizedReal(nnp_fft16_8real__fma3, nnp_fft16_real__ref);
}

/**
 * Test output of real 1D inverse FFT across rows against reference implementation.
 */

TEST(IFFT8_REAL_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(8)
		.simdWidth(8)
		.errorLimit(1.0e-7f);
	tester.testOptimizedReal(nnp_ifft8_8real__fma3, nnp_ifft8_real__ref);
}

TEST(IFFT16_REAL_ACROSS_ROWS, match_reference) {
	auto tester = FFTTester()
		.fftSize(16)
		.simdWidth(8)
		.errorLimit(1.0e-6f);
	tester.testOptimizedReal(nnp_ifft16_8real__fma3, nnp_ifft16_real__ref);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
