#include <gtest/gtest.h>

#include <testers/fourier.h>
#include <fft-samples.h>

/*
 * The tests in this validate the reference Fourier transform implementations.
 * Test plan:
 *   1. Check that output of complex 1D forward FFT with AOS layout matches reference samples.
 *   2. Check that complex 1D forward FFT + inverse FFT with AOS layout is an identity transformation.
 *   3. Check that complex 1D forward FFT with SOA layout matches output of complex 1D forward FFT with AOS layout.
 *   4. Check that complex 1D inverse FFT with SOA layout matches output of complex 1D inverse FFT with AOS layout.
 *   5. Check that real-to-complex 1D forward FFT matches output of complex 1D forward FFT with SOA layout.
 *   6. Check that complex-to-real 1D inverse FFT matches output of complex 1D inverse FFT with SOA layout.
 *   7. Check that dual real-to-complex 1D forward FFT matches output of real-to-complex 1D forward FFT.
 *   8. Check that dual complex-to-real 1D inverse FFT matches output of complex-to-real 1D inverse FFT.
 */


/**
 * Test output of complex 1D forward FFT with AOS layout against reference samples
 */

TEST(FFT2_AOS, reference_sample) {
	auto tester = FFTTester()
		.fftSize(2)
		.errorLimit(1.0e-6f);
	tester.testForwardAosSamples(nnp_fft2_aos__ref,
		samples::fft2::input, samples::fft2::output);
}

TEST(FFT4_AOS, reference_sample) {
	auto tester = FFTTester()
		.fftSize(4)
		.errorLimit(1.0e-6f);
	tester.testForwardAosSamples(nnp_fft4_aos__ref,
		samples::fft4::input, samples::fft4::output);
}

TEST(FFT8_AOS, reference_sample) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-6f);
	tester.testForwardAosSamples(nnp_fft8_aos__ref,
		samples::fft8::input, samples::fft8::output);
}

TEST(FFT16_AOS, reference_sample) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-5f);
	tester.testForwardAosSamples(nnp_fft16_aos__ref,
		samples::fft16::input, samples::fft16::output);
}

TEST(FFT32_AOS, reference_sample) {
	auto tester = FFTTester()
		.fftSize(32)
		.errorLimit(1.0e-5f);
	tester.testForwardAosSamples(nnp_fft32_aos__ref,
		samples::fft32::input, samples::fft32::output);
}

/**
 * Test that complex 1D forward FFT + inverse FFT with AOS layout is an identity transformation.
 */

TEST(FFT2_AOS, forward_and_inverse) {
	auto tester = FFTTester()
		.fftSize(2)
		.iterations(10000)
		.errorLimit(1.0e-4f);
	tester.testForwardAndInverseAos(nnp_fft2_aos__ref, nnp_ifft2_aos__ref);
}

TEST(FFT4_AOS, forward_and_inverse) {
	auto tester = FFTTester()
		.fftSize(4)
		.iterations(10000)
		.errorLimit(1.0e-3f);
	tester.testForwardAndInverseAos(nnp_fft4_aos__ref, nnp_ifft4_aos__ref);
}

TEST(FFT8_AOS, forward_and_inverse) {
	auto tester = FFTTester()
		.fftSize(8)
		.iterations(10000)
		.errorLimit(1.0e-2f);
	tester.testForwardAndInverseAos(nnp_fft8_aos__ref, nnp_ifft8_aos__ref);
}

TEST(FFT16_AOS, forward_and_inverse) {
	auto tester = FFTTester()
		.fftSize(16)
		.iterations(10000)
		.errorLimit(1.0e-2f);
	tester.testForwardAndInverseAos(nnp_fft16_aos__ref, nnp_ifft16_aos__ref);
}

TEST(FFT32_AOS, forward_and_inverse) {
	auto tester = FFTTester()
		.fftSize(32)
		.iterations(10000)
		.errorLimit(1.0e-2f);
	tester.testForwardAndInverseAos(nnp_fft32_aos__ref, nnp_ifft32_aos__ref);
}

/**
 * Test that complex 1D forward FFT with SOA layout matches output of complex 1D forward FFT with AOS layout.
 */

TEST(FFT2_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(2)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_fft2_soa__ref, nnp_fft2_aos__ref);
}

TEST(FFT4_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(4)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_fft4_soa__ref, nnp_fft4_aos__ref);
}

TEST(FFT8_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_fft8_soa__ref, nnp_fft8_aos__ref);
}

TEST(FFT16_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_fft16_soa__ref, nnp_fft16_aos__ref);
}

TEST(FFT32_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(32)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_fft32_soa__ref, nnp_fft32_aos__ref);
}

/**
 * Test that complex 1D inverse FFT with SOA layout matches output of complex 1D inverse FFT with AOS layout.
 */

TEST(IFFT2_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(2)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_ifft2_soa__ref, nnp_ifft2_aos__ref);
}

TEST(IFFT4_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(4)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_ifft4_soa__ref, nnp_ifft4_aos__ref);
}

TEST(IFFT8_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_ifft8_soa__ref, nnp_ifft8_aos__ref);
}

TEST(IFFT16_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_ifft16_soa__ref, nnp_ifft16_aos__ref);
}

TEST(IFFT32_SOA, match_aos) {
	auto tester = FFTTester()
		.fftSize(32)
		.errorLimit(1.0e-7f);
	tester.testSoa(nnp_ifft32_soa__ref, nnp_ifft32_aos__ref);
}

/**
 * Test that complex 1D inverse FFT with SOA layout matches output of complex 1D inverse FFT with AOS layout.
 */

TEST(FFT8_REAL, match_complex) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-3f);
	tester.testRealToComplex(nnp_fft8_real__ref, nnp_fft8_aos__ref);
}

TEST(FFT16_REAL, match_complex) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-2f);
	tester.testRealToComplex(nnp_fft16_real__ref, nnp_fft16_aos__ref);
}

TEST(FFT32_REAL, match_complex) {
	auto tester = FFTTester()
		.fftSize(32)
		.errorLimit(1.0e-2f);
	tester.testRealToComplex(nnp_fft32_real__ref, nnp_fft32_aos__ref);
}

/**
 * Test that complex 1D inverse FFT with SOA layout matches output of complex 1D inverse FFT with AOS layout.
 */

TEST(IFFT8_REAL, match_complex) {
	auto tester = FFTTester()
		.fftSize(8)
		.errorLimit(1.0e-3f);
	tester.testComplexToReal(nnp_ifft8_real__ref, nnp_ifft8_aos__ref);
}

TEST(IFFT16_REAL, match_complex) {
	auto tester = FFTTester()
		.fftSize(16)
		.errorLimit(1.0e-2f);
	tester.testComplexToReal(nnp_ifft16_real__ref, nnp_ifft16_aos__ref);
}

TEST(IFFT32_REAL, match_complex) {
	auto tester = FFTTester()
		.fftSize(32)
		.errorLimit(1.0e-2f);
	tester.testComplexToReal(nnp_ifft32_real__ref, nnp_ifft32_aos__ref);
}

/**
 * Test that dual real-to-complex 1D forward FFT matches output of real-to-complex 1D forward FFT.
 */

TEST(FFT8_DUALREAL, match_real) {
	auto tester = FFTTester()
		.fftSize(8);
	tester.testDualRealToComplex(nnp_fft8_dualreal__ref, nnp_fft8_real__ref);
}

TEST(FFT16_DUALREAL, match_real) {
	auto tester = FFTTester()
		.fftSize(16);
	tester.testDualRealToComplex(nnp_fft16_dualreal__ref, nnp_fft16_real__ref);
}

TEST(FFT32_DUALREAL, match_real) {
	auto tester = FFTTester()
		.fftSize(32);
	tester.testDualRealToComplex(nnp_fft32_dualreal__ref, nnp_fft32_real__ref);
}

/**
 * Test that dual real-to-complex 1D forward FFT matches output of real-to-complex 1D forward FFT.
 */

TEST(IFFT8_DUALREAL, match_real) {
	auto tester = FFTTester()
		.fftSize(8);
	tester.testDualComplexToReal(nnp_ifft8_dualreal__ref, nnp_ifft8_real__ref);
}

TEST(IFFT16_DUALREAL, match_real) {
	auto tester = FFTTester()
		.fftSize(16);
	tester.testDualComplexToReal(nnp_ifft16_dualreal__ref, nnp_ifft16_real__ref);
}

TEST(IFFT32_DUALREAL, match_real) {
	auto tester = FFTTester()
		.fftSize(32);
	tester.testDualComplexToReal(nnp_ifft32_dualreal__ref, nnp_ifft32_real__ref);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
