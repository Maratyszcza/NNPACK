#pragma once

#include <complex.h>

#ifndef CMPLXF
	#define CMPLXF(real, imag) ((real) + _Complex_I * (imag))
#endif

#ifdef __ANDROID__
	/* Work-around for pre-API 23 Android, where libc does not provide crealf */
	#if __ANDROID_API__ < 23
		static inline float crealf(_Complex float c) {
			return __real__ c;
		}

		static inline float cimagf(_Complex float c) {
			return __imag__ c;
		}
	#endif
#endif
