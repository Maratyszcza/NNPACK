#pragma once


#if defined(__GNUC__)
	#if defined(__clang__) || ((__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5)))
		#define NNP_UNREACHABLE do { __builtin_unreachable(); } while (0)
	#else
		#define NNP_UNREACHABLE do { __builtin_trap(); } while (0)
	#endif
#else
	#define NNP_UNREACHABLE do { } while (0)
#endif


#define NNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define NNP_SIMD_ALIGN NNP_ALIGN(64)
#define NNP_CACHE_ALIGN NNP_ALIGN(64)
