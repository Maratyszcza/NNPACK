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


#if defined(NNP_BACKEND_PSIMD)
	#if !(NNP_BACKEND_PSIMD)
		#error NNP_BACKEND_PSIMD predefined as 0
	#endif
#elif defined(NNP_BACKEND_SCALAR)
	#if !(NNP_BACKEND_SCALAR)
		#error NNP_BACKEND_SCALAR predefined as 0
	#endif
#elif defined(__arm__) || defined(__aarch64__)
	#define NNP_BACKEND_ARM 1
#elif defined(__ANDROID__) && (defined(__i686__) || defined(__x86_64__))
	#define NNP_BACKEND_PSIMD 1
#elif defined(__x86_64__)
	#define NNP_BACKEND_X86_64 1
#elif defined(__ANDROID__) && defined(__mips__)
	#define NNP_BACKEND_SCALAR 1
#else
	#define NNP_BACKEND_PSIMD 1
#endif

#ifndef NNP_BACKEND_PSIMD
	#define NNP_BACKEND_PSIMD 0
#endif
#ifndef NNP_BACKEND_SCALAR
	#define NNP_BACKEND_SCALAR 0
#endif
#ifndef NNP_BACKEND_ARM
	#define NNP_BACKEND_ARM 0
#endif
#ifndef NNP_BACKEND_X86_64
	#define NNP_BACKEND_X86_64 0
#endif

#define NNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define NNP_SIMD_ALIGN NNP_ALIGN(64)
#define NNP_CACHE_ALIGN NNP_ALIGN(64)

#define NNP_COUNT_OF(array) (sizeof(array) / sizeof(0[array]))

#if defined(__GNUC__)
	#define NNP_LIKELY(condition) (__builtin_expect(!!(condition), 1))
	#define NNP_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
	#define NNP_LIKELY(condition) (!!(condition))
	#define NNP_UNLIKELY(condition) (!!(condition))
#endif

#if defined(__GNUC__)
	#define NNP_INLINE inline __attribute__((__always_inline__))
#else
	#define NNP_INLINE inline
#endif
