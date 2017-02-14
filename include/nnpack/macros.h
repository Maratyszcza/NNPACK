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


#if defined(NNP_ARCH_PSIMD)
	#if !(NNP_ARCH_PSIMD)
		#error NNP_ARCH_PSIMD predefined as 0
	#endif
	#define NNP_ARCH_SCALAR 0
	#define NNP_ARCH_X86_64 0
	#define NNP_ARCH_ARM 0
	#define NNP_ARCH_ARM64 0
#elif defined(NNP_ARCH_SCALAR)
	#if !(NNP_ARCH_SCALAR)
		#error NNP_ARCH_SCALAR predefined as 0
	#endif
	#define NNP_ARCH_PSIMD 0
	#define NNP_ARCH_X86_64 0
	#define NNP_ARCH_ARM 0
	#define NNP_ARCH_ARM64 0
#elif defined(__pnacl__)
	#define NNP_ARCH_PSIMD 1
	#define NNP_ARCH_SCALAR 0
	#define NNP_ARCH_X86_64 0
	#define NNP_ARCH_ARM 0
	#define NNP_ARCH_ARM64 0
#elif defined(__arm__)
	#define NNP_ARCH_ARM 1
	#define NNP_ARCH_SCALAR 0
	#define NNP_ARCH_PSIMD 0
	#define NNP_ARCH_X86_64 0
	#define NNP_ARCH_ARM64 0
#elif defined(__aarch64__)
	#define NNP_ARCH_ARM64 1
	#define NNP_ARCH_SCALAR 0
	#define NNP_ARCH_PSIMD 0
	#define NNP_ARCH_X86_64 0
	#define NNP_ARCH_ARM 0
#elif defined(__x86_64__)
	#define NNP_ARCH_X86_64 1
	#define NNP_ARCH_SCALAR 0
	#define NNP_ARCH_PSIMD 0
	#define NNP_ARCH_ARM 0
	#define NNP_ARCH_ARM64 0
#else
	#error Unknown target architecture
#endif

#define NNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define NNP_SIMD_ALIGN NNP_ALIGN(64)
#define NNP_CACHE_ALIGN NNP_ALIGN(64)

#define NNP_COUNT_OF(array) (sizeof(array) / sizeof(0[array]))
