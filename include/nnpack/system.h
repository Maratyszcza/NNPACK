#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#if defined(__linux__) || defined(__native_client__)
	#include <time.h>
	#include <unistd.h>
	#include <sys/mman.h>
#elif defined(__MACH__)
	#include <mach/mach.h>
	#include <mach/mach_time.h>
#elif defined(EMSCRIPTEN)
	#include <emscripten.h>
#endif

inline static double read_timer() {
#if defined(__linux__) || defined(__native_client__)
	struct timespec ts;
	int result = clock_gettime(CLOCK_MONOTONIC, &ts);
	assert(result == 0);
	return ((double) ts.tv_sec) + ((double) ts.tv_nsec) * 1.0e-9;
#elif defined(__MACH__)
	static mach_timebase_info_data_t timebase_info;
	if (timebase_info.denom == 0) {
		mach_timebase_info(&timebase_info);
	}

	return ((double) (mach_absolute_time() * timebase_info.numer / timebase_info.denom)) * 1.0e-9;
#elif defined(EMSCRIPTEN)
	return emscripten_get_now() * 1.0e-3;
#else
	#error No implementation available
#endif
}

#define NNP_TOTAL_START(profile_ptr) \
	double total_start; \
	if (profile_ptr != NULL) { \
		*profile_ptr = (struct nnp_profile) { 0 }; \
		total_start = read_timer(); \
	}

#define NNP_KERNEL_TRANSFORM_START(profile_ptr) \
	double kernel_transform_start; \
	if (profile_ptr != NULL) { \
		kernel_transform_start = read_timer(); \
	}

#define NNP_INPUT_TRANSFORM_START(profile_ptr) \
	double input_transform_start; \
	if (profile_ptr != NULL) { \
		input_transform_start = read_timer(); \
	}

#define NNP_OUTPUT_TRANSFORM_START(profile_ptr) \
	double output_transform_start; \
	if (profile_ptr != NULL) { \
		output_transform_start = read_timer(); \
	}

#define NNP_BLOCK_MULTIPLICATION_START(profile_ptr) \
	double block_multiplication_start; \
	if (profile_ptr != NULL) { \
		block_multiplication_start = read_timer(); \
	}

#define NNP_TOTAL_END(profile_ptr) \
	if (profile_ptr != NULL) { \
		profile_ptr->total = read_timer() - total_start; \
	}

#define NNP_KERNEL_TRANSFORM_END(profile_ptr) \
	if (profile_ptr != NULL) { \
		profile_ptr->kernel_transform += read_timer() - kernel_transform_start; \
	}

#define NNP_INPUT_TRANSFORM_END(profile_ptr) \
	if (profile_ptr != NULL) { \
		profile_ptr->input_transform += read_timer() - input_transform_start; \
	}

#define NNP_OUTPUT_TRANSFORM_END(profile_ptr) \
	if (profile_ptr != NULL) { \
		profile_ptr->output_transform += read_timer() - output_transform_start; \
	}

#define NNP_BLOCK_MULTIPLICATION_END(profile_ptr) \
	if (profile_ptr != NULL) { \
		profile_ptr->block_multiplication += read_timer() - block_multiplication_start; \
	}

inline static void* allocate_memory(size_t memory_size) {
#if defined(__linux__)
	#if !defined(__ANDROID__)
		/* Try to use large page TLB */
		void* memory_block = mmap(NULL, memory_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);
	#else
		void* memory_block = MAP_FAILED;
	#endif
	if (memory_block == MAP_FAILED) {
		/* Fallback to standard pages */
		memory_block = mmap(NULL, memory_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
		if (memory_block == MAP_FAILED) {
			return NULL;
		}
	}
	return memory_block;
#else
	void* memory_block = NULL;
	int allocation_result = posix_memalign(&memory_block, 64, memory_size);
	return (allocation_result == 0) ? memory_block : NULL;
#endif
}

inline static void release_memory(void* memory_block, size_t memory_size) {
#if defined(__linux__)
	if (memory_block != NULL) {
		munmap(memory_block, memory_size);
	}
#else
	free(memory_block);
#endif
}
