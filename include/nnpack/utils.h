#pragma once

#ifdef __cplusplus
	#include <cstddef>
#else
	#include <stddef.h>
#endif

static inline float maxf(float a, float b) {
	return a > b ? a : b;
}

static inline size_t doz(size_t a, size_t b) {
	return a > b ? a - b : 0;
}

static inline size_t max(size_t a, size_t b) {
	return a > b ? a : b;
}

static inline size_t min(size_t a, size_t b) {
	return a > b ? b : a;
}

static inline size_t round_up(size_t number, size_t factor) {
	return (number + factor - 1) / factor * factor;
}

static inline size_t round_up_by_power_of_2(size_t number, size_t power_of_2_factor) {
	return (number + power_of_2_factor - 1) & ~(power_of_2_factor - 1);
}

static inline size_t round_down(size_t number, size_t factor) {
	return number / factor * factor;
}

static inline size_t divide_round_up(size_t dividend, size_t divisor) {
	if (dividend % divisor == 0) {
		return dividend / divisor;
	} else {
		return dividend / divisor + 1;
	}
}
