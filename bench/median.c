#include <stddef.h>
#include <stdlib.h>
#include <nnpack.h>

static int compare_ulonglong(const void *a_ptr, const void *b_ptr) {
	const unsigned long long a = *((unsigned long long*) a_ptr);
	const unsigned long long b = *((unsigned long long*) b_ptr);
	if (a < b) {
		return -1;
	} else if (a > b) {
		return 1;
	} else {
		return 0;
	}
}

static int compare_profile(const void *a_ptr, const void *b_ptr) {
	const double a_total = ((const struct nnp_profile*) a_ptr)->total;
	const double b_total = ((const struct nnp_profile*) b_ptr)->total;
	if (a_total < b_total) {
		return -1;
	} else if (a_total > b_total) {
		return 1;
	} else {
		return 0;
	}
}

static inline unsigned long long average(unsigned long long a, unsigned long long b) {
	return (a / 2) + (b / 2) + (a & b & 1ull);
}

static inline struct nnp_profile average_profile(struct nnp_profile a, struct nnp_profile b) {
	return (struct nnp_profile) {
		.total = 0.5 * (a.total + b.total),
		.input_transform = 0.5 * (a.input_transform + b.input_transform),
		.kernel_transform = 0.5 * (a.kernel_transform + b.kernel_transform),
		.output_transform = 0.5 * (a.output_transform + b.output_transform),
		.block_multiplication = 0.5 * (a.block_multiplication + b.block_multiplication)
	};
}

unsigned long long median(unsigned long long array[], size_t length) {
	qsort(array, length, sizeof(unsigned long long), &compare_ulonglong);
	if (length % 2 == 0) {
		const unsigned long long median_lo = array[length / 2 - 1];
		const unsigned long long median_hi = array[length / 2];
		return average(median_lo, median_hi);
	} else {
		return array[length / 2];
	}
}

struct nnp_profile median_profile(struct nnp_profile array[], size_t length) {
	qsort(array, length, sizeof(struct nnp_profile), &compare_profile);
	if (length % 2 == 0) {
		return average_profile(array[length / 2 - 1], array[length / 2]);
	} else {
		return array[length / 2];
	}
}
