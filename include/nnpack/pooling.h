#pragma once

#include <stddef.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nnp_pooling_function)(const float*, float*, size_t, size_t, size_t, size_t, size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t);

void nnp_maxpool_2x2_2x2__avx2(const float* src_pointer, float* dst_pointer, size_t src_stride,
	uint32_t src_row_offset, uint32_t src_row_count, uint32_t src_column_offset, uint32_t src_column_count, uint32_t dst_column_count);

#ifdef __cplusplus
} /* extern "C" */
#endif
