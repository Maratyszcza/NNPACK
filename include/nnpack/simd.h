#pragma once

#include <nnpack/macros.h>

#if defined(__clang__)

typedef float v4f __attribute__((__vector_size__(16), __aligned__(1)));
typedef int v4i __attribute__((__vector_size__(16), __aligned__(1)));

static inline v4f v4f_ld(const void* address) {
    return *((const v4f*) address);
}

static inline v4f v4f_ld1(const void* address) {
	return (v4f) { *((const float*) address), 0.0f, 0.0f, 0.0f };
}

static inline v4f v4f_ld2(const void* address) {
	const float* f32_address = (const float*) address;
	return (v4f) { f32_address[0], f32_address[1], 0.0f, 0.0f };
}

static inline v4f v4f_ld3(const void* address) {
	const float* f32_address = (const float*) address;
	return (v4f) { f32_address[0], f32_address[1], f32_address[2], 0.0f };
}

static inline v4f v4f_ld4(const void* address) {
	return v4f_ld(address);
}

static inline void v4f_st(void* address, v4f value) {
    *((v4f*) address) = value;
}

static inline void v4f_st1(void* address, v4f value) {
	*((float*) address) = value[0];
}

static inline void v4f_st2(void* address, v4f value) {
	float* f32_address = (float*) address;
	f32_address[0] = value[0];
	f32_address[1] = value[1];
}

static inline void v4f_st3(void* address, v4f value) {
	float* f32_address = (float*) address;
	f32_address[0] = value[0];
	f32_address[1] = value[1];
	f32_address[2] = value[2];
}

static inline void v4f_st4(void* address, v4f value) {
	v4f_st(address, value);
}

static inline v4f v4f_zero(void) {
	return (v4f) { };
}

static inline v4f v4f_splat(float c) {
	return (v4f) { c, c, c, c };
}

static inline v4i v4i_splat(int c) {
	return (v4i) { c, c, c, c };
}

static inline v4f v4f_andi(v4f v, v4i mask) {
	return (v4f) (((v4i) v) & mask);
}

/*
 * return (mask ? a : b)
 */
static inline v4f v4f_blend(v4i mask, v4f a, v4f b) {
	return (v4f) ((mask & ((v4i) a)) | (~mask & ((v4i) b)));
}

static inline v4i v4i_blend(v4i mask, v4i a, v4i b) {
	return (mask & a) | (~mask & b);
}

/*
 * return signbit(x) ? a : b;
 */
static inline v4f v4f_signblend(v4f x, v4f a, v4f b) {
	const v4i mask = ((v4i) x) >> ((v4i) { 31, 31, 31, 31 });
	return (v4f) ((mask & ((v4i) a)) | (~mask & ((v4i) b)));
}

static inline v4f v4f_max(v4f a, v4f b) {
	return v4f_blend(a > b, a, b);
}

static inline v4f v4f_min(v4f a, v4f b) {
	return v4f_blend(a < b, a, b);
}

static inline v4i v4i_max(v4i a, v4i b) {
	return v4i_blend(a > b, a, b);
}

static inline v4i v4i_min(v4i a, v4i b) {
	return v4i_blend(a < b, a, b);
}

static inline float v4f_reduce_sum(v4f v) {
	v = v + __builtin_shufflevector(v, v, 2, 3, -1, -1);
	return v[0] + v[1];
}

static inline float v4f_reduce_max(v4f v) {
	v = v4f_max(v, __builtin_shufflevector(v, v, 2, 3, -1, -1));
	const float v0 = v[0];
	const float v1 = v[1];
	return v0 > v1 ? v0 : v1;
}

static inline float v4f_reduce_min(v4f v) {
	v = v4f_min(v, __builtin_shufflevector(v, v, 2, 3, -1, -1));
	const float v0 = v[0];
	const float v1 = v[1];
	return v0 < v1 ? v0 : v1;
}

static inline void v4f_swap(v4f a[restrict static 1], v4f b[restrict static 1]) {
    const v4f new_a = *b;
    const v4f new_b = *a;
    *a = new_a;
    *b = new_b;
}

#endif
