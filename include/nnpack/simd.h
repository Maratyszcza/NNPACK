#pragma once

#include <nnpack/macros.h>

#if defined(__pnacl__)
typedef float v4f __attribute__((vector_size(16), aligned(1)));

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

static inline v4f v4f_splat(float c) {
	return (v4f) { c, c, c, c };
}
#endif
