#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reference versions */

typedef void (*nnp_strided_fft_function)(const float*, size_t, float*, size_t);

void nnp_fft2_aos__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft4_aos__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft8_aos__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft16_aos__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft32_aos__ref(const float t[], size_t t_stride, float f[], size_t f_stride);

void nnp_fft2_soa__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft4_soa__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft8_soa__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft16_soa__ref(const float t[], size_t t_stride, float f[], size_t f_stride);
void nnp_fft32_soa__ref(const float t[], size_t t_stride, float f[], size_t f_stride);

void nnp_ifft2_aos__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft4_aos__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft8_aos__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft16_aos__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft32_aos__ref(const float f[], size_t f_stride, float t[], size_t t_stride);

void nnp_ifft2_soa__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft4_soa__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft8_soa__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft16_soa__ref(const float f[], size_t f_stride, float t[], size_t t_stride);
void nnp_ifft32_soa__ref(const float f[], size_t f_stride, float t[], size_t t_stride);

void nnp_fft8_real__ref(const float t[], size_t f_stride, float f[], size_t t_stride);
void nnp_fft16_real__ref(const float t[], size_t f_stride, float f[], size_t t_stride);
void nnp_fft32_real__ref(const float t[], size_t f_stride, float f[], size_t t_stride);

void nnp_ifft8_real__ref(const float t[], size_t f_stride, float f[], size_t t_stride);
void nnp_ifft16_real__ref(const float t[], size_t f_stride, float f[], size_t t_stride);
void nnp_ifft32_real__ref(const float t[], size_t f_stride, float f[], size_t t_stride);

typedef void (*nnp_fft_function)(const float*, float*);

void nnp_fft8_dualreal__ref(const float t[], float f[]);
void nnp_fft16_dualreal__ref(const float t[], float f[]);
void nnp_fft32_dualreal__ref(const float t[], float f[]);

void nnp_ifft8_dualreal__ref(const float f[], float t[]);
void nnp_ifft16_dualreal__ref(const float f[], float t[]);
void nnp_ifft32_dualreal__ref(const float f[], float t[]);

/* Forward FFT within rows with SOA layout: used in the horizontal phase of 2D FFT */
void nnp_fft8_soa__avx2(const float t[], float f[]);
void nnp_fft16_soa__avx2(const float t[], float f[]);

void nnp_fft8_soa__psimd(const float t[], float f[]);
void nnp_fft16_soa__psimd(const float t[], float f[]);

void nnp_fft8_soa__scalar(const float t[], float f[]);
void nnp_fft16_soa__scalar(const float t[], float f[]);

/* Inverse FFT within rows with SOA layout: used in the horizontal phase of 2D IFFT */
void nnp_ifft8_soa__avx2(const float f[], float t[]);
void nnp_ifft16_soa__avx2(const float f[], float t[]);

void nnp_ifft8_soa__psimd(const float f[], float t[]);
void nnp_ifft16_soa__psimd(const float f[], float t[]);

void nnp_ifft8_soa__scalar(const float f[], float t[]);
void nnp_ifft16_soa__scalar(const float f[], float t[]);

/* Forward FFT across rows with SIMD AOS layout: used in the vertical phase of 2D FFT */
void nnp_fft4_8aos__fma3(const float t[], float f[]);
void nnp_fft8_8aos__fma3(const float t[], float f[]);

void nnp_fft4_4aos__psimd(const float t[], float f[]);
void nnp_fft8_4aos__psimd(const float t[], float f[]);

void nnp_fft4_aos__scalar(const float t[], float f[]);
void nnp_fft8_aos__scalar(const float t[], float f[]);

/* Inverse FFT across rows with SIMD AOS layout: used in the vertical phase of 2D IFFT */
void nnp_ifft8_8aos__fma3(const float f[], float t[]);

void nnp_ifft4_4aos__psimd(const float f[], float t[]);
void nnp_ifft8_4aos__psimd(const float f[], float t[]);

void nnp_ifft4_aos__scalar(const float f[], float t[]);
void nnp_ifft8_aos__scalar(const float f[], float t[]);

/* Forward real-to-complex FFT across rows with SIMD layout: used in the vertical phase of 2D FFT */
void nnp_fft8_8real__fma3(const float t[], float f[]);
void nnp_fft16_8real__fma3(const float t[], float f[]);

void nnp_fft8_4real__psimd(const float t[], float f[]);
void nnp_fft16_4real__psimd(const float t[], float f[]);

void nnp_fft8_real__scalar(const float t[], float f[]);
void nnp_fft16_real__scalar(const float t[], float f[]);

/* Inverse complex-to-real FFT across rows with SIMD layout: used in the vertical phase of 2D IFFT */
void nnp_ifft8_8real__fma3(const float f[], float t[]);
void nnp_ifft16_8real__fma3(const float f[], float t[]);

void nnp_ifft8_4real__psimd(const float f[], float t[]);
void nnp_ifft16_4real__psimd(const float f[], float t[]);

void nnp_ifft8_real__scalar(const float f[], float t[]);
void nnp_ifft16_real__scalar(const float f[], float t[]);



void nnp_fft8_dualreal__avx2(const float t[], float f[]);
void nnp_fft16_dualreal__avx2(const float t[], float f[]);

void nnp_fft8_dualreal__psimd(const float t[], float f[]);
void nnp_fft16_dualreal__psimd(const float t[], float f[]);

void nnp_fft8_dualreal__scalar(const float t[], float f[]);
void nnp_fft16_dualreal__scalar(const float t[], float f[]);

void nnp_ifft8_dualreal__avx2(const float f[], float t[]);
void nnp_ifft16_dualreal__avx2(const float f[], float t[]);

void nnp_ifft8_dualreal__psimd(const float f[], float t[]);
void nnp_ifft16_dualreal__psimd(const float f[], float t[]);

void nnp_ifft8_dualreal__scalar(const float f[], float t[]);
void nnp_ifft16_dualreal__scalar(const float f[], float t[]);

#ifdef __cplusplus
} /* extern "C" */
#endif
