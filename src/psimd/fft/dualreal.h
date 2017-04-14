#pragma once

#include <nnpack/fft-constants.h>
#include <psimd.h>
#include <psimd/butterfly.h>
#include <psimd/fft/soa.h>


static inline void psimd_fft8_dualreal_f32(
	psimd_f32 s0123[restrict static 1],
	psimd_f32 s4567[restrict static 1],
	psimd_f32 h0123[restrict static 1],
	psimd_f32 h4567[restrict static 1])
{
	psimd_fft8_soa_f32(s0123, s4567, h0123, h4567);

	psimd_f32 x0123 = *s0123, x4567 = *s4567;
	psimd_f32 y0123 = *h0123, y4567 = *h4567;

	/*
	 * Target:
	 *   x0 y0 .5(x1+x7) .5(y1+y7) .5(x2+x6) .5(y2+y6) .5(x3+x5) .5(y3+y5) 
	 *   x4 y4 .5(y1-y7) .5(x7-x1) .5(y2-y6) .5(x6-x2) .5(y3-y5) .5(x5-x3)
	 */

	#ifdef __clang__
		const psimd_f32 x0765 = __builtin_shufflevector(x4567, x0123, 4, 3, 2, 1);
		const psimd_f32 y0765 = __builtin_shufflevector(y4567, y0123, 4, 3, 2, 1);
	#else
		const psimd_f32 x0765 = __builtin_shuffle(x4567, x0123, (psimd_s32) { 4, 3, 2, 1 });
		const psimd_f32 y0765 = __builtin_shuffle(y4567, y0123, (psimd_s32) { 4, 3, 2, 1 });
	#endif

	const psimd_f32 half = psimd_splat_f32(0.5f);
	const psimd_f32 r0246 = half * (x0123 + x0765);
	const psimd_f32 r1357 = half * (y0123 + y0765);

	const psimd_f32 i_246 = half * (y0123 - y0765);
	const psimd_f32 i_357 = half * (x0765 - x0123);

	#ifdef __clang__
		const psimd_f32 i0246 = __builtin_shufflevector(i_246, x4567, 4, 1, 2, 3);
		const psimd_f32 i1357 = __builtin_shufflevector(i_357, y4567, 4, 1, 2, 3);
	#else
		const psimd_f32 i0246 = __builtin_shuffle(i_246, x4567, (psimd_s32) { 4, 1, 2, 3 });
		const psimd_f32 i1357 = __builtin_shuffle(i_357, y4567, (psimd_s32) { 4, 1, 2, 3 });
	#endif

	/* Interleave and store */
	*s0123 = psimd_interleave_lo_f32(r0246, r1357);
	*s4567 = psimd_interleave_hi_f32(r0246, r1357);
	*h0123 = psimd_interleave_lo_f32(i0246, i1357);
	*h4567 = psimd_interleave_hi_f32(i0246, i1357);
}

static inline void psimd_fft16_dualreal_f32(
	psimd_f32 s0123[restrict static 1],
	psimd_f32 s4567[restrict static 1],
	psimd_f32 s89AB[restrict static 1],
	psimd_f32 sCDEF[restrict static 1],
	psimd_f32 h0123[restrict static 1],
	psimd_f32 h4567[restrict static 1],
	psimd_f32 h89AB[restrict static 1],
	psimd_f32 hCDEF[restrict static 1])
{
	psimd_fft16_soa_f32(s0123, s4567, s89AB, sCDEF, h0123, h4567, h89AB, hCDEF);

	psimd_f32 x0123 = *s0123, x4567 = *s4567, x89AB = *s89AB, xCDEF = *sCDEF;
	psimd_f32 y0123 = *h0123, y4567 = *h4567, y89AB = *h89AB, yCDEF = *hCDEF;

	#ifdef __clang__
		const psimd_f32 x0FED = __builtin_shufflevector(xCDEF, x0123, 4, 3, 2, 1);
		const psimd_f32 y0FED = __builtin_shufflevector(yCDEF, y0123, 4, 3, 2, 1);
		const psimd_f32 xCBA9 = __builtin_shufflevector(x89AB, xCDEF, 4, 3, 2, 1);
		const psimd_f32 yCBA9 = __builtin_shufflevector(y89AB, yCDEF, 4, 3, 2, 1);
	#else
		const psimd_f32 x0FED = __builtin_shuffle(xCDEF, x0123, (psimd_s32) { 4, 3, 2, 1 });
		const psimd_f32 y0FED = __builtin_shuffle(yCDEF, y0123, (psimd_s32) { 4, 3, 2, 1 });
		const psimd_f32 xCBA9 = __builtin_shuffle(x89AB, xCDEF, (psimd_s32) { 4, 3, 2, 1 });
		const psimd_f32 yCBA9 = __builtin_shuffle(y89AB, yCDEF, (psimd_s32) { 4, 3, 2, 1 });
	#endif

	const psimd_f32 half = psimd_splat_f32(0.5f);
	const psimd_f32 r0246 = half * (x0123 + x0FED);
	const psimd_f32 r1357 = half * (y0123 + y0FED);
	const psimd_f32 r8ACE = half * (x4567 + xCBA9);
	const psimd_f32 r9BDF = half * (y4567 + yCBA9);

	const psimd_f32 i_246 = half * (y0123 - y0FED);
	const psimd_f32 i_357 = half * (x0FED - x0123);
	const psimd_f32 i8ACE = half * (y4567 - yCBA9);
	const psimd_f32 i9BDF = half * (xCBA9 - x4567);

	#ifdef __clang__
		const psimd_f32 i0246 = __builtin_shufflevector(i_246, x89AB, 4, 1, 2, 3);
		const psimd_f32 i1357 = __builtin_shufflevector(i_357, y89AB, 4, 1, 2, 3);
	#else
		const psimd_f32 i0246 = __builtin_shuffle(i_246, x89AB, (psimd_s32) { 4, 1, 2, 3 });
		const psimd_f32 i1357 = __builtin_shuffle(i_357, y89AB, (psimd_s32) { 4, 1, 2, 3 });
	#endif

	/* Interleave and store */
	*s0123 = psimd_interleave_lo_f32(r0246, r1357);
	*s4567 = psimd_interleave_hi_f32(r0246, r1357);
	*s89AB = psimd_interleave_lo_f32(r8ACE, r9BDF);
	*sCDEF = psimd_interleave_hi_f32(r8ACE, r9BDF);
	*h0123 = psimd_interleave_lo_f32(i0246, i1357);
	*h4567 = psimd_interleave_hi_f32(i0246, i1357);
	*h89AB = psimd_interleave_lo_f32(i8ACE, i9BDF);
	*hCDEF = psimd_interleave_hi_f32(i8ACE, i9BDF);
}

static inline void psimd_ifft8_dualreal_f32(
	psimd_f32 s0123[restrict static 1],
	psimd_f32 s4567[restrict static 1],
	psimd_f32 h0123[restrict static 1],
	psimd_f32 h4567[restrict static 1])
{
	const psimd_f32 r0123 = *s0123;
	const psimd_f32 r4567 = *s4567;
	const psimd_f32 i0123 = *h0123;
	const psimd_f32 i4567 = *h4567;

	const psimd_f32 r0246 = psimd_concat_even_f32(r0123, r4567);
	const psimd_f32 r1357 = psimd_concat_odd_f32(r0123, r4567);
	const psimd_f32 i0246 = psimd_concat_even_f32(i0123, i4567);
	const psimd_f32 i1357 = psimd_concat_odd_f32(i0123, i4567);

	#ifdef __clang__
		*s0123 = __builtin_shufflevector(r0246 - i1357, r0246, 4, 1, 2, 3);
		*s4567 = __builtin_shufflevector(r0246 + i1357, i0246, 4, 3, 2, 1);
		*h0123 = __builtin_shufflevector(r1357 + i0246, r1357, 4, 1, 2, 3);
		*h4567 = __builtin_shufflevector(r1357 - i0246, i1357, 4, 3, 2, 1);
	#else
		*s0123 = __builtin_shuffle(r0246 - i1357, r0246, (psimd_s32) { 4, 1, 2, 3 });
		*s4567 = __builtin_shuffle(r0246 + i1357, i0246, (psimd_s32) { 4, 3, 2, 1 });
		*h0123 = __builtin_shuffle(r1357 + i0246, r1357, (psimd_s32) { 4, 1, 2, 3 });
		*h4567 = __builtin_shuffle(r1357 - i0246, i1357, (psimd_s32) { 4, 3, 2, 1 });
	#endif

	psimd_ifft8_soa_f32(s0123, s4567, h0123, h4567);
}

static inline void psimd_ifft16_dualreal_f32(
	psimd_f32 s0123[restrict static 1],
	psimd_f32 s4567[restrict static 1],
	psimd_f32 s89AB[restrict static 1],
	psimd_f32 sCDEF[restrict static 1],
	psimd_f32 h0123[restrict static 1],
	psimd_f32 h4567[restrict static 1],
	psimd_f32 h89AB[restrict static 1],
	psimd_f32 hCDEF[restrict static 1])
{
	const psimd_f32 r0123 = *s0123;
	const psimd_f32 r4567 = *s4567;
	const psimd_f32 r89AB = *s89AB;
	const psimd_f32 rCDEF = *sCDEF;
	const psimd_f32 i0123 = *h0123;
	const psimd_f32 i4567 = *h4567;
	const psimd_f32 i89AB = *h89AB;
	const psimd_f32 iCDEF = *hCDEF;

	const psimd_f32 r0246 = psimd_concat_even_f32(r0123, r4567);
	const psimd_f32 r1357 = psimd_concat_odd_f32(r0123, r4567);
	const psimd_f32 r8ACE = psimd_concat_even_f32(r89AB, rCDEF);
	const psimd_f32 r9BDF = psimd_concat_odd_f32(r89AB, rCDEF);
	const psimd_f32 i0246 = psimd_concat_even_f32(i0123, i4567);
	const psimd_f32 i1357 = psimd_concat_odd_f32(i0123, i4567);
	const psimd_f32 i8ACE = psimd_concat_even_f32(i89AB, iCDEF);
	const psimd_f32 i9BDF = psimd_concat_odd_f32(i89AB, iCDEF);

	const psimd_f32 sCBA9 = r8ACE + i9BDF;
	const psimd_f32 s_FED = r0246 + i1357;
	const psimd_f32 hCBA9 = r9BDF - i8ACE;
	const psimd_f32 h_FED = r1357 - i0246;

	#ifdef __clang__
		*s0123 = __builtin_shufflevector(r0246 - i1357, r0246, 4, 1, 2, 3);
	#else
		*s0123 = __builtin_shuffle(r0246 - i1357, r0246, (psimd_s32) { 4, 1, 2, 3 });
	#endif
	*s4567 = r8ACE - i9BDF;
	#ifdef __clang__
		*s89AB = __builtin_shufflevector(sCBA9, i0246, 4, 3, 2, 1);
		*sCDEF = __builtin_shufflevector(s_FED, sCBA9, 4, 3, 2, 1);
		*h0123 = __builtin_shufflevector(r1357 + i0246, r1357, 4, 1, 2, 3);
	#else
		*s89AB = __builtin_shuffle(sCBA9, i0246, (psimd_s32) { 4, 3, 2, 1 });
		*sCDEF = __builtin_shuffle(s_FED, sCBA9, (psimd_s32) { 4, 3, 2, 1 });
		*h0123 = __builtin_shuffle(r1357 + i0246, r1357, (psimd_s32) { 4, 1, 2, 3 });
	#endif
	*h4567 = r9BDF + i8ACE;
	#ifdef __clang__
		*h89AB = __builtin_shufflevector(hCBA9, i1357, 4, 3, 2, 1);
		*hCDEF = __builtin_shufflevector(h_FED, hCBA9, 4, 3, 2, 1);
	#else
		*h89AB = __builtin_shuffle(hCBA9, i1357, (psimd_s32) { 4, 3, 2, 1 });
		*hCDEF = __builtin_shuffle(h_FED, hCBA9, (psimd_s32) { 4, 3, 2, 1 });
	#endif

	psimd_ifft16_soa_f32(s0123, s4567, s89AB, sCDEF, h0123, h4567, h89AB, hCDEF);
}
