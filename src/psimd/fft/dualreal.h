#pragma once

#include <nnpack/simd.h>
#include <nnpack/fft-constants.h>
#include <psimd/butterfly.h>
#include <psimd/fft/soa.h>


static inline void v4f_fft8_dualreal(
	v4f s0123[restrict static 1],
	v4f s4567[restrict static 1],
	v4f h0123[restrict static 1],
	v4f h4567[restrict static 1])
{
	v4f_fft8_soa(s0123, s4567, h0123, h4567);

	v4f x0123 = *s0123, x4567 = *s4567;
	v4f y0123 = *h0123, y4567 = *h4567;

	/*
	 * Target:
	 *   x0 y0 .5(x1+x7) .5(y1+y7) .5(x2+x6) .5(y2+y6) .5(x3+x5) .5(y3+y5) 
	 *   x4 y4 .5(y1-y7) .5(x7-x1) .5(y2-y6) .5(x6-x2) .5(y3-y5) .5(x5-x3)
	 */

	const v4f x0765 = __builtin_shufflevector(x4567, x0123, 4, 3, 2, 1);
	const v4f y0765 = __builtin_shufflevector(y4567, y0123, 4, 3, 2, 1);

	const v4f half = v4f_splat(0.5f);
	const v4f r0246 = half * (x0123 + x0765);
	const v4f r1357 = half * (y0123 + y0765);

	const v4f i_246 = half * (y0123 - y0765);
	const v4f i_357 = half * (x0765 - x0123);

	const v4f i0246 = __builtin_shufflevector(i_246, x4567, 4, 1, 2, 3);
	const v4f i1357 = __builtin_shufflevector(i_357, y4567, 4, 1, 2, 3);

	/* Interleave and store */
	*s0123 = __builtin_shufflevector(r0246, r1357, 0, 4, 1, 5);
	*s4567 = __builtin_shufflevector(r0246, r1357, 2, 6, 3, 7);
	*h0123 = __builtin_shufflevector(i0246, i1357, 0, 4, 1, 5);
	*h4567 = __builtin_shufflevector(i0246, i1357, 2, 6, 3, 7);
}

static inline void v4f_fft16_dualreal(
	v4f s0123[restrict static 1],
	v4f s4567[restrict static 1],
	v4f s89AB[restrict static 1],
	v4f sCDEF[restrict static 1],
	v4f h0123[restrict static 1],
	v4f h4567[restrict static 1],
	v4f h89AB[restrict static 1],
	v4f hCDEF[restrict static 1])
{
	v4f_fft16_soa(s0123, s4567, s89AB, sCDEF, h0123, h4567, h89AB, hCDEF);

	v4f x0123 = *s0123, x4567 = *s4567, x89AB = *s89AB, xCDEF = *sCDEF;
	v4f y0123 = *h0123, y4567 = *h4567, y89AB = *h89AB, yCDEF = *hCDEF;

	const v4f x0FED = __builtin_shufflevector(xCDEF, x0123, 4, 3, 2, 1);
	const v4f y0FED = __builtin_shufflevector(yCDEF, y0123, 4, 3, 2, 1);
	const v4f xCBA9 = __builtin_shufflevector(x89AB, xCDEF, 4, 3, 2, 1);
	const v4f yCBA9 = __builtin_shufflevector(y89AB, yCDEF, 4, 3, 2, 1);

	const v4f half = v4f_splat(0.5f);
	const v4f r0246 = half * (x0123 + x0FED);
	const v4f r1357 = half * (y0123 + y0FED);
	const v4f r8ACE = half * (x4567 + xCBA9);
	const v4f r9BDF = half * (y4567 + yCBA9);

	const v4f i_246 = half * (y0123 - y0FED);
	const v4f i_357 = half * (x0FED - x0123);
	const v4f i8ACE = half * (y4567 - yCBA9);
	const v4f i9BDF = half * (xCBA9 - x4567);

	const v4f i0246 = __builtin_shufflevector(i_246, x89AB, 4, 1, 2, 3);
	const v4f i1357 = __builtin_shufflevector(i_357, y89AB, 4, 1, 2, 3);

	/* Interleave and store */
	*s0123 = __builtin_shufflevector(r0246, r1357, 0, 4, 1, 5);
	*s4567 = __builtin_shufflevector(r0246, r1357, 2, 6, 3, 7);
	*s89AB = __builtin_shufflevector(r8ACE, r9BDF, 0, 4, 1, 5);
	*sCDEF = __builtin_shufflevector(r8ACE, r9BDF, 2, 6, 3, 7);
	*h0123 = __builtin_shufflevector(i0246, i1357, 0, 4, 1, 5);
	*h4567 = __builtin_shufflevector(i0246, i1357, 2, 6, 3, 7);
	*h89AB = __builtin_shufflevector(i8ACE, i9BDF, 0, 4, 1, 5);
	*hCDEF = __builtin_shufflevector(i8ACE, i9BDF, 2, 6, 3, 7);
}

static inline void v4f_ifft8_dualreal(
	v4f s0123[restrict static 1],
	v4f s4567[restrict static 1],
	v4f h0123[restrict static 1],
	v4f h4567[restrict static 1])
{
	const v4f r0123 = *s0123;
	const v4f r4567 = *s4567;
	const v4f i0123 = *h0123;
	const v4f i4567 = *h4567;

	const v4f r0246 = __builtin_shufflevector(r0123, r4567, 0, 2, 4, 6);
	const v4f r1357 = __builtin_shufflevector(r0123, r4567, 1, 3, 5, 7);
	const v4f i0246 = __builtin_shufflevector(i0123, i4567, 0, 2, 4, 6);
	const v4f i1357 = __builtin_shufflevector(i0123, i4567, 1, 3, 5, 7);

	*s0123 = __builtin_shufflevector(r0246 - i1357, r0246, 4, 1, 2, 3);
	*s4567 = __builtin_shufflevector(r0246 + i1357, i0246, 4, 3, 2, 1);
	*h0123 = __builtin_shufflevector(r1357 + i0246, r1357, 4, 1, 2, 3);
	*h4567 = __builtin_shufflevector(r1357 - i0246, i1357, 4, 3, 2, 1);

	v4f_ifft8_soa(s0123, s4567, h0123, h4567);
}

static inline void v4f_ifft16_dualreal(
	v4f s0123[restrict static 1],
	v4f s4567[restrict static 1],
	v4f s89AB[restrict static 1],
	v4f sCDEF[restrict static 1],
	v4f h0123[restrict static 1],
	v4f h4567[restrict static 1],
	v4f h89AB[restrict static 1],
	v4f hCDEF[restrict static 1])
{
	const v4f r0123 = *s0123;
	const v4f r4567 = *s4567;
	const v4f r89AB = *s89AB;
	const v4f rCDEF = *sCDEF;
	const v4f i0123 = *h0123;
	const v4f i4567 = *h4567;
	const v4f i89AB = *h89AB;
	const v4f iCDEF = *hCDEF;

	const v4f r0246 = __builtin_shufflevector(r0123, r4567, 0, 2, 4, 6);
	const v4f r1357 = __builtin_shufflevector(r0123, r4567, 1, 3, 5, 7);
	const v4f r8ACE = __builtin_shufflevector(r89AB, rCDEF, 0, 2, 4, 6);
	const v4f r9BDF = __builtin_shufflevector(r89AB, rCDEF, 1, 3, 5, 7);
	const v4f i0246 = __builtin_shufflevector(i0123, i4567, 0, 2, 4, 6);
	const v4f i1357 = __builtin_shufflevector(i0123, i4567, 1, 3, 5, 7);
	const v4f i8ACE = __builtin_shufflevector(i89AB, iCDEF, 0, 2, 4, 6);
	const v4f i9BDF = __builtin_shufflevector(i89AB, iCDEF, 1, 3, 5, 7);

	const v4f sCBA9 = r8ACE + i9BDF;
	const v4f s_FED = r0246 + i1357;
	const v4f hCBA9 = r9BDF - i8ACE;
	const v4f h_FED = r1357 - i0246;

	*s0123 = __builtin_shufflevector(r0246 - i1357, r0246, 4, 1, 2, 3);
	*s4567 = r8ACE - i9BDF;
	*s89AB = __builtin_shufflevector(sCBA9, i0246, 4, 3, 2, 1);
	*sCDEF = __builtin_shufflevector(s_FED, sCBA9, 4, 3, 2, 1);
	*h0123 = __builtin_shufflevector(r1357 + i0246, r1357, 4, 1, 2, 3);
	*h4567 = r9BDF + i8ACE;
	*h89AB = __builtin_shufflevector(hCBA9, i1357, 4, 3, 2, 1);
	*hCDEF = __builtin_shufflevector(h_FED, hCBA9, 4, 3, 2, 1);

	v4f_ifft16_soa(s0123, s4567, s89AB, sCDEF, h0123, h4567, h89AB, hCDEF);
}
