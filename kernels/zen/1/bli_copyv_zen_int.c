/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name(s) of the copyright holder(s) nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "immintrin.h"
#include "blis.h"

// -----------------------------------------------------------------------------

void bli_scopyv_zen_int
(
	conj_t           conjx,
	dim_t            n,
	float*  restrict x, inc_t incx,
	float*  restrict y, inc_t incy,
	cntx_t* restrict cntx
)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)

	const dim_t   num_elem_per_reg = 8;
	__m256  xv[16];
	dim_t i = 0;

	// If the vector dimension is zero return early.
	if (bli_zero_dim1(n))
	{
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
		return;
	}

	if (incx == 1 && incy == 1)
	{
#if 0
	  PRAGMA_SIMD
	  for (i = 0; i < n; i++)
	  {
	    y[i] = x[i];
	  }
#endif
#if 0
        memcpy(y, x, n << 2);
#endif
#if 1

		// For loop with n & ~0x7F => n & 0xFFFFFF80 masks the lower bits and results in multiples of 128
		// for example if n = 255
		// n & ~0x7F results in 128: copy from 0 to 128 happens in first loop
		// n & ~0x3F results in 192: copy from 128 to 192 happens in second loop
		// n & ~0x1F results in 224: copy from 128 to 192 happens in third loop and so on.
		for (i = 0; i < (n & (~0x7F)); i += 128)
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_ps(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_ps(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_ps(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_ps(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_ps(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_ps(x + num_elem_per_reg * 7);
			xv[8] = _mm256_loadu_ps(x + num_elem_per_reg * 8);
			xv[9] = _mm256_loadu_ps(x + num_elem_per_reg * 9);
			xv[10] = _mm256_loadu_ps(x + num_elem_per_reg * 10);
			xv[11] = _mm256_loadu_ps(x + num_elem_per_reg * 11);
			xv[12] = _mm256_loadu_ps(x + num_elem_per_reg * 12);
			xv[13] = _mm256_loadu_ps(x + num_elem_per_reg * 13);
			xv[14] = _mm256_loadu_ps(x + num_elem_per_reg * 14);
			xv[15] = _mm256_loadu_ps(x + num_elem_per_reg * 15);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_ps(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_ps(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_ps(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_ps(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_ps(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_ps(y + num_elem_per_reg * 7, xv[7]);
			_mm256_storeu_ps(y + num_elem_per_reg * 8, xv[8]);
			_mm256_storeu_ps(y + num_elem_per_reg * 9, xv[9]);
			_mm256_storeu_ps(y + num_elem_per_reg * 10, xv[10]);
			_mm256_storeu_ps(y + num_elem_per_reg * 11, xv[11]);
			_mm256_storeu_ps(y + num_elem_per_reg * 12, xv[12]);
			_mm256_storeu_ps(y + num_elem_per_reg * 13, xv[13]);
			_mm256_storeu_ps(y + num_elem_per_reg * 14, xv[14]);
			_mm256_storeu_ps(y + num_elem_per_reg * 15, xv[15]);

			y += 128;
			x += 128;
		}
		for (; i < (n & (~0x3F)); i += 64)
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_ps(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_ps(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_ps(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_ps(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_ps(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_ps(x + num_elem_per_reg * 7);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_ps(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_ps(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_ps(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_ps(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_ps(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_ps(y + num_elem_per_reg * 7, xv[7]);

			y += 64;
			x += 64;
		}
		for (; i < (n & (~0x1F)); i += 32)
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_ps(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_ps(x + num_elem_per_reg * 3);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_ps(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_ps(y + num_elem_per_reg * 3, xv[3]);

			y += 32;
			x += 32;
		}
		for (; i < (n & (~0x0F)); i += 16)
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);

			y += 16;
			x += 16;
		}
		for (; i < (n & (~0x07)); i += 8)
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			y += 8;
			x += 8;
		}
		for (; i < n; i++)
		{
			*y++ = *x++;
		}
#endif
	}
	else
	{
		for (dim_t i = 0; i < n; ++i)
		{
			*y = *x;
			x += incx;
			y += incy;
		}
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}

// -----------------------------------------------------------------------------

void bli_dcopyv_zen_int
(
	conj_t           conjx,
	dim_t            n,
	double*  restrict x, inc_t incx,
	double*  restrict y, inc_t incy,
	cntx_t* restrict cntx
)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)
	const dim_t      num_elem_per_reg = 4;
	__m256d  xv[16];
	dim_t i = 0;

	// If the vector dimension is zero return early.
	if (bli_zero_dim1(n))
	{
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
		return;
	}

	if (incx == 1 && incy == 1)
	{
#if 0
		PRAGMA_SIMD
			for (i = 0; i < n; ++i)
			{
				y[i] = x[i];
			}
#endif
#if 0
		memcpy(y, x, n << 3);
#endif
#if 1
		// n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
		// the copy operation will be done for the multiples of 64
		for (i = 0; i < (n & (~0x3F)); i += 64)
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_pd(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_pd(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_pd(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_pd(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_pd(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_pd(x + num_elem_per_reg * 7);
			xv[8] = _mm256_loadu_pd(x + num_elem_per_reg * 8);
			xv[9] = _mm256_loadu_pd(x + num_elem_per_reg * 9);
			xv[10] = _mm256_loadu_pd(x + num_elem_per_reg * 10);
			xv[11] = _mm256_loadu_pd(x + num_elem_per_reg * 11);
			xv[12] = _mm256_loadu_pd(x + num_elem_per_reg * 12);
			xv[13] = _mm256_loadu_pd(x + num_elem_per_reg * 13);
			xv[14] = _mm256_loadu_pd(x + num_elem_per_reg * 14);
			xv[15] = _mm256_loadu_pd(x + num_elem_per_reg * 15);
			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_pd(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_pd(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_pd(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_pd(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_pd(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_pd(y + num_elem_per_reg * 7, xv[7]);
			_mm256_storeu_pd(y + num_elem_per_reg * 8, xv[8]);
			_mm256_storeu_pd(y + num_elem_per_reg * 9, xv[9]);
			_mm256_storeu_pd(y + num_elem_per_reg * 10, xv[10]);
			_mm256_storeu_pd(y + num_elem_per_reg * 11, xv[11]);
			_mm256_storeu_pd(y + num_elem_per_reg * 12, xv[12]);
			_mm256_storeu_pd(y + num_elem_per_reg * 13, xv[13]);
			_mm256_storeu_pd(y + num_elem_per_reg * 14, xv[14]);
			_mm256_storeu_pd(y + num_elem_per_reg * 15, xv[15]);
			y += num_elem_per_reg * 16;
			x += num_elem_per_reg * 16;
		}
		for (; i < (n & (~0x1F)); i += 32)
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_pd(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_pd(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_pd(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_pd(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_pd(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_pd(x + num_elem_per_reg * 7);

			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_pd(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_pd(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_pd(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_pd(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_pd(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_pd(y + num_elem_per_reg * 7, xv[7]);

			y += num_elem_per_reg * 8;
			x += num_elem_per_reg * 8;
		}
		for (; i < (n & (~0xF)); i += 16)
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_pd(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_pd(x + num_elem_per_reg * 3);

			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_pd(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_pd(y + num_elem_per_reg * 3, xv[3]);

			y += num_elem_per_reg * 4;
			x += num_elem_per_reg * 4;
		}
		for (; i < (n & (~0x07)); i += 8)
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);

			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);

			y += num_elem_per_reg * 2;
			x += num_elem_per_reg * 2;
		}
		for (; i < (n & (~0x03)); i += 4)
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			y += num_elem_per_reg;
			x += num_elem_per_reg;
		}
		for (; i < n; i++)
		{
			*y++ = *x++;
		}
#endif	
	}
	else
	{
		for ( i = 0; i < n; ++i)
		{
			*y = *x;

			x += incx;
			y += incy;
		}
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}

void bli_zcopyv_zen_int
(
	conj_t           conjx,
	dim_t            n,
	dcomplex*  restrict x, inc_t incx,
	dcomplex*  restrict y, inc_t incy,
	cntx_t* restrict cntx
)
{
	// If the vector dimension is zero return early.
	if (bli_zero_dim1(n))
	{
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
		return;
	}

	dim_t i = 0;
	dcomplex *x0 = x;
	dcomplex *y0 = y;

	if (bli_is_conj(conjx))
	{

		if (incx == 1 && incy == 1)
		{
			const dim_t n_elem_per_reg = 2;
			__m256d x_vec[8];

			__m256d conj_reg = _mm256_setr_pd(1, -1, 1, -1);

			for (; (i + 15) < n; i += 16)
			{
				/* 4 double values = 2 double complex values are loaded*/
				x_vec[0] = _mm256_loadu_pd((double *)x0);
				x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));
				x_vec[2] = _mm256_loadu_pd((double *)(x0 + 2 * n_elem_per_reg));
				x_vec[3] = _mm256_loadu_pd((double *)(x0 + 3 * n_elem_per_reg));
				x_vec[4] = _mm256_loadu_pd((double *)(x0 + 4 * n_elem_per_reg));
				x_vec[5] = _mm256_loadu_pd((double *)(x0 + 5 * n_elem_per_reg));
				x_vec[6] = _mm256_loadu_pd((double *)(x0 + 6 * n_elem_per_reg));
				x_vec[7] = _mm256_loadu_pd((double *)(x0 + 7 * n_elem_per_reg));

				/* Perform conjugation by multiplying the imaginary
				   part with -1 and real part with 1*/
				x_vec[0] = _mm256_mul_pd(x_vec[0], conj_reg);
				x_vec[1] = _mm256_mul_pd(x_vec[1], conj_reg);
				x_vec[2] = _mm256_mul_pd(x_vec[2], conj_reg);
				x_vec[3] = _mm256_mul_pd(x_vec[3], conj_reg);
				x_vec[4] = _mm256_mul_pd(x_vec[4], conj_reg);
				x_vec[5] = _mm256_mul_pd(x_vec[5], conj_reg);
				x_vec[6] = _mm256_mul_pd(x_vec[6], conj_reg);
				x_vec[7] = _mm256_mul_pd(x_vec[7], conj_reg);

				_mm256_storeu_pd((double *)y0, x_vec[0]);
				_mm256_storeu_pd((double *)(y0 + n_elem_per_reg), x_vec[1]);
				_mm256_storeu_pd((double *)(y0 + 2 * n_elem_per_reg), x_vec[2]);
				_mm256_storeu_pd((double *)(y0 + 3 * n_elem_per_reg), x_vec[3]);
				_mm256_storeu_pd((double *)(y0 + 4 * n_elem_per_reg), x_vec[4]);
				_mm256_storeu_pd((double *)(y0 + 5 * n_elem_per_reg), x_vec[5]);
				_mm256_storeu_pd((double *)(y0 + 6 * n_elem_per_reg), x_vec[6]);
				_mm256_storeu_pd((double *)(y0 + 7 * n_elem_per_reg), x_vec[7]);

				x0 += 8 * n_elem_per_reg;
				y0 += 8 * n_elem_per_reg;
			}

			for (; (i + 7) < n; i += 8)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);
				x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));
				x_vec[2] = _mm256_loadu_pd((double *)(x0 + 2 * n_elem_per_reg));
				x_vec[3] = _mm256_loadu_pd((double *)(x0 + 3 * n_elem_per_reg));

				x_vec[0] = _mm256_mul_pd(x_vec[0], conj_reg);
				x_vec[1] = _mm256_mul_pd(x_vec[1], conj_reg);
				x_vec[2] = _mm256_mul_pd(x_vec[2], conj_reg);
				x_vec[3] = _mm256_mul_pd(x_vec[3], conj_reg);

				x0 += 4 * n_elem_per_reg;

				_mm256_storeu_pd((double *)y0, x_vec[0]);
				_mm256_storeu_pd((double *)(y0 + n_elem_per_reg), x_vec[1]);
				_mm256_storeu_pd((double *)(y0 + 2 * n_elem_per_reg), x_vec[2]);
				_mm256_storeu_pd((double *)(y0 + 3 * n_elem_per_reg), x_vec[3]);

				y0 += 4 * n_elem_per_reg;
			}

			for (; (i + 3) < n; i += 4)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);
				x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));

				x0 += 2 * n_elem_per_reg;

				x_vec[0] = _mm256_mul_pd(x_vec[0], conj_reg);
				x_vec[1] = _mm256_mul_pd(x_vec[1], conj_reg);

				_mm256_storeu_pd((double *)y0, x_vec[0]);
				_mm256_storeu_pd((double *)(y0 + n_elem_per_reg), x_vec[1]);

				y0 += 2 * n_elem_per_reg;
			}

			for (; (i + 1) < n; i += 2)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);

				x_vec[0] = _mm256_mul_pd(x_vec[0], conj_reg);

				x0 += n_elem_per_reg;

				_mm256_storeu_pd((double *)y0, x_vec[0]);

				y0 += n_elem_per_reg;
			}

			// Issue vzeroupper instruction to clear upper lanes of ymm registers.
			// This avoids a performance penalty caused by false dependencies when
			// transitioning from AVX to SSE instructions (which may occur as soon
			// as the n_left cleanup loop below if BLIS is compiled with
			// -mfpmath=sse).
			_mm256_zeroupper();
		}
		else
		{
			/*Since double complex elements are of size 128 bits, vectorization
			can be done using XMM registers when incx and incy are not 1. This is done
			in the else condition.*/
			__m128d conj_reg = _mm_setr_pd(1, -1);
			__m128d x_vec[4];

			for (; (i + 3) < n; i += 4)
			{
				/* 2 double values = 1 double complex value(s) are(is) loaded*/
				x_vec[0] = _mm_loadu_pd((double *)x0);
				x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));
				x_vec[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
				x_vec[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

				x_vec[0] = _mm_mul_pd(x_vec[0], conj_reg);
				x_vec[1] = _mm_mul_pd(x_vec[1], conj_reg);
				x_vec[2] = _mm_mul_pd(x_vec[2], conj_reg);
				x_vec[3] = _mm_mul_pd(x_vec[3], conj_reg);

				_mm_storeu_pd((double *)y0, x_vec[0]);
				_mm_storeu_pd((double *)(y0 + incy), x_vec[1]);
				_mm_storeu_pd((double *)(y0 + 2 * incy), x_vec[2]);
				_mm_storeu_pd((double *)(y0 + 3 * incy), x_vec[3]);

				x0 += 4 * incx;
				y0 += 4 * incy;
			}

			for (; (i + 1) < n; i += 2)
			{
				x_vec[0] = _mm_loadu_pd((double *)x0);
				x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));

				x_vec[0] = _mm_mul_pd(x_vec[0], conj_reg);
				x_vec[1] = _mm_mul_pd(x_vec[1], conj_reg);

				_mm_storeu_pd((double *)y0, x_vec[0]);
				_mm_storeu_pd((double *)(y0 + incy), x_vec[1]);

				x0 += 2 * incx;
				y0 += 2 * incy;
			}
		}

		__m128d conj_reg = _mm_setr_pd(1, -1);
		__m128d x_vec[1];

		for (; i < n; i += 1)
		{
			x_vec[0] = _mm_loadu_pd((double *)x0);

			x_vec[0] = _mm_mul_pd(x_vec[0], conj_reg);

			_mm_storeu_pd((double *)y0, x_vec[0]);

			x0 += incx;
			y0 += incy;
		}
	}
	else
	{

		if (incx == 1 && incy == 1)
		{
			const dim_t n_elem_per_reg = 2;
			__m256d x_vec[8];

			for (; (i + 15) < n; i += 16)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);
				x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));
				x_vec[2] = _mm256_loadu_pd((double *)(x0 + 2 * n_elem_per_reg));
				x_vec[3] = _mm256_loadu_pd((double *)(x0 + 3 * n_elem_per_reg));
				x_vec[4] = _mm256_loadu_pd((double *)(x0 + 4 * n_elem_per_reg));
				x_vec[5] = _mm256_loadu_pd((double *)(x0 + 5 * n_elem_per_reg));
				x_vec[6] = _mm256_loadu_pd((double *)(x0 + 6 * n_elem_per_reg));
				x_vec[7] = _mm256_loadu_pd((double *)(x0 + 7 * n_elem_per_reg));

				x0 += 8 * n_elem_per_reg;

				_mm256_storeu_pd((double *)y0, x_vec[0]);
				_mm256_storeu_pd((double *)(y0 + n_elem_per_reg), x_vec[1]);
				_mm256_storeu_pd((double *)(y0 + 2 * n_elem_per_reg), x_vec[2]);
				_mm256_storeu_pd((double *)(y0 + 3 * n_elem_per_reg), x_vec[3]);
				_mm256_storeu_pd((double *)(y0 + 4 * n_elem_per_reg), x_vec[4]);
				_mm256_storeu_pd((double *)(y0 + 5 * n_elem_per_reg), x_vec[5]);
				_mm256_storeu_pd((double *)(y0 + 6 * n_elem_per_reg), x_vec[6]);
				_mm256_storeu_pd((double *)(y0 + 7 * n_elem_per_reg), x_vec[7]);

				y0 += 8 * n_elem_per_reg;
			}

			for (; (i + 7) < n; i += 8)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);
				x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));
				x_vec[2] = _mm256_loadu_pd((double *)(x0 + 2 * n_elem_per_reg));
				x_vec[3] = _mm256_loadu_pd((double *)(x0 + 3 * n_elem_per_reg));

				x0 += 4 * n_elem_per_reg;

				_mm256_storeu_pd((double *)y0, x_vec[0]);
				_mm256_storeu_pd((double *)(y0 + n_elem_per_reg), x_vec[1]);
				_mm256_storeu_pd((double *)(y0 + 2 * n_elem_per_reg), x_vec[2]);
				_mm256_storeu_pd((double *)(y0 + 3 * n_elem_per_reg), x_vec[3]);

				y0 += 4 * n_elem_per_reg;
			}

			for (; (i + 3) < n; i += 4)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);
				x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));

				x0 += 2 * n_elem_per_reg;

				_mm256_storeu_pd((double *)y0, x_vec[0]);
				_mm256_storeu_pd((double *)(y0 + n_elem_per_reg), x_vec[1]);

				y0 += 2 * n_elem_per_reg;
			}

			for (; (i + 1) < n; i += 2)
			{
				x_vec[0] = _mm256_loadu_pd((double *)x0);

				x0 += n_elem_per_reg;

				_mm256_storeu_pd((double *)y0, x_vec[0]);

				y0 += n_elem_per_reg;
			}

			// Issue vzeroupper instruction to clear upper lanes of ymm registers.
			// This avoids a performance penalty caused by false dependencies when
			// transitioning from AVX to SSE instructions (which may occur as soon
			// as the n_left cleanup loop below if BLIS is compiled with
			// -mfpmath=sse).
			_mm256_zeroupper();
		}
		else
		{
			/*Since double complex elements are of size 128 bits, vectorization
			can be done using XMM registers when incx and incy are not 1. This is done
			in the else condition.*/
			__m128d x_vec[4];

			for (; (i + 3) < n; i += 4)
			{
				x_vec[0] = _mm_loadu_pd((double *)x0);
				x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));
				x_vec[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
				x_vec[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

				x0 += 4 * incx;

				_mm_storeu_pd((double *)y0, x_vec[0]);
				_mm_storeu_pd((double *)(y0 + incy), x_vec[1]);
				_mm_storeu_pd((double *)(y0 + 2 * incy), x_vec[2]);
				_mm_storeu_pd((double *)(y0 + 3 * incy), x_vec[3]);

				y0 += 4 * incy;
			}

			for (; (i + 1) < n; i += 2)
			{
				x_vec[0] = _mm_loadu_pd((double *)x0);
				x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));

				x0 += 2 * incx;

				_mm_storeu_pd((double *)y0, x_vec[0]);
				_mm_storeu_pd((double *)(y0 + incy), x_vec[1]);

				y0 += 2 * incy;
			}
		}
		__m128d x_vec[1];

		for (; i < n; i += 1)
		{
			x_vec[0] = _mm_loadu_pd((double *)x0);

			x0 += incx;

			_mm_storeu_pd((double *)y0, x_vec[0]);

			y0 += incy;
		}
	}
}
