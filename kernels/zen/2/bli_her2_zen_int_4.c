/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_dher2_trans_zen_int_4
     (
       double *a,
       double *x,
       double *y,
       double *alpha,
       dim_t m,
       dim_t lda
     )
{
	dim_t row = 0;
	dim_t rem = m  % 4;

	/*holds 4 diagonal elements of triangular part of 4x4 tile*/
	double a_diag[4] = {0};
	/*alpha_chi holds x*alpha and alpha_psi holds y*alpha*/
	double alpha_chi[4] = {0};
	double alpha_psi[4] = {0};
	/*Extracts diagonal element and store into a_diag buffer*/
	PRAGMA_SIMD
		for(dim_t i = 0; i < 4; i++)
		{
			a_diag[i] = *(a + m + i + (i * lda));
		}

	__m256d x0, x1, x2, x3;
	__m256d y0, y1, y2, y3;

	__m256d xr, yr, zero, gamma;
	__m256d a0, a1, a2, a3;

	zero = _mm256_setzero_pd();

	/*Loading elements of x & y vectors*/
	x0 = _mm256_loadu_pd(x + m);
	y0 = _mm256_loadu_pd(y + m);
	/*Broadcasting alpha to compute alpha_psi and alpha_chi*/
	x1 = _mm256_broadcast_sd(alpha);

	x2 = _mm256_mul_pd(x0, x1);
	y0 = _mm256_mul_pd(y0, x1);

	/*Storing alpha_chi and alpha_psi for later usage in computation loop*/
	_mm256_storeu_pd(alpha_chi, x2);
	_mm256_storeu_pd(alpha_psi, y0);

	x0 = _mm256_mul_pd(x0, y0);
	gamma = _mm256_loadu_pd(a_diag);
	gamma = _mm256_add_pd(gamma, x0);
	gamma = _mm256_add_pd(gamma, x0);
	_mm256_storeu_pd(a_diag, gamma);

	/* Broadcasting 4 alpha_psis and alpha_chis which
	 * are to be used througout the computation of 4x4 tile
	 * upto m rows.
	 */
	x0 = _mm256_broadcast_sd(&alpha_chi[0]);
	x1 = _mm256_broadcast_sd(&alpha_chi[1]);
	x2 = _mm256_broadcast_sd(&alpha_chi[2]);
	x3 = _mm256_broadcast_sd(&alpha_chi[3]);

	y0 = _mm256_broadcast_sd(&alpha_psi[0]);
	y1 = _mm256_broadcast_sd(&alpha_psi[1]);
	y2 = _mm256_broadcast_sd(&alpha_psi[2]);
	y3 = _mm256_broadcast_sd(&alpha_psi[3]);

	/* Loading 4x4 tile of A matrix for
	 * triangular part computation
	 */
	a0 = _mm256_loadu_pd(a +  (0 * lda) + m);
	a1 = _mm256_loadu_pd(a +  (1 * lda) + m);
	a2 = _mm256_loadu_pd(a +  (2 * lda) + m);
	a3 = _mm256_loadu_pd(a +  (3 * lda) + m);

	yr = _mm256_loadu_pd(y);
	xr = _mm256_loadu_pd(x);

	/*Setting first element of x & y vectors to zero
	 * to eliminate diagonal element of 1st column
	 * from computation
	 */
	xr = _mm256_blend_pd(xr, zero, 0x1);
	yr = _mm256_blend_pd(yr, zero, 0x1);
	a0 = _mm256_blend_pd(a0, zero, 0x1);

	a1 = _mm256_blend_pd(a1, zero, 0x3);
	a2 = _mm256_blend_pd(a2, zero, 0x7);
	a3 = _mm256_blend_pd(a3, zero, 0xF);

	a0 = _mm256_fmadd_pd(xr, y0, a0);
	a0 = _mm256_fmadd_pd(yr, x0, a0);

	/*Setting two elements of x & y vectors to zero
	 * to eliminate diagonal element of 2nd column
	 * from computation
	 */
	xr = _mm256_blend_pd(xr, zero, 0x3);
	yr = _mm256_blend_pd(yr, zero, 0x3);
	a1 = _mm256_fmadd_pd(xr, y1, a1);
	a1 = _mm256_fmadd_pd(yr, x1, a1);

	/*Setting three elements of x & y vectors to zero
	 * to eliminate diagonal element of 3rd column
	 * from computation
	 */
	xr = _mm256_blend_pd(xr, zero, 0x7);
	yr = _mm256_blend_pd(yr, zero, 0x7);
	a2 = _mm256_fmadd_pd(xr, y2, a2);
	a2 = _mm256_fmadd_pd(yr, x2, a2);

	_mm256_storeu_pd(a +  (0 * lda) + m, a0 );

	/* Loading data from memory location first
	 * so it could be blend with and finally
	 * gets stored at same location to prevent
	 * unnecessary data overwriting at nearby
	 * memory locations
	 */
	a3 = _mm256_loadu_pd(a +  (1 * lda) + m );
	a1 = _mm256_blend_pd(a1, a3, 0x1);
	_mm256_storeu_pd(a +  (1 * lda) + m, a1 );

	a3 = _mm256_loadu_pd(a +  (2 * lda) + m );
	a2 = _mm256_blend_pd(a2, a3, 0x3);
	_mm256_storeu_pd(a +  (2 * lda) + m, a2 );

	/* Triangular part of matrix is computed, remaining
	 * part is computed in below loop upto m rows.
	 */
	for(; (row + 4) <=  m; row+=4)
	{
		/* Loading elements of x and y vector */
		xr = _mm256_loadu_pd(x + row);
		yr = _mm256_loadu_pd(y + row);
		/* Loading tile of A matrix of size 4x4 */
		a0 = _mm256_loadu_pd(a + row  + (0 * lda) );
		a1 = _mm256_loadu_pd(a + row  + (1 * lda) );
		a2 = _mm256_loadu_pd(a + row  + (2 * lda) );
		a3 = _mm256_loadu_pd(a + row  + (3 * lda) );

		a0 = _mm256_fmadd_pd(xr, y0, a0);
		a1 = _mm256_fmadd_pd(xr, y1, a1);
		a2 = _mm256_fmadd_pd(xr, y2, a2);
		a3 = _mm256_fmadd_pd(xr, y3, a3);

		a0 = _mm256_fmadd_pd(yr, x0, a0);
		a1 = _mm256_fmadd_pd(yr, x1, a1);
		a2 = _mm256_fmadd_pd(yr, x2, a2);
		a3 = _mm256_fmadd_pd(yr, x3, a3);

		_mm256_storeu_pd(a + row + (0 * lda), a0);
		_mm256_storeu_pd(a + row + (1 * lda), a1);
		_mm256_storeu_pd(a + row + (2 * lda), a2);
		_mm256_storeu_pd(a + row + (3 * lda), a3);
	}

	/* Computes remainder cases where m is less than 4 */
	if(rem)
	{
		PRAGMA_SIMD
			for(dim_t i = 0; i < 4; i++)
			{
				for(dim_t j = row; j < m; j++)
				{
					a[ j + (i * lda)] += x[j] * (y[i] * (*alpha));
					a[ j + (i * lda)] += y[j] * (x[i] * (*alpha));
				}
			}
	}

	/* Computing 4 diagonal elements of triangular part of matrix
	 * and storing result back at corresponding location in matrix A
	 */
	PRAGMA_SIMD
		for(dim_t i = 0; i < 4; i++)
		{
			*(a + m + i + (i * lda)) = a_diag[i];
		}
}


void bli_dher2_zen_int_4
     (
       double *a,
       double *x,
       double *y,
       double *alpha,
       dim_t m,
       dim_t lda
     )
{
	dim_t row = 4;
	dim_t rem = m  % 4;

	 /*holds 4 diagonal elements of triangular part of 4x4 tile*/
	double a_diag[4] = {0};
	 /*alpha_chi holds x*alpha and alpha_psi holds y*alpha*/
	double alpha_chi[4] = {0};
	double alpha_psi[4] = {0};
	/*Extracts diagonal element and store into a_diag buffer*/
	PRAGMA_SIMD
		for(dim_t i = 0; i < 4; i++)
		{
			a_diag[i] = *(a + i + (i * lda));
		}

	__m256d x0, x1, x2, x3;
	__m256d y0, y1, y2, y3;

	__m256d xr, yr, zero, gamma;
	__m256d a0, a1, a2, a3;

	zero = _mm256_setzero_pd();

	/*Loading elements of x & y vectors*/
	x0 = _mm256_loadu_pd(x);
	y0 = _mm256_loadu_pd(y);
	/*Broadcasting alpha to compute alpha_psi and alpha_chi*/
	x1 = _mm256_broadcast_sd(alpha);

	x2 = _mm256_mul_pd(x0, x1);
	y0 = _mm256_mul_pd(y0, x1);

	/*Storing alpha_chi and alpha_psi for later usage in computation loop*/
	_mm256_storeu_pd(alpha_chi, x2);
	_mm256_storeu_pd(alpha_psi, y0);

	x0 = _mm256_mul_pd(x0, y0);
	gamma = _mm256_loadu_pd(a_diag);
	gamma = _mm256_add_pd(gamma, x0);
	gamma = _mm256_add_pd(gamma, x0);
	_mm256_storeu_pd(a_diag, gamma);

	/* Broadcasting 4 alpha_psis and alpha_chis which
         * are to be used througout the computation of 4x4 tile
         * upto m rows.
         */
	x0 = _mm256_broadcast_sd(&alpha_chi[0]);
	x1 = _mm256_broadcast_sd(&alpha_chi[1]);
	x2 = _mm256_broadcast_sd(&alpha_chi[2]);
	x3 = _mm256_broadcast_sd(&alpha_chi[3]);

	y0 = _mm256_broadcast_sd(&alpha_psi[0]);
	y1 = _mm256_broadcast_sd(&alpha_psi[1]);
	y2 = _mm256_broadcast_sd(&alpha_psi[2]);
	y3 = _mm256_broadcast_sd(&alpha_psi[3]);

	/* Loading 4x4 tile of A matrix for
	 * triangular part computation
	 */
	a0 = _mm256_loadu_pd(a +  (0 * lda) );
	a1 = _mm256_loadu_pd(a +  (1 * lda) );
	a2 = _mm256_loadu_pd(a +  (2 * lda) );
	a3 = _mm256_loadu_pd(a +  (3 * lda) );

	yr = _mm256_loadu_pd(y);
	xr = _mm256_loadu_pd(x);

	/*Setting first element of x & y vectors to zero
	 * to eliminate diagonal element of 1st column
	 * from computation
	 */
	xr = _mm256_blend_pd(xr, zero, 0x1);
	yr = _mm256_blend_pd(yr, zero, 0x1);
	a0 = _mm256_blend_pd(a0, zero, 0x1);
	a1 = _mm256_blend_pd(a1, zero, 0x3);
	a2 = _mm256_blend_pd(a2, zero, 0x7);
	a3 = _mm256_blend_pd(a3, zero, 0xF);

	a0 = _mm256_fmadd_pd(xr, y0, a0);
	a0 = _mm256_fmadd_pd(yr, x0, a0);

	/*Setting two elements of x & y vectors to zero
         * to eliminate diagonal element of 2nd column
         * from computation
         */
	xr = _mm256_blend_pd(xr, zero, 0x3);
	yr = _mm256_blend_pd(yr, zero, 0x3);
	a1 = _mm256_fmadd_pd(xr, y1, a1);
	a1 = _mm256_fmadd_pd(yr, x1, a1);

	/*Setting three elements of x & y vectors to zero
	 * to eliminate diagonal element of 3rd column
	 * from computation
	 */
	xr = _mm256_blend_pd(xr, zero, 0x7);
	yr = _mm256_blend_pd(yr, zero, 0x7);
	a2 = _mm256_fmadd_pd(xr, y2, a2);
	a2 = _mm256_fmadd_pd(yr, x2, a2);

	_mm256_storeu_pd(a +  (0 * lda), a0 );

	/* Loading data from memory location first
	 * so it could be blend with and finally
	 * gets stored at same location to prevent
	 * unnecessary data overwriting at nearby
	 * memory locations
	 */
	a3 = _mm256_loadu_pd(a +  (1 * lda) );
	a1 = _mm256_blend_pd(a1, a3, 0x1);
	_mm256_storeu_pd(a +  (1 * lda), a1 );

	a3 = _mm256_loadu_pd(a +  (2 * lda) );
	a2 = _mm256_blend_pd(a2, a3, 0x3);
	_mm256_storeu_pd(a +  (2 * lda), a2 );

	/* Triangular part of matrix is computed, remaining
	 * part is computed in below loop upto m rows.
	 */
	for(; (row + 4) <=  m; row+=4)
	{
		/* Loading elements of x and y vector */
		xr = _mm256_loadu_pd(x + row);
		yr = _mm256_loadu_pd(y + row);
		/* Loading tile of A matrix of size 4x4 */
		a0 = _mm256_loadu_pd(a + row  + (0 * lda) );
		a1 = _mm256_loadu_pd(a + row  + (1 * lda) );
		a2 = _mm256_loadu_pd(a + row  + (2 * lda) );
		a3 = _mm256_loadu_pd(a + row  + (3 * lda) );

		a0 = _mm256_fmadd_pd(xr, y0, a0);
		a1 = _mm256_fmadd_pd(xr, y1, a1);
		a2 = _mm256_fmadd_pd(xr, y2, a2);
		a3 = _mm256_fmadd_pd(xr, y3, a3);

		a0 = _mm256_fmadd_pd(yr, x0, a0);
		a1 = _mm256_fmadd_pd(yr, x1, a1);
		a2 = _mm256_fmadd_pd(yr, x2, a2);
		a3 = _mm256_fmadd_pd(yr, x3, a3);

		_mm256_storeu_pd(a + row + (0 * lda), a0);
		_mm256_storeu_pd(a + row + (1 * lda), a1);
		_mm256_storeu_pd(a + row + (2 * lda), a2);
		_mm256_storeu_pd(a + row + (3 * lda), a3);
	}

	/* Computes remainder cases where m is less than 4 */
	if(rem)
	{
		PRAGMA_SIMD
			for(dim_t i = 0; i < 4; i++)
			{
				for(dim_t j = row; j < m; j++)
				{
					a[ j + (i * lda)] += x[j] * (y[i] * (*alpha));
					a[ j + (i * lda)] += y[j] * (x[i] * (*alpha));
				}
			}
	}

	/* Computing 4 diagonal elements of triangular part of matrix
	 * and storing result back at corresponding location in matrix A
	 */
	PRAGMA_SIMD
		for(dim_t i = 0; i < 4; i++)
		{
			*(a + i + (i * lda)) = a_diag[i];
		}
}
