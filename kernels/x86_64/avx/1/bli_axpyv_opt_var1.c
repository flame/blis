/*

   BLIS   
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <immintrin.h> 

typedef union
{
	__m256d v;
	double  d[4];
} v4df_t;

void bli_daxpyv_opt_var1(
			 conj_t           conjx,
			 dim_t            n,
			 double* restrict alpha,
			 double* restrict x, inc_t incx,
			 double* restrict y, inc_t incy
		       )
{
	double*  restrict x_cast   = x;
	double*  restrict y_cast   = y;

	double*  restrict x1;
	double*  restrict y1;
	gint_t            i;
	double            alphad;
	v4df_t            alphav;
	v4df_t            x0v, x1v, x2v, x3v;
	v4df_t            y0v, y1v, y2v, y3v;
	gint_t            n_run, n_pre, n_left;

	const dim_t       n_elem_per_reg = 4;
	const dim_t       n_iter_unroll  = 4;

	bool_t            use_ref = FALSE;

	// If the vector lengths are zero, return.
	if ( bli_zero_dim1( n ) )
	{
		return;
	}

        alphad = *alpha;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( incx != 1 || incy != 1 )
	{
		use_ref = TRUE;
	}

	n_pre = 0;
	if ( bli_is_unaligned_to( x, 32 ) ||
	     bli_is_unaligned_to( y, 32 ) )
	{
		guint_t x_offset = bli_offset_from_alignment( x, 32 );
		guint_t y_offset = bli_offset_from_alignment( y, 32 );

		if ( x_offset % 8 != 0 ||
		     x_offset != y_offset )
		{
			use_ref = TRUE;
		}
		else
		{
			n_pre = ( 32 - x_offset ) / 8;
		}
	}


	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		BLIS_DAXPYV_KERNEL_REF( conjx,
					n,
					alpha,
					x, incx,
					y, incy );
		return;
	}

	x1 = x_cast;
	y1 = y_cast;

	n_run       = ( n - n_pre ) / ( n_elem_per_reg * n_iter_unroll );
	n_left      = ( n - n_pre ) % ( n_elem_per_reg * n_iter_unroll );

	while ( n_pre-- > 0 )
	{
		*(y1++) += alphad * *(x1++);
	}

	alphav.v = _mm256_set_pd(alphad, alphad, alphad, alphad);
	x0v.v = _mm256_setzero_pd();
	x1v.v = _mm256_setzero_pd();
	x2v.v = _mm256_setzero_pd();
	x3v.v = _mm256_setzero_pd();
	y0v.v = _mm256_setzero_pd();
	y1v.v = _mm256_setzero_pd();
	y2v.v = _mm256_setzero_pd();
	y3v.v = _mm256_setzero_pd();

	for ( i = 0; i < n_run; i++ )
	{
		x0v.v = _mm256_load_pd( ( double* )(x1 + 0*n_elem_per_reg) );
		x1v.v = _mm256_load_pd( ( double* )(x1 + 1*n_elem_per_reg) );
		x2v.v = _mm256_load_pd( ( double* )(x1 + 2*n_elem_per_reg) );
		x3v.v = _mm256_load_pd( ( double* )(x1 + 3*n_elem_per_reg) );

		y0v.v = _mm256_load_pd( ( double* )(y1 + 0*n_elem_per_reg) );
		y1v.v = _mm256_load_pd( ( double* )(y1 + 1*n_elem_per_reg) );
		y2v.v = _mm256_load_pd( ( double* )(y1 + 2*n_elem_per_reg) );
		y3v.v = _mm256_load_pd( ( double* )(y1 + 3*n_elem_per_reg) );

		y0v.v += x0v.v * alphav.v;
		y1v.v += x1v.v * alphav.v;
		y2v.v += x2v.v * alphav.v;
		y3v.v += x3v.v * alphav.v;

		_mm256_store_pd( ( double* )(y1 + 0*n_elem_per_reg), y0v.v );
		_mm256_store_pd( ( double* )(y1 + 1*n_elem_per_reg), y1v.v );
		_mm256_store_pd( ( double* )(y1 + 2*n_elem_per_reg), y2v.v );
		_mm256_store_pd( ( double* )(y1 + 3*n_elem_per_reg), y3v.v );

		x1 += n_iter_unroll * n_elem_per_reg;
		y1 += n_iter_unroll * n_elem_per_reg;
	}

	while ( n_left-- > 0 )
	{
		*(y1++) += alphad * *(x1++);
	}
}
