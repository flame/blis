/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2022, Advanced Micro Devices, Inc.

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
#include "blis.h"
#include "immintrin.h"


/**
 * daxpy2v kernel performs axpy2v operation.
 *  z := y + alphax * conjx(x) + alphay * conjy(y)
 * where x, y, and z are vectors of length n.
 */
void bli_daxpy2v_zen_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       double*  restrict alphax,
       double*  restrict alphay,
       double*  restrict x, inc_t incx,
       double*  restrict y, inc_t incy,
       double*  restrict z, inc_t incz,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	if ( incz == 1 && incx == 1 && incy == 1 )
	{
		dim_t i = 0;
		dim_t rem = n%4;
		const dim_t n_elem_per_reg = 4;
		__m256d xv[4], yv[4], zv[4];
		__m256d alphaxv, alphayv;

		alphaxv = _mm256_broadcast_sd((double const*) alphax);
		alphayv = _mm256_broadcast_sd((double const*) alphay);

		for( ; (i + 15) < n; i+= 16 )
		{
			xv[0] = _mm256_loadu_pd( x + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( y + 3*n_elem_per_reg );

			zv[0] = _mm256_loadu_pd( z + 0*n_elem_per_reg );
			zv[1] = _mm256_loadu_pd( z + 1*n_elem_per_reg );
			zv[2] = _mm256_loadu_pd( z + 2*n_elem_per_reg );
			zv[3] = _mm256_loadu_pd( z + 3*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd(xv[0], alphaxv, zv[0]);
			zv[1] = _mm256_fmadd_pd(xv[1], alphaxv, zv[1]);
			zv[2] = _mm256_fmadd_pd(xv[2], alphaxv, zv[2]);
			zv[3] = _mm256_fmadd_pd(xv[3], alphaxv, zv[3]);

			zv[0] = _mm256_fmadd_pd(yv[0], alphayv, zv[0]);
			zv[1] = _mm256_fmadd_pd(yv[1], alphayv, zv[1]);
			zv[2] = _mm256_fmadd_pd(yv[2], alphayv, zv[2]);
			zv[3] = _mm256_fmadd_pd(yv[3], alphayv, zv[3]);

			_mm256_storeu_pd((z + 0*n_elem_per_reg), zv[0]);
			_mm256_storeu_pd((z + 1*n_elem_per_reg), zv[1]);
			_mm256_storeu_pd((z + 2*n_elem_per_reg), zv[2]);
			_mm256_storeu_pd((z + 3*n_elem_per_reg), zv[3]);

			z += 4*n_elem_per_reg;
			x += 4*n_elem_per_reg;
			y += 4*n_elem_per_reg;
		}

		for( ; (i + 7) < n; i+= 8 )
		{
			xv[0] = _mm256_loadu_pd( x + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y + 1*n_elem_per_reg );

			zv[0] = _mm256_loadu_pd( z + 0*n_elem_per_reg );
			zv[1] = _mm256_loadu_pd( z + 1*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd(xv[0], alphaxv, zv[0]);
			zv[1] = _mm256_fmadd_pd(xv[1], alphaxv, zv[1]);

			zv[0] = _mm256_fmadd_pd(yv[0], alphayv, zv[0]);
			zv[1] = _mm256_fmadd_pd(yv[1], alphayv, zv[1]);

			_mm256_storeu_pd((z + 0*n_elem_per_reg), zv[0]);
			_mm256_storeu_pd((z + 1*n_elem_per_reg), zv[1]);

			z += 2*n_elem_per_reg;
			x += 2*n_elem_per_reg;
			y += 2*n_elem_per_reg;
		}

		for( ; (i + 3) < n; i+= 4 )
		{
			xv[0] = _mm256_loadu_pd( x + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y + 0*n_elem_per_reg );

			zv[0] = _mm256_loadu_pd( z + 0*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd(xv[0], alphaxv, zv[0]);

			zv[0] = _mm256_fmadd_pd(yv[0], alphayv, zv[0]);

			_mm256_storeu_pd((z + 0*n_elem_per_reg), zv[0]);

			z += n_elem_per_reg;
			x += n_elem_per_reg;
			y += n_elem_per_reg;
		}
		if(rem)
		{
			PRAGMA_SIMD
				for ( i = 0; i < rem; ++i )
				{
					PASTEMAC(d,axpys)( *alphax, x[i], z[i] );
					PASTEMAC(d,axpys)( *alphay, y[i], z[i] );
				}
		}
	}
	else
	{
		/* Query the context for the kernel function pointer. */
		const num_t              dt     = PASTEMAC(d,type);
		PASTECH(d,axpyv_ker_ft) kfp_av
		=
		bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx );

		kfp_av
		(
		  conjx,
		  n,
		  alphax,
		  x, incx,
		  z, incz,
		  cntx
		);

		kfp_av
		(
		  conjy,
		  n,
		  alphay,
		  y, incy,
		  z, incz,
		  cntx
		);
	}
}
