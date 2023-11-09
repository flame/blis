/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * zaxpy2v kernel performs axpy2v operation.
 * z := z + alphax * conjx(x) + alphay * conjy(y)
 * where,
 *      x, y & z are double complex vectors of length n.
 *      alpha & beta are complex scalers.
 */
void bli_zaxpy2v_zen_int
     (
       conj_t             conjx,
       conj_t             conjy,
       dim_t              n,
       dcomplex* restrict alphax,
       dcomplex* restrict alphay,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict z, inc_t incz,
       cntx_t*   restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

    // If the vectors are empty or if both alpha are zero, return early
    if ( ( bli_zero_dim1( n ) ) ||
         ( PASTEMAC(z,eq0)( *alphax ) && PASTEMAC(z,eq0)( *alphay ) ) ) {
             AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
             return;
         }

        const dim_t      n_elem_per_reg = 4; // Number of elements per register

        dim_t i = 0;        // Iterator

        double*  restrict x0;
        double*  restrict y0;
        double*  restrict z0;
        double*  restrict alphax0;
        double*  restrict alphay0;

        // Initialize local pointers.
        x0 		= (double*) x;
        y0 		= (double*) y;
        z0 		= (double*) z;
        alphax0 = (double*) alphax;
        alphay0 = (double*) alphay;

    if ( incx == 1 && incy == 1 && incz == 1 )
    {
        //---------- Scalar algorithm BLIS_NO_CONJUGATE -------------
        //
        // z = z + alphax * x + alphay * y
        // z =  ( zR + izI ) +
        //      ( axR + iaxI ) * ( xR + ixI ) +
        //      ( ayR + iayI ) * ( yR + iyI )
        // z =  ( zR + izI ) +
        //      ( axR.xR + iaxR.xI + iaxI.xR - axI.xI ) +
        //      ( xyR.yR + iayR.yI + iayI.yR - ayI.yI )
        // z =  ( zR + izI ) +
        //      ( ( axR.xR - axI.xI ) + i( axR.xI + axI.xR ) ) +
        //      ( ( ayR.yR - ayI.yI ) + i( ayR.yI + ayI.yR ) )
        // z =  ( zR + axR.xR - axI.xI + ayR.yR - ayI.yI ) +
        //     i( zI + axR.xI + axI.xR + ayR.yI + ayI.yR )
        //
        // SIMD Algorithm BLIS_NO_CONJUGATE
        // xv   =  xR0   xI0   xR1   xI1
        // xv'  =  xI0   xR0   xI1   xR1
        // yv   =  yR0   yI0   yR1   yI1
        // yv'  =  yI0   yR0   yI1   yR1
        // zv   =  zR0   zI0   zR1   zI1
        // zv'  =  zI0   zR0   zI1   zR1
        // axrv =  axR   axR   axR   axR
        // axiv = -axI   axI  -axI   axI
        // ayrv =  ayR   ayR   ayR   ayR
        // ayiv = -ayI   ayI  -ayI   ayI
        //
        // step 1: FMA zv = zv + axrv * xv
        // step 2: shuffle xv -> xv'
        // step 3: FMA zv = zv + axiv * xv'
        // step 4: FMA zv = zv + ayrv * yv
        // step 5: shuffle yv -> xyv'
        // step 6: FMA zv = zv + ayiv * yv'

        //---------- Scalar algorithm BLIS_CONJUGATE -------------
        //
        // z = z + alphax * x + alphay * y
        // z =  ( zR + izI ) +
        //      ( axR + iaxI ) * ( xR - ixI ) +
        //      ( ayR + iayI ) * ( yR - iyI )
        // z =  ( zR + izI ) +
        //      ( axR.xR - iaxR.xI + iaxI.xR + axI.xI ) +
        //      ( xyR.yR - iayR.yI + iayI.yR + ayI.yI )
        // z =  ( zR + izI ) +
        //      ( ( axR.xR + axI.xI ) + i( -axR.xI + axI.xR ) ) +
        //      ( ( ayR.yR + ayI.yI ) + i( -ayR.yI + ayI.yR ) )
        // z =  ( zR + axR.xR + axI.xI + ayR.yR + ayI.yI ) +
        //     i( zI - axR.xI + axI.xR - ayR.yI + ayI.yR )
        //
        // SIMD Algorithm BLIS_CONJUGATE
        // xv   =  xR0   xI0   xR1   xI1
        // xv'  =  xI0   xR0   xI1   xR1
        // yv   =  yR0   yI0   yR1   yI1
        // yv'  =  yI0   yR0   yI1   yR1
        // zv   =  zR0   zI0   zR1   zI1
        // zv'  =  zI0   zR0   zI1   zR1
        // axrv =  axR  -axR   axR  -axR
        // axiv =  axI   axI   axI   axI
        // ayrv =  ayR  -ayR   ayR  -ayR
        // ayiv =  ayI   ayI   ayI   ayI
        //
        // step 1: FMA zv = zv + axrv * xv
        // step 2: shuffle xv -> xv'
        // step 3: FMA zv = zv + axiv * xv'
        // step 4: FMA zv = zv + ayrv * yv
        // step 5: shuffle yv -> xyv'
        // step 6: FMA zv = zv + ayiv * yv'

        __m256d alphaxRv;
        __m256d alphaxIv;
        __m256d alphayRv;
        __m256d alphayIv;
        __m256d xv[4];
        __m256d yv[4];
        __m256d zv[4];

        double alphaxR, alphaxI;
        double alphayR, alphayI;

        alphaxR = alphax->real;
        alphaxI = alphax->imag;
        alphayR = alphay->real;
        alphayI = alphay->imag;

        // Broadcast alphax & alphay to respective vector registers
        if ( !bli_is_conj( conjx ) ) // If not x conjugate
        {
            // alphaxRv =  axR  axR  axR  axR
            // alphaxIv = -axI  axI -axI  axI
            alphaxRv = _mm256_broadcast_sd( &alphaxR );
            alphaxIv = _mm256_set_pd( alphaxI, -alphaxI, alphaxI, -alphaxI );
        }
        else
        {
            // alphaxRv =  axR -axR  axR -axR
            // alphaxIv =  axI  axI  axI  axI
            alphaxRv = _mm256_set_pd( -alphaxR, alphaxR, -alphaxR, alphaxR );
            alphaxIv = _mm256_broadcast_sd( &alphaxI );
        }

        if ( !bli_is_conj( conjy ) ) // If not y conjugate
        {
            // alphayRv =  ayR  ayR  ayR  ayR
            // alphayIv = -ayI  ayI -ayI  ayI
            alphayRv = _mm256_broadcast_sd( &alphayR );
            alphayIv = _mm256_set_pd( alphayI, -alphayI, alphayI, -alphayI );
        }
        else
        {
            // alphayRv =  ayR -ayR  ayR -ayR
            // alphayIv =  ayI  ayI  ayI  ayI
            alphayRv = _mm256_set_pd( -alphayR, alphayR, -alphayR, alphayR );
            alphayIv = _mm256_broadcast_sd( &alphayI );
        }

        // Processing 8 elements per loop, 16 FMAs
        for ( ; ( i + 7 ) < n; i += 8 )
        {
            // Loading x vector
            // xv = xR0 xI0 xR1 xI1
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

            // Loading y vector
            // yv = yR0 yI0 yR1 yI1
            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

            // Loading z vector
            // zv = zR0 zI0 zR1 zI1
            zv[0] = _mm256_loadu_pd( z0 + 0*n_elem_per_reg );
            zv[1] = _mm256_loadu_pd( z0 + 1*n_elem_per_reg );
            zv[2] = _mm256_loadu_pd( z0 + 2*n_elem_per_reg );
            zv[3] = _mm256_loadu_pd( z0 + 3*n_elem_per_reg );

            // zv = zv + alphaxRv * xv
            // zv = zR0 + axR.xR0, zI0 + axR.xI0, ...
            zv[0] = _mm256_fmadd_pd( xv[0], alphaxRv, zv[0] );
            zv[1] = _mm256_fmadd_pd( xv[1], alphaxRv, zv[1] );
            zv[2] = _mm256_fmadd_pd( xv[2], alphaxRv, zv[2] );
            zv[3] = _mm256_fmadd_pd( xv[3], alphaxRv, zv[3] );

            // Shuffling xv
            // xv = xI0 xR0 xI1 xR1
            xv[0] = _mm256_permute_pd( xv[0], 5 );
            xv[1] = _mm256_permute_pd( xv[1], 5 );
            xv[2] = _mm256_permute_pd( xv[2], 5 );
            xv[3] = _mm256_permute_pd( xv[3], 5 );

            // zv = zv + alphaxIv * xv
            // zv = zR0 + axR.xR0 - axI.xI0, zI0 + axR.xI0 + axI.xR0, ...
            zv[0] = _mm256_fmadd_pd( xv[0], alphaxIv, zv[0] );
            zv[1] = _mm256_fmadd_pd( xv[1], alphaxIv, zv[1] );
            zv[2] = _mm256_fmadd_pd( xv[2], alphaxIv, zv[2] );
            zv[3] = _mm256_fmadd_pd( xv[3], alphaxIv, zv[3] );

            // zv = zv + alphayRv * yv
            // zv = zR0 + axR.xR0 - axI.xI0 + ayR.yR0,
            //      zI0 + axR.xI0 + axI.xR0 + ayR.yI0, ...
            zv[0] = _mm256_fmadd_pd( yv[0], alphayRv, zv[0] );
            zv[1] = _mm256_fmadd_pd( yv[1], alphayRv, zv[1] );
            zv[2] = _mm256_fmadd_pd( yv[2], alphayRv, zv[2] );
            zv[3] = _mm256_fmadd_pd( yv[3], alphayRv, zv[3] );

            // Shuffling yv
            // yv = yI0 yR0 yI1 yR1
            yv[0] = _mm256_permute_pd( yv[0], 5 );
            yv[1] = _mm256_permute_pd( yv[1], 5 );
            yv[2] = _mm256_permute_pd( yv[2], 5 );
            yv[3] = _mm256_permute_pd( yv[3], 5 );

            // zv = zv + alphayIv * yv
            // zv = zR0 + axR.xR0 - axI.xI0 + ayR.yR0 - ayI.yI0,
            //      zI0 + axR.xI0 + axI.xR0 + ayR.yI0 + ayI.yR0, ...
            zv[0] = _mm256_fmadd_pd( yv[0], alphayIv, zv[0] );
            zv[1] = _mm256_fmadd_pd( yv[1], alphayIv, zv[1] );
            zv[2] = _mm256_fmadd_pd( yv[2], alphayIv, zv[2] );
            zv[3] = _mm256_fmadd_pd( yv[3], alphayIv, zv[3] );

            // Storing results from zv
            _mm256_storeu_pd( (z0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_pd( (z0 + 1*n_elem_per_reg), zv[1] );
            _mm256_storeu_pd( (z0 + 2*n_elem_per_reg), zv[2] );
            _mm256_storeu_pd( (z0 + 3*n_elem_per_reg), zv[3] );

            x0 += 4*n_elem_per_reg;
            y0 += 4*n_elem_per_reg;
            z0 += 4*n_elem_per_reg;
        }

        // Processing 4 elements per loop, 8 FMAs
        for ( ; ( i + 3 ) < n; i += 4 )
        {
            // Loading x vector
            // xv = xR0 xI0 xR1 xI1
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

            // Loading y vector
            // yv = yR0 yI0 yR1 yI1
            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            // Loading z vector
            // zv = zR0 zI0 zR1 zI1
            zv[0] = _mm256_loadu_pd( z0 + 0*n_elem_per_reg );
            zv[1] = _mm256_loadu_pd( z0 + 1*n_elem_per_reg );

            // zv = zv + alphaxRv * xv
            // zv = zR0 + axR.xR0, zI0 + axR.xI0, ...
            zv[0] = _mm256_fmadd_pd( xv[0], alphaxRv, zv[0] );
            zv[1] = _mm256_fmadd_pd( xv[1], alphaxRv, zv[1] );

            // Shuffling xv
            // xv = xI0 xR0 xI1 xR1
            xv[0] = _mm256_permute_pd( xv[0], 5 );
            xv[1] = _mm256_permute_pd( xv[1], 5 );

            // zv = zv + alphaxIv * xv
            // zv = zR0 + axR.xR0 - axI.xI0, zI0 + axR.xI0 + axI.xR0, ...
            zv[0] = _mm256_fmadd_pd( xv[0], alphaxIv, zv[0] );
            zv[1] = _mm256_fmadd_pd( xv[1], alphaxIv, zv[1] );

            // zv = zv + alphayRv * yv
            // zv = zR0 + axR.xR0 - axI.xI0 + ayR.yR0,
            //      zI0 + axR.xI0 + axI.xR0 + ayR.yI0, ...
            zv[0] = _mm256_fmadd_pd( yv[0], alphayRv, zv[0] );
            zv[1] = _mm256_fmadd_pd( yv[1], alphayRv, zv[1] );

            // Shuffling yv
            // yv = yI0 yR0 yI1 yR1
            yv[0] = _mm256_permute_pd( yv[0], 5 );
            yv[1] = _mm256_permute_pd( yv[1], 5 );

            // zv = zv + alphayIv * yv
            // zv = zR0 + axR.xR0 - axI.xI0 + ayR.yR0 - ayI.yI0,
            //      zI0 + axR.xI0 + axI.xR0 + ayR.yI0 + ayI.yR0, ...
            zv[0] = _mm256_fmadd_pd( yv[0], alphayIv, zv[0] );
            zv[1] = _mm256_fmadd_pd( yv[1], alphayIv, zv[1] );

            // Storing results from zv
            _mm256_storeu_pd( (z0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_pd( (z0 + 1*n_elem_per_reg), zv[1] );

            x0 += 2*n_elem_per_reg;
            y0 += 2*n_elem_per_reg;
            z0 += 2*n_elem_per_reg;
        }

        // Processing 2 elements per loop, 4FMAs
        for ( ; ( i + 1 ) < n; i += 2 )
        {
            // Loading x vector
            // xv = xR0 xI0 xR1 xI1
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

            // Loading y vector
            // yv = yR0 yI0 yR1 yI1
            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            // Loading z vector
            // zv = zR0 zI0 zR1 zI1
            zv[0] = _mm256_loadu_pd( z0 + 0*n_elem_per_reg );

            // zv = zv + alphaxRv * xv
            // zv = zR0 + axR.xR0, zI0 + axR.xI0, ...
            zv[0] = _mm256_fmadd_pd( xv[0], alphaxRv, zv[0] );

            // Shuffling xv
            // xv = xI0 xR0 xI1 xR1
            xv[0] = _mm256_permute_pd( xv[0], 5 );

            // zv = zv + alphaxIv * xv
            // zv = zR0 + axR.xR0 - axI.xI0, zI0 + axR.xI0 + axI.xR0, ...
            zv[0] = _mm256_fmadd_pd( xv[0], alphaxIv, zv[0] );

            // zv = zv + alphayRv * yv
            // zv = zR0 + axR.xR0 - axI.xI0 + ayR.yR0,
            //      zI0 + axR.xI0 + axI.xR0 + ayR.yI0, ...
            zv[0] = _mm256_fmadd_pd( yv[0], alphayRv, zv[0] );

            // Shuffling yv
            // yv = yI0 yR0 yI1 yR1
            yv[0] = _mm256_permute_pd( yv[0], 5 );

            // zv = zv + alphayIv * yv
            // zv = zR0 + axR.xR0 - axI.xI0 + ayR.yR0 - ayI.yI0,
            //      zI0 + axR.xI0 + axI.xR0 + ayR.yI0 + ayI.yR0, ...
            zv[0] = _mm256_fmadd_pd( yv[0], alphayIv, zv[0] );

            // Storing results from zv
            _mm256_storeu_pd( (z0 + 0*n_elem_per_reg), zv[0] );

            x0 += 1*n_elem_per_reg;
            y0 += 1*n_elem_per_reg;
            z0 += 1*n_elem_per_reg;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).
        _mm256_zeroupper();

        if ( !bli_is_conj( conjx ) && !bli_is_conj( conjy ) )
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR - axI.xI + ayR.yR - ayI.yI )
                *z0       += (*alphax0) * (*x0) -
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) -
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI + axI.xR + ayR.yI + ayI.yR )
                *(z0 + 1) += (*alphax0) * (*(x0 + 1)) +
                             (*(alphax0 + 1)) * (*x0) +
                             (*alphay0) * (*(y0 + 1)) +
                             (*(alphay0 + 1)) * (*y0);

                x0 += 2;
                y0 += 2;
                z0 += 2;
            }
        }
        else if ( !bli_is_conj( conjx ) && bli_is_conj( conjy ) )
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR - axI.xI + ayR.yR + ayI.yI )
                *z0       += (*alphax0) * (*x0) -
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) +
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI + axI.xR + ayR.yI - ayI.yR )
                *(z0 + 1) += (*alphax0) * (*(x0 + 1)) +
                             (*(alphax0 + 1)) * (*x0) +
                             (*(alphay0 + 1)) * (*y0) -
                             (*alphay0) * (*(y0 + 1));

                x0 += 2;
                y0 += 2;
                z0 += 2;
            }
        }
        else if ( bli_is_conj( conjx ) && !bli_is_conj( conjy ) )
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR + axI.xI + ayR.yR - ayI.yI )
                *z0       += (*alphax0) * (*x0) +
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) -
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI - axI.xR + ayR.yI + ayI.yR )
                *(z0 + 1) += (*(alphax0 + 1)) * (*x0) -
                             (*alphax0) * (*(x0 + 1)) +
                             (*alphay0) * (*(y0 + 1)) +
                             (*(alphay0 + 1)) * (*y0);

                x0 += 2;
                y0 += 2;
                z0 += 2;
            }
        }
        else
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR + axI.xI + ayR.yR + ayI.yI )
                *z0       += (*alphax0) * (*x0) +
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) +
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI - axI.xR + ayR.yI - ayI.yR )
                *(z0 + 1) += (*(alphax0 + 1)) * (*x0) -
                             (*alphax0) * (*(x0 + 1)) +
                             (*(alphay0 + 1)) * (*y0) -
                             (*alphay0) * (*(y0 + 1));

                x0 += 2;
                y0 += 2;
                z0 += 2;
            }
        }
    }
    else
    {
        // Using scalar code for non-unit increments
        if ( !bli_is_conj( conjx ) && !bli_is_conj( conjy ) )
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR - axI.xI + ayR.yR - ayI.yI )
                *z0       += (*alphax0) * (*x0) -
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) -
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI + axI.xR + ayR.yI + ayI.yR )
                *(z0 + 1) += (*alphax0) * (*(x0 + 1)) +
                             (*(alphax0 + 1)) * (*x0) +
                             (*alphay0) * (*(y0 + 1)) +
                             (*(alphay0 + 1)) * (*y0);

                x0 += 2 * incx;
                y0 += 2 * incy;
                z0 += 2 * incz;
            }
        }
        else if ( !bli_is_conj( conjx ) && bli_is_conj( conjy ) )
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR - axI.xI + ayR.yR + ayI.yI )
                *z0       += (*alphax0) * (*x0) -
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) +
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI + axI.xR + ayR.yI - ayI.yR )
                *(z0 + 1) += (*alphax0) * (*(x0 + 1)) +
                             (*(alphax0 + 1)) * (*x0) +
                             (*(alphay0 + 1)) * (*y0) -
                             (*alphay0) * (*(y0 + 1));

                x0 += 2 * incx;
                y0 += 2 * incy;
                z0 += 2 * incz;
            }
        }
        else if ( bli_is_conj( conjx ) && !bli_is_conj( conjy ) )
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR + axI.xI + ayR.yR - ayI.yI )
                *z0       += (*alphax0) * (*x0) +
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) -
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI - axI.xR + ayR.yI + ayI.yR )
                *(z0 + 1) += (*(alphax0 + 1)) * (*x0) -
                             (*alphax0) * (*(x0 + 1)) +
                             (*alphay0) * (*(y0 + 1)) +
                             (*(alphay0 + 1)) * (*y0);

                x0 += 2 * incx;
                y0 += 2 * incy;
                z0 += 2 * incz;
            }
        }
        else
        {
            for ( ; i < n; i++ )
            {
                // zR     += ( axR.xR + axI.xI + ayR.yR + ayI.yI )
                *z0       += (*alphax0) * (*x0) +
                             (*(alphax0 + 1)) * (*(x0 + 1)) +
                             (*alphay0) * (*y0) +
                             (*(alphay0 + 1)) * (*(y0 + 1));

                // zI     += ( axR.xI - axI.xR + ayR.yI - ayI.yR )
                *(z0 + 1) += (*(alphax0 + 1)) * (*x0) -
                             (*alphax0) * (*(x0 + 1)) +
                             (*(alphay0 + 1)) * (*y0) -
                             (*alphay0) * (*(y0 + 1));

                x0 += 2 * incx;
                y0 += 2 * incy;
                z0 += 2 * incz;
            }
        }
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
