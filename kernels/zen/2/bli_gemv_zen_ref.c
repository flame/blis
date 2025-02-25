/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

/* This routine assumes that the matrix is stored in column-major order
 * and does not have transpose.
 */
void bli_dgemv_zen_ref_c
     (
      conj_t           conja,
      conj_t           conjx,
      dim_t            m,
      dim_t            n,
      double* restrict alpha,
      double* restrict a, inc_t inca, inc_t lda,
      double* restrict x, inc_t incx,
      double* restrict beta,
      double* restrict y, inc_t incy,
      cntx_t* restrict cntx
     )
{

    double* restrict x0 = x;
    double* restrict y0 = y;
    double* restrict a0 = a;

    dim_t i,j;

    if((incy == 1) && (incx == 1))
    {

        /* n==0 case will be handled in the framework,
           where it returns immediately.
        * For the first column of A, execute Y = alpha*A*X + beta*Y
        */
        double x0_val = (x0[0]);
        /* if beta = 0, do not scale y with beta */
        if(PASTEMAC(d,eq0)(*beta))
        {   PRAGMA_SIMD
            for(i = 0; i < m; i++)
                y0[i] = (a0[i]) * (x0_val) * (*alpha);
        }
        else
        {
            PRAGMA_SIMD
            for(i = 0; i < m; i++)
                (y0[i]) = (a0[i]) * (x0_val) * (*alpha) + y0[i] * (*beta);
        }
        a0 += lda;

        /* For remaining columns of A, execute,  Y = alpha*A*X + Y; */
        for(j = 1; j < n; j++)
        {
            const double xp = (x0[j]);
            PRAGMA_SIMD
            for(i = 0; i < m; i++)
            {
                (y0[i]) += (a0[i]) * xp * (*alpha);
            }
            a0 += lda;
        }

    }
    else
    {
        /*if beta = 0, populate y vector with zeroes,
         * otherwise scale with beta */
        if(PASTEMAC(d,eq0)(*beta))
        {
            for(j = 0; j < m; j++)
                y0[j*incy] = 0.0;
        }
        else
        {
            for(j = 0; j < m; j++)
                PASTEMAC(d,scals)(*beta, *(y0+j*incy))
        }

        for(j = 0; j < n; j++)
        {
            const double xp = *(x0+j*incx);
            for(i = 0; i < m; i++)
            {
                *(y0 + i*incy) += (a0[j*lda+i]) * xp * (*alpha);
            }
        }
    }
    return;
}

/**
 * bli_dgemv_zen_ref( ... )
 * This reference kernel for DGEMV supports row/colum storage schemes for both
 * transpose and no-transpose cases.
 */
void bli_dgemv_zen_ref
     (
      trans_t          transa,
      dim_t            m,
      dim_t            n,
      double* restrict alpha,
      double* restrict a, inc_t inca, inc_t lda,
      double* restrict x, inc_t incx,
      double* restrict beta,
      double* restrict y, inc_t incy,
      cntx_t* restrict cntx
     )
{
    dim_t m0 = m;
    dim_t n0 = n;
    dim_t leny = m0;    // Initializing length of y vector.

    double* a0 = (double*) a;
    double* x0 = (double*) x;
    double* y0 = (double*) y;

    if ( bli_is_trans( transa ) || bli_is_conjtrans( transa ) )
    {
        // Updating length of y matrix if transpose is enabled.
        leny = n0;
    }

    // Perform y := beta * y
    if ( !bli_deq1(*beta) ) // beta != 1
    {
        if ( bli_deq0(*beta) )  // beta == 0
        {
            for ( dim_t i = 0; i < leny; ++i )
            {
                PASTEMAC(d,sets)( 0.0, 0.0, *(y0 + i*incy))
            }
        }
        else    // beta != 0
        {
            for ( dim_t i = 0; i < leny; ++i )
            {
                PASTEMAC(d,scals)( *beta, *(y0 + i*incy) )
            }
        }
    }

    // If alpha == 0, return.
    if ( bli_deq0( *alpha ) ) return;

    if ( bli_is_notrans( transa ) )     // BLIS_NO_TRANSPOSE
    {
        if ( incy == 1 )
        {
            for ( dim_t i = 0; i < n0; ++i )
            {
                double rho = (*alpha) * (*x0);
                for ( dim_t j = 0; j < m0; ++j )
                {
                    *(y0 + j) += rho * (*(a0 + j*inca));
                }
                x0 += incx;
                a0 += lda;
            }
        }
        else // if ( incy != 1 )
        {
            for ( dim_t i = 0; i < n0; ++i )
            {
                double rho = (*alpha) * (*x0);
                for ( dim_t j = 0; j < m0; ++j )
                {
                    *(y0 + j*incy) += rho * (*(a0 + j*inca));
                }
                x0 += incx;
                a0 += lda;
            }
        }
    }
    else    // BLIS_TRANSPOSE
    {
        if ( incx == 1 )
        {
            for ( dim_t i = 0; i < n0; ++i )
            {
                double rho = 0.0;
                for ( dim_t j = 0; j < m0; ++j )
                {
                    rho += (*(a0 + j*inca)) * (*(x0 + j));
                }
                (*y0) += (*alpha) * rho;
                y0 += incy;
                a0 += lda;
            }
        }
        else    // if ( incx != 1 )
        {
            for ( dim_t i = 0; i < n0; ++i )
            {
                double rho = 0.0;
                for ( dim_t j = 0; j < m0; ++j )
                {
                    rho += (*(a0 + j*inca)) * (*(x0 + j*incx));
                }
                (*y0) += (*alpha) * rho;
                y0 += incy;
                a0 += lda;
            }
        }
    }
}
