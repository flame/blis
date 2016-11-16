/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.
   
   Copyright (C) 2014, The University of Texas at Austin
   Copyright(C) 2016, Advanced Micro Devices, Inc.
   
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
   derived from this software without specific prior written permission.
   
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
#include <immintrin.h>

typedef union
{
    __m256 v;
    float f[8];
}v8ff_t;

typedef union
{
    __m256d v;
    double d[4];
}v4df_t;

void bli_sdotxv_unb_var1      ( 
                                 conj_t          conjx, 
                                 conj_t          conjy, 
                                 dim_t           n, 
                                 float* restrict alpha, 
                                 float* restrict x, inc_t incx, 
                                 float* restrict y, inc_t incy, 
                                 float* restrict beta, 
                                 float* restrict rho, 
                                 cntx_t*         cntx  
                              ) 
{ 
    float* restrict chi1; 
    float* restrict psi1; 
    float  dotxy; 
    dim_t  i; 
    conj_t conjx_use; 

    /* If beta is zero, clear rho. Otherwise, scale by beta. */ 
    if ( ( ((*beta)) == (0.0F) ) ) 
    { 
      ((*rho)) = (0.0F);
    } 
    else 
    { 
      (( *rho )) = (( *beta )) * (( *rho ));        
    } 

    if ( n == 0 ) return; 

    dotxy = (0.0F);

    chi1 = x; 
    psi1 = y;

    dim_t n_run;
    dim_t n_left;    

    const dim_t n_elem_per_reg = 8; // 256 bit register can hold 8 floats
    const dim_t n_iter_unroll = 4;  // In a single loop 4 vectors of length 8 will be processed

    v8ff_t            rho1v, rho2v, rho3v, rho4v;
    v8ff_t            x1v, y1v;
    v8ff_t            x2v, y2v;
    v8ff_t            x3v, y3v;
    v8ff_t            x4v, y4v;

    n_run = (n) / (n_elem_per_reg * n_iter_unroll);
    n_left = (n) % (n_elem_per_reg * n_iter_unroll);

    rho1v.v = _mm256_setzero_ps();
    rho2v.v = _mm256_setzero_ps();
    rho3v.v = _mm256_setzero_ps();
    rho4v.v = _mm256_setzero_ps();

    /* If y must be conjugated, we do so indirectly by first toggling the
       effective conjugation of x and then conjugating the resulting dot
       product. */ 
    conjx_use = conjx; 

    if ( ( conjy == BLIS_CONJUGATE ) ) 
    {
        conjx_use = ( conjx_use ^ ( 0x1  << 4 ) ); 
    }

    if (incx == 1 && incy == 1)
    {
        for (i = 0; i < n_run; ++i)
        {
            //(( dotxy )) += (( psi1[i] )) * (( chi1[i] ));

            // load the input
            x1v.v = _mm256_loadu_ps((float*)chi1);
            y1v.v = _mm256_loadu_ps((float*)psi1);

            x2v.v = _mm256_loadu_ps((float*)(chi1 + n_elem_per_reg));
            y2v.v = _mm256_loadu_ps((float*)(psi1 + n_elem_per_reg));

            x3v.v = _mm256_loadu_ps((float*)(chi1 + 2 * n_elem_per_reg));
            y3v.v = _mm256_loadu_ps((float*)(psi1 + 2 * n_elem_per_reg));

            x4v.v = _mm256_loadu_ps((float*)(chi1 + 3 * n_elem_per_reg));
            y4v.v = _mm256_loadu_ps((float*)(psi1 + 3 * n_elem_per_reg));

            // Calculate the dot product
            rho1v.v += y1v.v * x1v.v;
            rho2v.v += y2v.v * x2v.v;
            rho3v.v += y3v.v * x3v.v;
            rho4v.v += y4v.v * x4v.v;

            chi1 += (n_elem_per_reg * n_iter_unroll);
            psi1 += (n_elem_per_reg * n_iter_unroll);
        }

        //accumulate the results
        rho1v.v += rho2v.v;
        rho3v.v += rho4v.v;
        rho1v.v += rho3v.v;

        dotxy += rho1v.f[0] + rho1v.f[1] + rho1v.f[2] + rho1v.f[3] +
                 rho1v.f[4] + rho1v.f[5] + rho1v.f[6] + rho1v.f[7];

        if (n_left > 0)
        {
            for (i = 0; i < n_left; ++i)
            {
                dotxy += (*chi1) * (*psi1);

                chi1 += incx;
                psi1 += incy;
            }
        }
    }
    else
    {
        for (i = 0; i < n; ++i)
        {            
	  ((dotxy)) += ((*psi1)) * ((*chi1));            

            chi1 += incx;
            psi1 += incy;
        }
    }

    ((*rho)) += ((*alpha)) * ((dotxy));
} // End of function


void bli_ddotxv_unb_var1 (
                            conj_t          conjx,
                            conj_t          conjy,
                            dim_t           n,
                            double* restrict alpha,
                            double* restrict x, inc_t incx,
                            double* restrict y, inc_t incy,
                            double* restrict beta,
                            double* restrict rho,
                            cntx_t*         cntx  
                         )
{
    double* restrict chi1;
    double* restrict psi1;
    double  dotxy;
    dim_t  i;
    conj_t conjx_use;

    /* If beta is zero, clear rho. Otherwise, scale by beta. */
    if ((*beta) == (0.0))
    {
        (*rho) = (0.0);
    }
    else
    {
        ((*rho)) = ((*beta)) * ((*rho));
    }

    if (n == 0) return;

    dotxy = (0.0);

    chi1 = x;
    psi1 = y;

    dim_t n_run;
    dim_t n_left;

    const dim_t n_elem_per_reg = 4; // 256 bit register can hold 4 double
    const dim_t n_iter_unroll  = 4;  // In a single loop 4 vectors of length 4 will be processed

    v4df_t            rho1v;
    v4df_t            x1v, y1v;

    v4df_t            rho2v;
    v4df_t            x2v, y2v;

    v4df_t            rho3v;
    v4df_t            x3v, y3v;

    v4df_t            rho4v;
    v4df_t            x4v, y4v;

    rho1v.v = _mm256_setzero_pd();
    rho2v.v = _mm256_setzero_pd();
    rho3v.v = _mm256_setzero_pd();
    rho4v.v = _mm256_setzero_pd();

    n_run       = ( n ) / (n_elem_per_reg * n_iter_unroll);
    n_left      = ( n ) % (n_elem_per_reg * n_iter_unroll);


    /* If y must be conjugated, we do so indirectly by first toggling the
    effective conjugation of x and then conjugating the resulting dot
    product. */
    conjx_use = conjx;

    if ((conjy == BLIS_CONJUGATE))
    {
      conjx_use = (conjx_use ^ (0x1 << 4));
    }

    if (incx == 1 && incy == 1)
    {
        for (i = 0; i < n_run; ++i)
        {
            //(( dotxy )) += (( psi1[i] )) * (( chi1[i] ));

            // load the input
            x1v.v = _mm256_loadu_pd((double*)chi1);
            y1v.v = _mm256_loadu_pd((double*)psi1);

            y2v.v = _mm256_loadu_pd((double*)(psi1 + n_elem_per_reg));
            x2v.v = _mm256_loadu_pd((double*)(chi1 + n_elem_per_reg));

            x3v.v = _mm256_loadu_pd((double*)(chi1 + 2 * n_elem_per_reg));
            y3v.v = _mm256_loadu_pd((double*)(psi1 + 2 * n_elem_per_reg));

            x4v.v = _mm256_loadu_pd((double*)(chi1 + 3 * n_elem_per_reg));
            y4v.v = _mm256_loadu_pd((double*)(psi1 + 3 * n_elem_per_reg));

            // Calculate the dot product
            rho1v.v += y1v.v * x1v.v;
            rho2v.v += y2v.v * x2v.v;
            rho3v.v += y3v.v * x3v.v;
            rho4v.v += y4v.v * x4v.v;

            chi1 += (n_elem_per_reg * n_iter_unroll);
            psi1 += (n_elem_per_reg * n_iter_unroll);
        }

        //accumulate the results
        rho1v.v += rho2v.v;
        rho3v.v += rho4v.v;
        rho1v.v += rho3v.v;

        dotxy += rho1v.d[0] + rho1v.d[1] + rho1v.d[2] + rho1v.d[3];
            
        if (n_left > 0)
        {
            for (i = 0; i < n_left; ++i)
            {
                dotxy += (*chi1) * (*psi1);

                chi1 += incx;
                psi1 += incy;
            }
        }
    }
    else
    {
        for (i = 0; i < n; ++i)
        {
            dotxy += ((*psi1)) * ((*chi1));

            chi1 += incx;
            psi1 += incy;
        }
    }

    ((*rho)) += ((*alpha)) * ((dotxy));

}// End of function
