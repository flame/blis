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


void bli_sscalv_opt_var2(
                          conj_t          conjalpha,
                          dim_t           n,
                          float* restrict alpha,
                          float* restrict x, 
                          inc_t           incx,
                          cntx_t*         cntx
                        )
{
    float* restrict chi1;
    float  alpha_conj;
    dim_t  i;
    

    if (((n) == 0)) return;

    /* If alpha is one, return. */
    if ((((*alpha)) == (1.0F))) return;

    /* If alpha is zero, use setv. */
    if ((((*alpha)) == (0.0F)))
    {
        float* zero = ((float*)(void*)(
            ((char*)(((BLIS_ZERO).buffer))) +
            (dim_t)(BLIS_FLOAT * sizeof(dcomplex))
            ));

        /* Query the context for the kernel function pointer. */
        const num_t         dt = (BLIS_FLOAT);
        ssetv_ft setv_p = (((&((((cntx))->l1v_kers))[BLIS_SETV_KER]))->ptr[(dt)]);

        setv_p(
            BLIS_NO_CONJUGATE,
            n,
            zero,
            x, incx,
            cntx
            );
        return;
    }

    {
        {
            (((alpha_conj))) = (((*alpha)));
        };
    };

   
    float* x1 = x;
    __m256 alphaV;
    __m256 x1v; //
    __m256 x2v;
    __m256 y1v; // x
    __m256 y2v; 

    if (incx == 1)
    {
      alphaV = _mm256_broadcast_ss(alpha);

        for (i = 0; i+15 < n; i += 16)       
        {
	  x1v = _mm256_loadu_ps(x1);
	  x2v = _mm256_loadu_ps(x1 + 8);
	  y1v = _mm256_mul_ps(x1v, alphaV);
	  y2v = _mm256_mul_ps(x2v, alphaV);
	  _mm256_storeu_ps(x1, y1v);
	  _mm256_storeu_ps(x1 + 8, y2v);
	  x1 += 16;
	}
	for (; i + 7 < n; i += 8)
	  {
	    x1v = _mm256_loadu_ps(x1);
	    y1v = _mm256_mul_ps(x1v, alphaV);
	    _mm256_storeu_ps( (float*)x1, y1v);
	    x1 += 8;
	  }
	for (; i < n; i++)
	  {
	    x[i] = (alpha_conj) * x[i];
	  }
    }
    else
    {
      chi1 = x;
        for (i = 0; i < n; ++i)
        {
            {
                ((*chi1)) = ((alpha_conj)) * ((*chi1));
            };

            chi1 += incx;
        }
    }
}// end of function


void bli_dscalv_opt_var2(
                      conj_t           conjalpha,
                      dim_t            n,
                      double* restrict alpha,
                      double* restrict x, 
                      inc_t           incx,
                      cntx_t*         cntx
                   )
{
    double* restrict chi1;
    double  alpha_conj;
    dim_t  i;

    if (((n) == 0)) return;

    /* If alpha is one, return. */
    if ((((*alpha)) == (1.0))) return;

    /* If alpha is zero, use setv. */
    if ((((*alpha)) == (0.0)))
    {
        double* zero = ((double*)(void*)(
            ((char*)(((BLIS_ZERO).buffer))) +
            (dim_t)(BLIS_DOUBLE * sizeof(dcomplex))
            ));

        /* Query the context for the kernel function pointer. */
        const num_t         dt = (BLIS_DOUBLE);
        dsetv_ft setv_p = (((&((((cntx))->l1v_kers))[BLIS_SETV_KER]))->ptr[(dt)]);

        setv_p(
                BLIS_NO_CONJUGATE,
                n,
                zero,
                x, incx,
                cntx
            );
        return;
    }

  {
      {
          (((alpha_conj))) = (((*alpha)));
      };
  };
  
  double* x1 = x;
    __m256d alphaV;
    __m256d x1v; //
    __m256d x2v;
    __m256d y1v; 
    __m256d y2v; 

    if (incx == 1)
    {
      alphaV = _mm256_broadcast_sd(alpha);

        for (i = 0; i+7 < n; i += 8)       
        {
	  x1v = _mm256_loadu_pd(x1);
	  x2v = _mm256_loadu_pd(x1 + 4);
	  y1v = _mm256_mul_pd(x1v, alphaV);
	  y2v = _mm256_mul_pd(x2v, alphaV);
	  _mm256_storeu_pd(x1, y1v);
	  _mm256_storeu_pd(x1 + 4, y2v);
	  x1 += 8;
	}
	for (; i + 3 < n; i += 4)
	  {
	    x1v = _mm256_loadu_pd(x1);
	    y1v = _mm256_mul_pd(x1v, alphaV);
	    _mm256_storeu_pd( (double*)x1, y1v);
	    x1 += 4;
	  }
	for (; i < n; i++)
	  {
	    x[i] = (alpha_conj) * x[i];
	  }
    }
    else
    {
      chi1 = x;
        for (i = 0; i < n; ++i)
        {
            {
                ((*chi1)) = ((alpha_conj)) * ((*chi1));
            };

            chi1 += incx;
        }
    }
}// end of function
