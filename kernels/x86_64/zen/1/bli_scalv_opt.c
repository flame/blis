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

void bli_sscalv_opt(
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

    chi1 = x;
    v8ff_t ymm1; // to store alpha_conj
    v8ff_t ymm2; // x
    v8ff_t ymm3; // output
    inc_t packs    = n / 8; // Number of sets of 8 element floats in vector of size n
    inc_t elements = n % 8; // Number of elements in the last set (< 8)

    ymm1.v = _mm256_broadcast_ss(alpha);

    if (incx == 1)
    {
        //for (i = 0; i < n; ++i)
        for (i = 0; i < packs; i++)
        {
            ymm2.v = _mm256_loadu_ps(x + i * 8);
            ymm3.v = _mm256_mul_ps(ymm1.v, ymm2.v);
            _mm256_storeu_ps(x + i * 8, ymm3.v);
                //((chi1[i])) = ((alpha_conj)) * ((chi1[i]));
        }
        inc_t offset = packs * 8;
        for (i = 0; i < elements; i++)
        {
            ((chi1[i + offset])) = ((alpha_conj)) * ((chi1[i + offset]));
        }
    }
    else
    {
        for (i = 0; i < n; ++i)
        {
            {
                ((*chi1)) = ((alpha_conj)) * ((*chi1));
            };

            chi1 += incx;
        }
    }
}

void bli_dscalv_opt(
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

  chi1 = x;
  v4df_t ymm1; // to store alpha_conj
  v4df_t ymm2; // x
  v4df_t ymm3; // output
  inc_t packs = n / 4; // Number of sets of 4 element doubles in vector of size n
  inc_t elements = n % 4; // Number of elements in the last set (< 4)

  ymm1.v = _mm256_broadcast_sd(alpha);

  if (incx == 1)
  {
      for (i = 0; i < packs; i++)
      {
          ymm2.v = _mm256_loadu_pd(x + i * 4);
          ymm3.v = _mm256_mul_pd(ymm1.v, ymm2.v);
          _mm256_storeu_pd(x + i * 4, ymm3.v);
          //((chi1[i])) = ((alpha_conj)) * ((chi1[i]));
      }
      inc_t offset = packs * 4;
      for (i = 0; i < elements; i++)
      {
          ((chi1[i + offset])) = ((alpha_conj)) * ((chi1[i + offset]));
      }
  }
  else
  {
      for (i = 0; i < n; ++i)
      {
          {
              ((*chi1)) = ((alpha_conj)) * ((*chi1));
          };

          chi1 += incx;
      }
  }
}  // End of function
