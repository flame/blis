/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc.

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


// Union data structure to access AVX registers
//  One 256-bit AVX register holds 4 DP elements.
typedef union
{
  __m256d v;
  double  d[4] __attribute__((aligned(64)));
} v4df_t;




// packing routine for dgemm/trsm
// when op(A) = n & op(B) = n
void bli_dpackm_8xk_nn_zen
(
    conj_t           conja,
    pack_t           schema,
    dim_t            cdim,
    dim_t            n,
    dim_t            n_max,
    void*   restrict kappa,
    void*   restrict a, inc_t inca, inc_t lda,  // inca = 1
    void*   restrict p, inc_t ldp,
    cntx_t* restrict cntx
)
{
  double* restrict alpha1 = a;
  double* restrict pi1 = p;

    dim_t           n_iter = n / 2;
    dim_t           n_left = n % 2;

    if (cdim == 8)
    {
      // (*kappa_cast) = 1.0 for GEMM
      __m256d ymmSrc_0_0123; // source registers
      __m256d ymmSrc_0_4567;
      __m256d ymmSrc_1_0123;
      __m256d ymmSrc_1_4567;

      for (; n_iter != 0; --n_iter)
	{
	  // Works when inca = 1, which is the case for op(A) = n and op(B) = n
	  ymmSrc_0_0123 = _mm256_loadu_pd(alpha1 + 0 * inca + 0 * lda);
	  ymmSrc_0_4567 = _mm256_loadu_pd(alpha1 + 4 * inca + 0 * lda);
	  ymmSrc_1_0123 = _mm256_loadu_pd(alpha1 + 0 * inca + 1 * lda);
	  ymmSrc_1_4567 = _mm256_loadu_pd(alpha1 + 4 * inca + 1 * lda);

          // Store
#if 1
	  _mm256_storeu_pd((pi1 + 0 + 0 * ldp), ymmSrc_0_0123);
	  _mm256_storeu_pd((pi1 + 4 + 0 * ldp), ymmSrc_0_4567);

	  _mm256_storeu_pd((pi1 + 0 + 1 * ldp), ymmSrc_1_0123);
	  _mm256_storeu_pd((pi1 + 4 + 1 * ldp), ymmSrc_1_4567);
#else
	  _mm256_stream_pd((pi1 + 0), ymmSrc_0_0123);
	  _mm256_stream_pd((pi1 + 4), ymmSrc_0_4567);

	  _mm256_stream_pd((pi1 + 0 + 1 * ldp), ymmSrc_1_0123);
	  _mm256_stream_pd((pi1 + 4 + 1 * ldp), ymmSrc_1_4567);
#endif
	  alpha1 += 2 * lda;
	  pi1 += 2 * ldp;
	}

      if (n_left & 1) //for (; n_left != 0; --n_left)
	{
	  ymmSrc_0_0123 = _mm256_loadu_pd(alpha1 + 0 * inca);
	  ymmSrc_0_4567 = _mm256_loadu_pd(alpha1 + 4 * inca);

	  _mm256_storeu_pd((pi1 + 0), ymmSrc_0_0123);
	  _mm256_storeu_pd((pi1 + 4), ymmSrc_0_4567);

	  alpha1 += lda;
	  pi1 += ldp;
	}
    }
    else /* if ( cdim < mnr ) */
    {
      double* restrict a_cast = a;
      double* restrict p_cast = p;
        // (*kappa_cast == 1.0) for GEMM

        PRAGMA_SIMD
        for (dim_t j = 0; j < n; ++j)
            for (dim_t i = 0; i < cdim; ++i)
                p_cast[i + j*ldp] = a_cast[i + j*lda];


        const dim_t     i = cdim;
        const dim_t     m_edge = 8 - cdim;
        const dim_t     n_edge = n_max;
	//     double* restrict p_cast = p;
        double* restrict p_edge = p_cast + (i) * 1;

        PRAGMA_SIMD
        for (dim_t j = 0; j < n_edge; ++j)
            for (dim_t i = 0; i < m_edge; ++i)
                *(p_edge + i + j*ldp) = 0.0;
    }

    if (n < n_max)
    {
        const dim_t     j = n;
        const dim_t     m_edge = 8;
        const dim_t     n_edge = n_max - n;
        double* restrict p_cast = p;
        double* restrict p_edge = p_cast + (j)*ldp;

	PRAGMA_SIMD
        for (dim_t j = 0; j < n_edge; ++j)
            for (dim_t i = 0; i < m_edge; ++i)
                *(p_edge + i + j*ldp) = 0.0;
    }
}// End of function


void bli_dpackm_6xk_nn_zen
(
    conj_t           conja,
    pack_t           schema,
    dim_t            cdim,
    dim_t            n,
    dim_t            n_max,
    void*   restrict kappa,
    void*   restrict a, inc_t inca, inc_t lda,
    void*   restrict p, inc_t ldp,
    cntx_t* restrict cntx
)
{
  double* restrict alpha1 = a;
  double* restrict pi1 = p;

  if (cdim == 6)
    {
      //if ( (*kappa_cast) == 1.0 ) // Kappa_cast = 1.0 for dgemm
      for (dim_t k = n; k != 0; --k)
        {
	  (*(pi1 + 0)) = (*(alpha1 + 0 * inca));
	  (*(pi1 + 1)) = (*(alpha1 + 1 * inca));
	  (*(pi1 + 2)) = (*(alpha1 + 2 * inca));
	  (*(pi1 + 3)) = (*(alpha1 + 3 * inca));

	  (*(pi1 + 4)) = (*(alpha1 + 4 * inca));
	  (*(pi1 + 5)) = (*(alpha1 + 5 * inca));

	  alpha1 += lda;
	  pi1 += ldp;
        }
    }
  else /* if ( cdim < mnr ) */
    {
      double* restrict a_cast = a;
      double* restrict p_cast = p;

      // (*kappa_cast == 1.0) for GEMM
      // a will be in row-major, inca != 1 and lda = 1
      PRAGMA_SIMD
      for (dim_t i = 0; i < cdim; ++i)
	for(dim_t j = 0; j < n; ++j)
	  p_cast[i + j*ldp] = a_cast[i * inca + j]; // i * inca + j * lda, lda = 1

     
      const dim_t     m_edge = 6 - cdim;
      const dim_t     n_edge = n_max;
      //     double* restrict p_cast = p;
      double* restrict p_edge = p_cast + (cdim) * 1;

      PRAGMA_SIMD
      for (dim_t j = 0; j < n_edge; ++j)
	for (dim_t i = 0; i < m_edge; ++i)
	  *(p_edge + i + j*ldp) = 0.0;
    }

  if (n < n_max)
    {
      const dim_t     j = n;
      const dim_t     m_edge = 6;
      const dim_t     n_edge = n_max - n;
      double* restrict p_cast = p;
      double* restrict p_edge = p_cast + (j)*ldp;

      PRAGMA_SIMD
      for (dim_t j = 0; j < n_edge; ++j)
	for (dim_t i = 0; i < m_edge; ++i)
	  *(p_edge + i + j*ldp) = 0.0;
    }
}// end of function



// Packing routine for general operations op(A) = ? op(B) = ?
void bli_dpackm_8xk_gen_zen
(
    conj_t           conja,
    pack_t           schema,
    dim_t            cdim,
    dim_t            n,
    dim_t            n_max,
    void*   restrict kappa,
    void*   restrict a, inc_t inca, inc_t lda,
    void*   restrict p, inc_t ldp,
    cntx_t* restrict cntx
)
{
    double* restrict kappa_cast = kappa;
    double* restrict alpha1 = a;
    double* restrict pi1 = p;

    dim_t           n_iter = n / 2;
    dim_t           n_left = n % 2;

    if (cdim == 8)
    {
        if ((*kappa_cast) == (1.0))
        {
            if (bli_is_conj(conja))
            {
                //for (dim_t k = n; k != 0; --k)
                for (dim_t k = n; k--;)
                {
                    (((*(pi1 + 0)))) = (((*(alpha1 + 0 * inca))));
                    (((*(pi1 + 1)))) = (((*(alpha1 + 1 * inca))));
                    (((*(pi1 + 2)))) = (((*(alpha1 + 2 * inca))));
                    (((*(pi1 + 3)))) = (((*(alpha1 + 3 * inca))));
                    (((*(pi1 + 4)))) = (((*(alpha1 + 4 * inca))));
                    (((*(pi1 + 5)))) = (((*(alpha1 + 5 * inca))));
                    (((*(pi1 + 6)))) = (((*(alpha1 + 6 * inca))));
                    (((*(pi1 + 7)))) = (((*(alpha1 + 7 * inca))));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
            else
            {
                for (; n_iter != 0; --n_iter)
                {

                    ((*(pi1 + 0 + 0 * ldp))) = ((*(alpha1 + 0 * inca + 0 * lda)));
                    ((*(pi1 + 1 + 0 * ldp))) = ((*(alpha1 + 1 * inca + 0 * lda)));
                    ((*(pi1 + 2 + 0 * ldp))) = ((*(alpha1 + 2 * inca + 0 * lda)));
                    ((*(pi1 + 3 + 0 * ldp))) = ((*(alpha1 + 3 * inca + 0 * lda)));
                    ((*(pi1 + 4 + 0 * ldp))) = ((*(alpha1 + 4 * inca + 0 * lda)));
                    ((*(pi1 + 5 + 0 * ldp))) = ((*(alpha1 + 5 * inca + 0 * lda)));
                    ((*(pi1 + 6 + 0 * ldp))) = ((*(alpha1 + 6 * inca + 0 * lda)));
                    ((*(pi1 + 7 + 0 * ldp))) = ((*(alpha1 + 7 * inca + 0 * lda)));

                    ((*(pi1 + 0 + 1 * ldp))) = ((*(alpha1 + 0 * inca + 1 * lda)));
                    ((*(pi1 + 1 + 1 * ldp))) = ((*(alpha1 + 1 * inca + 1 * lda)));
                    ((*(pi1 + 2 + 1 * ldp))) = ((*(alpha1 + 2 * inca + 1 * lda)));
                    ((*(pi1 + 3 + 1 * ldp))) = ((*(alpha1 + 3 * inca + 1 * lda)));
                    ((*(pi1 + 4 + 1 * ldp))) = ((*(alpha1 + 4 * inca + 1 * lda)));
                    ((*(pi1 + 5 + 1 * ldp))) = ((*(alpha1 + 5 * inca + 1 * lda)));
                    ((*(pi1 + 6 + 1 * ldp))) = ((*(alpha1 + 6 * inca + 1 * lda)));
                    ((*(pi1 + 7 + 1 * ldp))) = ((*(alpha1 + 7 * inca + 1 * lda)));

                    alpha1 += 2 * lda;
                    pi1 += 2 * ldp;
                }

                //for (; n_left != 0; --n_left)
                if (n_left == 1)
                {
                    ((*(pi1 + 0))) = ((*(alpha1 + 0 * inca)));
                    ((*(pi1 + 1))) = ((*(alpha1 + 1 * inca)));
                    ((*(pi1 + 2))) = ((*(alpha1 + 2 * inca)));
                    ((*(pi1 + 3))) = ((*(alpha1 + 3 * inca)));
                    ((*(pi1 + 4))) = ((*(alpha1 + 4 * inca)));
                    ((*(pi1 + 5))) = ((*(alpha1 + 5 * inca)));
                    ((*(pi1 + 6))) = ((*(alpha1 + 6 * inca)));
                    ((*(pi1 + 7))) = ((*(alpha1 + 7 * inca)));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
        }
        else
        {
            if (bli_is_conj(conja))
            {
                for (dim_t k = n; k != 0; --k)
                {
                    ((*(pi1 + 0))) = ((*kappa_cast)) * ((*(alpha1 + 0 * inca)));
                    ((*(pi1 + 1))) = ((*kappa_cast)) * ((*(alpha1 + 1 * inca)));
                    ((*(pi1 + 2))) = ((*kappa_cast)) * ((*(alpha1 + 2 * inca)));
                    ((*(pi1 + 3))) = ((*kappa_cast)) * ((*(alpha1 + 3 * inca)));
                    ((*(pi1 + 4))) = ((*kappa_cast)) * ((*(alpha1 + 4 * inca)));
                    ((*(pi1 + 5))) = ((*kappa_cast)) * ((*(alpha1 + 5 * inca)));
                    ((*(pi1 + 6))) = ((*kappa_cast)) * ((*(alpha1 + 6 * inca)));
                    ((*(pi1 + 7))) = ((*kappa_cast)) * ((*(alpha1 + 7 * inca)));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
            else
            {
                for (dim_t k = n; k != 0; --k)
                {
                    ((*(pi1 + 0))) = ((*kappa_cast)) * ((*(alpha1 + 0 * inca)));
                    ((*(pi1 + 1))) = ((*kappa_cast)) * ((*(alpha1 + 1 * inca)));
                    ((*(pi1 + 2))) = ((*kappa_cast)) * ((*(alpha1 + 2 * inca)));
                    ((*(pi1 + 3))) = ((*kappa_cast)) * ((*(alpha1 + 3 * inca)));
                    ((*(pi1 + 4))) = ((*kappa_cast)) * ((*(alpha1 + 4 * inca)));
                    ((*(pi1 + 5))) = ((*kappa_cast)) * ((*(alpha1 + 5 * inca)));
                    ((*(pi1 + 6))) = ((*kappa_cast)) * ((*(alpha1 + 6 * inca)));
                    ((*(pi1 + 7))) = ((*kappa_cast)) * ((*(alpha1 + 7 * inca)));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
        }
    }
    else /* if ( cdim < mnr ) */
    {
        bli_dscal2m_ex
        (
            0,
            BLIS_NONUNIT_DIAG,
            BLIS_DENSE,
            (trans_t)conja,
            cdim,
            n,
            kappa,
            a, inca, lda,
            p, 1, ldp,
            cntx,
            NULL
        );

        /* if ( cdim < mnr ) */
        {
            const dim_t     i = cdim;
            const dim_t     m_edge = 8 - cdim;
            const dim_t     n_edge = n_max;
            double* restrict p_cast = p;
            double* restrict p_edge = p_cast + (i) * 1;

            bli_dset0s_mxn
            (
                m_edge,
                n_edge,
                p_edge, 1, ldp
            );
        }
    }

    if (n < n_max)
    {
        const dim_t     j = n;
        const dim_t     m_edge = 8;
        const dim_t     n_edge = n_max - n;
        double* restrict p_cast = p;
        double* restrict p_edge = p_cast + (j)*ldp;

        bli_dset0s_mxn
        (
            m_edge,
            n_edge,
            p_edge, 1, ldp
        );
    }
}// End of function

// Packing routine for general operations
void bli_dpackm_6xk_gen_zen
(
    conj_t           conja,
    pack_t           schema,
    dim_t            cdim,
    dim_t            n,
    dim_t            n_max,
    void*   restrict kappa,
    void*   restrict a, inc_t inca, inc_t lda,
    void*   restrict p, inc_t ldp,
    cntx_t* restrict cntx
)
{
    double* restrict kappa_cast = kappa;
    double* restrict alpha1 = a;
    double* restrict pi1 = p;

    if (cdim == 6)
    {
        if ((((*kappa_cast)) == (1.0)))
        {
            if (bli_is_conj(conja))
            {
                for (dim_t k = n; k != 0; --k)
                {
                    ((*(pi1 + 0))) = (((*(alpha1 + 0 * inca))));
                    ((*(pi1 + 1))) = (((*(alpha1 + 1 * inca))));
                    ((*(pi1 + 2))) = (((*(alpha1 + 2 * inca))));
                    ((*(pi1 + 3))) = (((*(alpha1 + 3 * inca))));
                    ((*(pi1 + 4))) = (((*(alpha1 + 4 * inca))));
                    ((*(pi1 + 5))) = (((*(alpha1 + 5 * inca))));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
            else
            {
                for (dim_t k = n; k != 0; --k)
                {
                    ((*(pi1 + 0))) = ((*(alpha1 + 0 * inca)));
                    ((*(pi1 + 1))) = ((*(alpha1 + 1 * inca)));
                    ((*(pi1 + 2))) = ((*(alpha1 + 2 * inca)));
                    ((*(pi1 + 3))) = ((*(alpha1 + 3 * inca)));

                    ((*(pi1 + 4))) = ((*(alpha1 + 4 * inca)));
                    ((*(pi1 + 5))) = ((*(alpha1 + 5 * inca)));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
        }
        else
        {
            if (bli_is_conj(conja))
            {
                for (dim_t k = n; k != 0; --k)
                {
                    ((*(pi1 + 0))) = ((*kappa_cast)) * ((*(alpha1 + 0 * inca)));
                    ((*(pi1 + 1))) = ((*kappa_cast)) * ((*(alpha1 + 1 * inca)));
                    ((*(pi1 + 2))) = ((*kappa_cast)) * ((*(alpha1 + 2 * inca)));
                    ((*(pi1 + 3))) = ((*kappa_cast)) * ((*(alpha1 + 3 * inca)));
                    ((*(pi1 + 4))) = ((*kappa_cast)) * ((*(alpha1 + 4 * inca)));
                    ((*(pi1 + 5))) = ((*kappa_cast)) * ((*(alpha1 + 5 * inca)));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
            else
            {
                for (dim_t k = n; k != 0; --k)
                {
                    ((*(pi1 + 0))) = ((*kappa_cast)) * ((*(alpha1 + 0 * inca)));
                    ((*(pi1 + 1))) = ((*kappa_cast)) * ((*(alpha1 + 1 * inca)));
                    ((*(pi1 + 2))) = ((*kappa_cast)) * ((*(alpha1 + 2 * inca)));
                    ((*(pi1 + 3))) = ((*kappa_cast)) * ((*(alpha1 + 3 * inca)));
                    ((*(pi1 + 4))) = ((*kappa_cast)) * ((*(alpha1 + 4 * inca)));
                    ((*(pi1 + 5))) = ((*kappa_cast)) * ((*(alpha1 + 5 * inca)));

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
        }
    }
    else /* if ( cdim < mnr ) */
    {
        bli_dscal2m_ex
        (
            0,
            BLIS_NONUNIT_DIAG,
            BLIS_DENSE,
            (trans_t)conja,
            cdim,
            n,
            kappa,
            a, inca, lda,
            p, 1, ldp,
            cntx,
            NULL
        );

        /* if ( cdim < mnr ) */
        {
            const dim_t     i = cdim;
            const dim_t     m_edge = 6 - cdim;
            const dim_t     n_edge = n_max;
            double* restrict p_cast = p;
            double* restrict p_edge = p_cast + (i) * 1;

            bli_dset0s_mxn
            (
                m_edge,
                n_edge,
                p_edge, 1, ldp
            );
        }
    }

    if (n < n_max)
    {
        const dim_t     j = n;
        const dim_t     m_edge = 6;
        const dim_t     n_edge = n_max - n;
        double* restrict p_cast = p;
        double* restrict p_edge = p_cast + (j)*ldp;

        bli_dset0s_mxn
        (
            m_edge,
            n_edge,
            p_edge, 1, ldp
        );
    }
}// end of function
