/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#if defined __clang__
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define UNROLL_LOOP_FULL() _Pragma("GCC unroll 8")
#else
    #define UNROLL_LOOP_FULL()
#endif

void bli_ddotxf_zen_int_avx512
     (
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a_, inc_t inca, inc_t lda,
       double* restrict x_, inc_t incx,
       double* restrict beta,
       double* restrict y_, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t fuse_fac = 8;
    const dim_t n_elem_per_reg = 8;
    double* a = a_;
    double* x = x_;
    double* y = y_;


    // If the b_n dimension is zero, y is empty and there is no computation.
    if (bli_zero_dim1(b_n))
        return;

    // If the m dimension is zero, or if alpha is zero, the computation
    // simplifies to updating y.
    if (bli_zero_dim1(m) || PASTEMAC(d, eq0)(*alpha))
    {
        bli_dscalv_zen_int_avx512(
            BLIS_NO_CONJUGATE,
            b_n,
            beta,
            y, incy,
            cntx);
        return;
    }

    /*
      If b_n is not equal to the fusing factor, then perform the entire
      operation as dotxv or perform the operation using dotxf kernels with
      lower fuse factor.
    */
    if (b_n != fuse_fac)
    {
        if (b_n >= 4)
        {
            dim_t fuse = 4;

            bli_ddotxf_zen_int_4
            (
              conjat,
              conjx,
              m,
              fuse,
              alpha,
              a, inca, lda,
              x, incx,
              beta,
              y, incy,
              cntx
            );

            // Increment the pointers
            a = a + (fuse)*lda;
            y = y + (fuse)*incy;

            // Decrement to point to the remaining compute left
            b_n -= 4;
        }

        if (b_n >= 2)
        {
            dim_t fuse = 2;

            bli_ddotxf_zen_int_2
            (
              conjat,
              conjx,
              m,
              fuse,
              alpha,
              a, inca, lda,
              x, incx,
              beta,
              y, incy,
              cntx
            );

            // Increment the pointers
            a = a + (fuse)*lda;
            y = y + (fuse)*incy;

            b_n -= 2;
        }

        if (b_n == 1)
        {
            double *a1 = a;
            double *x1 = x;
            double *psi1 = y;

            bli_ddotxv_zen_int(
                conjat,
                conjx,
                m,
                alpha,
                a1, inca,
                x1, incx,
                beta,
                psi1,
                cntx);
        }
        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.
    // However, m may not be a multiple of the number of elements per vector.

    // Going forward, we handle two possible storage formats of A explicitly:
    // (1) A is stored by columns, or (2) A is stored by rows. Either case is
    // further split into two subproblems along the m dimension:
    // (a) a vectorized part, starting at m = 0 and ending at any 0 <= m' <= m.
    // (b) a scalar part, starting at m' and ending at m. If no vectorization
    //     is possible then m' == 0 and thus the scalar part is the entire
    //     problem. If 0 < m', then the a and x pointers and m variable will
    //     be adjusted accordingly for the second subproblem.
    // Note: since parts (b) for both (1) and (2) are so similar, they are
    // factored out into one code block after the following conditional, which
    // distinguishes between (1) and (2).


    __m512d yv;
    __m512d rho[8];
    double *restrict av[8];
    __m512d xv;
    rho[0] = _mm512_setzero_pd();

    if ( inca == 1 && incx == 1 )
    {
        __m512d a_vec[8];
        dim_t m_iter = m / ( n_elem_per_reg );

        UNROLL_LOOP_FULL()
        for (dim_t ii = 0; ii < 8; ++ii)
        {
            rho[ii] = _mm512_setzero_pd();
            av[ii] = a + ii * lda;
        }

        for(dim_t i = 0; i < m_iter; ++i)
        {
            xv = _mm512_loadu_pd( x );

            UNROLL_LOOP_FULL()
            for (dim_t ii = 0; ii < 8; ++ii)
            {
                a_vec[ii] = _mm512_loadu_pd( av[ii] );
                av[ii] += n_elem_per_reg;
			    rho[ii] = _mm512_fmadd_pd(a_vec[ii], xv, rho[ii]);
            }
            x += n_elem_per_reg;
        }
        UNROLL_LOOP_FULL()
        for (dim_t ii = 0; ii < 8; ++ii)
        {
            rho[0][ii] = _mm512_reduce_add_pd(rho[ii]);
        }
        m -= n_elem_per_reg * m_iter;
        a += n_elem_per_reg * m_iter;
    }

    // Initialize pointers for x and the b_n columns of A (rows of A^T).
    double *restrict x0 = x;

    if( m > 0)
    {
        UNROLL_LOOP_FULL()
        for (dim_t ii = 0; ii < 8; ++ii)
        {
            av[ii] = a + ii * lda;
        }
    }

	// If there are leftover iterations, perform them with scalar code.
	for (dim_t i = 0; i < m; ++i)
	{
		const double x0c = *x0;

		rho[0][0] += (*av[0]) * x0c;
		rho[0][1] += (*av[1]) * x0c;
		rho[0][2] += (*av[2]) * x0c;
		rho[0][3] += (*av[3]) * x0c;
		rho[0][4] += (*av[4]) * x0c;
		rho[0][5] += (*av[5]) * x0c;
		rho[0][6] += (*av[6]) * x0c;
		rho[0][7] += (*av[7]) * x0c;

		x0 += incx;
		av[0] += inca;
		av[1] += inca;
		av[2] += inca;
		av[3] += inca;
		av[4] += inca;
		av[5] += inca;
		av[6] += inca;
		av[7] += inca;
	}

    // Broadcast the alpha scalar.
    __m512d alphav = _mm512_set1_pd( *alpha );

    // We know at this point that alpha is nonzero; however, beta may still
    // be zero. If beta is indeed zero, we must overwrite y rather than scale
    // by beta (in case y contains NaN or Inf).
    if (PASTEMAC(d, eq0)(*beta))
        yv = _mm512_mul_pd(alphav, rho[0]);
    else
    {
        // Broadcast the beta scalar
        __m512d betav = _mm512_set1_pd(*beta);

        // Load y.
        if( incy == 1 )
        {
            yv = _mm512_loadu_pd( y );
        }
        else
        {
            UNROLL_LOOP_FULL()
            for (dim_t ii = 0; ii < 8; ++ii)
            {
                yv[ii] = *(y + ii * incy);
            }
        }
        // Apply beta to y and alpha to the accumulated dot product in rho:
        //   y := beta * y + alpha * rho
        yv = _mm512_mul_pd(betav, yv);
        yv = _mm512_fmadd_pd(alphav, rho[0], yv);
    }

    // Store the output.
    if (incy == 1)
    {
        _mm512_storeu_pd(y, yv);
    }
    else
    {
        UNROLL_LOOP_FULL()
        for (dim_t ii = 0; ii < 8; ++ii)
        {
            *(y + ii * incy) = yv[ii];
        }
    }

}