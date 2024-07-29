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
    if (b_n < fuse_fac)
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
    else if (b_n > fuse_fac)
    {
        for (dim_t i = 0; i < b_n; ++i)
		{
			double *a1 = a + (0) * inca + (i)*lda;
			double *x1 = x + (0) * incx;
			double *psi1 = y + (i)*incy;

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



/* Union data structure to access AVX-512 registers
*  One 512-bit AVX register holds 8 DP elements. */
typedef union
{
  __m512d v;
  double  d[8] __attribute__((aligned(64)));
} v8df_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
  __m256d v;
  double  d[4] __attribute__((aligned(64)));
} v4df_t;

/* Union data structure to access AVX registers
*  One 128-bit AVX register holds 2 DP elements. */
typedef union
{
  __m128d v;
  double  d[2] __attribute__((aligned(64)));
} v2df_t;

void bli_zdotxf_zen_int_2_avx512
  (
     conj_t conjat,
     conj_t conjx,
     dim_t m,
     dim_t b_n,
     dcomplex* restrict alpha,
     dcomplex* restrict a, inc_t inca, inc_t lda,
     dcomplex* restrict x, inc_t incx,
     dcomplex* restrict beta,
     dcomplex* restrict y, inc_t incy,
     cntx_t* restrict cntx
   )
{
  /* If the vectors are empty or if alpha is zero, return early */
  if ( bli_zero_dim1( m ) || PASTEMAC(z,eq0)( *alpha ) )
  {
    bli_zscalv_zen_int
    (
      BLIS_NO_CONJUGATE,
      b_n,
      beta,
      y, incy,
      cntx
    );

    return;
  }

  // If b_n is not equal to the fusing factor(2), then perform the entire
  // operation with a dotxv kernel call.
  if ( b_n != 2 )
  {
    dcomplex* restrict a1   = a;
    dcomplex* restrict x1   = x;
    dcomplex* restrict psi1 = y;

    bli_zdotxv_zen_int_avx512
    (
      conjat,
      conjx,
      m,
      alpha,
      a1, inca,
      x1, incx,
      beta,
      psi1,
      cntx
    );

    return;
  }

  // Declaring and initializing the iterator and pointers
  dim_t i = 0;

  double *restrict av[2];
  double *restrict x_temp = (double *)(x);

  av[0] = (double *)(a + 0 * lda);
  av[1] = (double *)(a + 1 * lda);

  // Local memory to store the dot-products
  dcomplex res[2] __attribute__((aligned(64)));
  res[0] = res[1] = (*bli_z0);

  // Performing XOR of conjx and conjat.
  // conj_op is set if either X or A has conjugate(not both)
  conj_t conj_op = conjx ^ conjat;

  // Computation for unit-strided case
  if (incx == 1 && inca == 1)
  {
    // Declaring 4 registers, to store partial sums over multiple loads
    // Further declaring 2 registers for load, 2 for broadcast(real and imag)
    v8df_t rhov[4], a_vec[2], xv[2];

    // Clearing the partial-sum accumulators
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();

    for (; (i + 3) < m; i += 4)
    {
      // Load 4 elements from X
      xv[0].v = _mm512_loadu_pd(x_temp);

      // Permute to duplicate the imag part for every element
      // xv[1].v = I0 I0 I1 I1 ...
      xv[1].v = _mm512_permute_pd(xv[0].v, 0xFF);

      // Permute to duplicate the real part for every element
      // xv[0].v = R0 R0 R1 R1 ...
      xv[0].v = _mm512_permute_pd(xv[0].v, 0x00);

      // Load 4 elements from first 4 columns of A
      a_vec[0].v = _mm512_loadu_pd(av[0]);
      a_vec[1].v = _mm512_loadu_pd(av[1]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm512_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[2].v = _mm512_fmadd_pd(a_vec[0].v, xv[1].v, rhov[2].v);

      rhov[1].v = _mm512_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[3].v = _mm512_fmadd_pd(a_vec[1].v, xv[1].v, rhov[3].v);

      // Adjust the pointers accordingly
      av[0] += 8;
      av[1] += 8;

      x_temp += 8;
    }
    if (i < m)
    {
      // Setting the mask bit based on remaining elements
      // Since each dcomplex elements corresponds to 2 doubles
      // we need to load and store 2*(m-i) elements.
      __mmask8 m_mask = (1 << 2*(m - i)) - 1;

      // Load remaining elements from X
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      xv[0].v = _mm512_maskz_loadu_pd(m_mask, x_temp);

      // Permute to duplicate the imag part for every element
      // xv[1].v = I0 I0 I1 I1 ...
      xv[1].v = _mm512_permute_pd(xv[0].v, 0xFF);

      // Permute to duplicate the real part for every element
      // xv[0].v = R0 R0 R1 R1 ...
      xv[0].v = _mm512_permute_pd(xv[0].v, 0x00);

      // Load remaining elements from first 4 columns of A
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      a_vec[0].v = _mm512_maskz_loadu_pd(m_mask, av[0]);
      a_vec[1].v = _mm512_maskz_loadu_pd(m_mask, av[1]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm512_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[2].v = _mm512_fmadd_pd(a_vec[0].v, xv[1].v, rhov[2].v);

      rhov[1].v = _mm512_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[3].v = _mm512_fmadd_pd(a_vec[1].v, xv[1].v, rhov[3].v);
    }

    // Permuting for final accumulation of real and imag parts
    rhov[2].v = _mm512_permute_pd(rhov[2].v, 0x55);
    rhov[3].v = _mm512_permute_pd(rhov[3].v, 0x55);

    v8df_t scale_one;
    v4df_t zero_reg;

    zero_reg.v = _mm256_setzero_pd();
    scale_one.v = _mm512_set1_pd(1.0);

    /*
      conj_op maps to the compute as follows :
      A = (a + ib), X = (x + iy)
      -----------------------------------------------------------
      |      A       |      X       |  Real part  |  Imag Part  |
      -----------------------------------------------------------
      | No-Conjugate | No-Conjugate |   ax - by	  |   bx + ay   |
      | No-Conjugate |   Conjugate  |   ax + by   |   bx - ay   |
      |   Conjugate  | No-Conjugate |   ax + by   | -(bx - ay)  |
      |   Conjugate  |   Conjugate  |   ax - by   | -(bx + ay)  |
      -----------------------------------------------------------

      If only X or A has conjugate, fmsubadd is performed.
      Else, fmaddsub is performed.

      In the final reduction step, the imaginary part of every
      partial sum is negated if conjat is conjugate
    */
    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm512_fmaddsub_pd(scale_one.v, rhov[0].v, rhov[2].v);
      rhov[1].v = _mm512_fmaddsub_pd(scale_one.v, rhov[1].v, rhov[3].v);
    }
    else
    {
      rhov[0].v = _mm512_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[2].v);
      rhov[1].v = _mm512_fmsubadd_pd(scale_one.v, rhov[1].v, rhov[3].v);
    }

    // rhov[0 ... 1] will have the element wise product.
    // These have to be added horizontally(reduction) to get the
    // final result for every element in y.
    // If rhov[0]   = R0 I0 R1 I1 R2 I2 R3 I3
    // Then rhov[2] = R1 I1 R0 I0 R3 I2 R2 I2
    rhov[2].v = _mm512_permutex_pd(rhov[0].v, 0x4E);
    rhov[3].v = _mm512_permutex_pd(rhov[1].v, 0x4E);

    // rhov[0] = (R0 + R1) (I0 + I1) (R1 + R0) (I1 + I0)
    //           (R2 + R3) (I2 + I3) (R3 + R2) (I3 + I2)
    rhov[0].v = _mm512_add_pd(rhov[0].v, rhov[2].v);
    rhov[1].v = _mm512_add_pd(rhov[1].v, rhov[3].v);

    // 256-bit registers declared to extract 256-bit lanes
    v4df_t reduce_sum[4];

    // reduce_sum[0] = (R0 + R1) (I0 + I1) (R1 + R0) (I1 + I0)
    reduce_sum[0].v = _mm512_extractf64x4_pd(rhov[0].v, 0x00);
    reduce_sum[1].v = _mm512_extractf64x4_pd(rhov[1].v, 0x00);

    // reduce_sum[2] = (R2 + R3) (I2 + I3) (R3 + R2) (I3 + I2)
    reduce_sum[2].v = _mm512_extractf64x4_pd(rhov[0].v, 0x1);
    reduce_sum[3].v = _mm512_extractf64x4_pd(rhov[1].v, 0x1);

    // reduce_sum[0] = (R0 + R1 + R2 + R3) (I0 + I1 + I2 + I3) ...
    reduce_sum[0].v = _mm256_add_pd(reduce_sum[0].v, reduce_sum[2].v);
    reduce_sum[1].v = _mm256_add_pd(reduce_sum[1].v, reduce_sum[3].v);

    // The next set of shuffles and permutes are performed to store
    // all the dot-products onto one 256-bit register. This is used to
    //	perform aligned stores onto the stack memory.
    reduce_sum[2].v = _mm256_shuffle_pd(reduce_sum[0].v, reduce_sum[1].v, 0xC);

    reduce_sum[3].v = _mm256_permutex_pd(reduce_sum[2].v, 0xD8);

    // Negate the sign bit of imaginary part of dot-products if conjat is conjugate
    if ( bli_is_conj( conjat ) )
    {
      reduce_sum[3].v = _mm256_fmsubadd_pd(zero_reg.v, zero_reg.v, reduce_sum[3].v);
    }

    /*
      Computed dot product result is being stored
      in temp buffer r for further computation.
    */
    _mm256_store_pd((double *)res, reduce_sum[3].v);
  }

  // This section will have the whole of compute when incx != 1 || inca != 1
  else
  {
    // Declaring 128-bit registers, for element by element computation
    v2df_t rhov[4], a_vec[2], xv[2];

    // Clearing the partial-sum accumulators
    rhov[0].v = _mm_setzero_pd();
    rhov[1].v = _mm_setzero_pd();
    rhov[2].v = _mm_setzero_pd();
    rhov[3].v = _mm_setzero_pd();

    for (dim_t i = 0; i < m; i++)
    {
      // Load from X
      xv[0].v = _mm_loadu_pd(x_temp);

      // Permute to duplicate the imag part for every element
      xv[1].v = _mm_permute_pd(xv[0].v, 0b11);

      // Permute to duplicate the real part for every element
      xv[0].v = _mm_permute_pd(xv[0].v, 0b00);

      // Load elements from first 4 columns of A
      a_vec[0].v = _mm_loadu_pd(av[0]);
      a_vec[1].v = _mm_loadu_pd(av[1]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[2].v = _mm_fmadd_pd(a_vec[0].v, xv[1].v, rhov[2].v);

      rhov[1].v = _mm_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[3].v = _mm_fmadd_pd(a_vec[1].v, xv[1].v, rhov[3].v);

      av[0] += 2 * inca;
      av[1] += 2 * inca;

      x_temp += 2 * incx;
    }

    // Permuting to help with final reduction
    rhov[3].v = _mm_permute_pd(rhov[3].v, 0b01);
    rhov[2].v = _mm_permute_pd(rhov[2].v, 0b01);

    v2df_t zero_reg, scale_one;

    zero_reg.v = _mm_setzero_pd();
    scale_one.v = _mm_set1_pd(1.0);

    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm_addsub_pd(rhov[0].v, rhov[2].v);
      rhov[1].v = _mm_addsub_pd(rhov[1].v, rhov[3].v);
    }
    else
    {
      rhov[0].v = _mm_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[2].v);
      rhov[1].v = _mm_fmsubadd_pd(scale_one.v, rhov[1].v, rhov[3].v);
    }
    if( bli_is_conj( conjat ) )
    {
      rhov[0].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[0].v);
      rhov[1].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[1].v);
    }

    // Storing onto static memory, to be used later
    _mm_storeu_pd((double *)res, rhov[0].v);
    _mm_storeu_pd((double *)(res + 1), rhov[1].v);

  }

  // Scaling by alpha
  // Registers to load partial sums, stored in static memory
  v4df_t rhov, temp;

  rhov.v = _mm256_load_pd((double *)res);

  if ( !bli_zeq1( *alpha ) )
  {
    __m256d alphaRv, alphaIv;
    alphaRv = _mm256_set1_pd((*alpha).real);
    alphaIv = _mm256_set1_pd((*alpha).imag);

    temp.v = _mm256_permute_pd(rhov.v, 0x5);

    // Scaling with imag part of alpha
    temp.v = _mm256_mul_pd(temp.v, alphaIv);

    // Scaling with real part of alpha, and addsub
    rhov.v = _mm256_fmaddsub_pd(rhov.v, alphaRv, temp.v);
  }
  // When 'beta' is not zero we need to multiply scale 'y' by 'beta'
  v4df_t yv;

  yv.v = _mm256_setzero_pd();

  if (!PASTEMAC(z, eq0)(*beta))
  {
    __m256d betaRv, betaIv;

    betaRv = _mm256_set1_pd((*beta).real);
    betaIv = _mm256_set1_pd((*beta).imag);

    if (incy == 1)
    {
      yv.v = _mm256_loadu_pd((double *)(y));
    }
    else
    {
      /*
        This can be done using SSE instructions
        but has been kept as scalar code to avoid
        mixing SSE with AVX
      */
      yv.d[0] = (*(y + 0 * incy)).real;
      yv.d[1] = (*(y + 0 * incy)).imag;
      yv.d[2] = (*(y + 1 * incy)).real;
      yv.d[3] = (*(y + 1 * incy)).imag;

    }

    temp.v = _mm256_permute_pd(yv.v, 0x5);

    // Scaling with imag part of alpha
    temp.v = _mm256_mul_pd(temp.v, betaIv);

    // Scaling with real part of alpha, and addsub
    yv.v = _mm256_fmaddsub_pd(yv.v, betaRv, temp.v);
  }

  // Adding alpha*A*x to beta*Y
  yv.v = _mm256_add_pd(yv.v, rhov.v);

  if (incy == 1)
  {
    _mm256_storeu_pd((double *)y, yv.v);
  }
  else
  {
    (*(y + 0 * incy)).real = yv.d[0];
    (*(y + 0 * incy)).imag = yv.d[1];
    (*(y + 1 * incy)).real = yv.d[2];
    (*(y + 1 * incy)).imag = yv.d[3];

  }

}

void bli_zdotxf_zen_int_4_avx512
  (
     conj_t conjat,
     conj_t conjx,
     dim_t m,
     dim_t b_n,
     dcomplex* restrict alpha,
     dcomplex* restrict a, inc_t inca, inc_t lda,
     dcomplex* restrict x, inc_t incx,
     dcomplex* restrict beta,
     dcomplex* restrict y, inc_t incy,
     cntx_t* restrict cntx
   )
{
  /* If the vectors are empty or if alpha is zero, return early */
  if ( bli_zero_dim1( m ) || PASTEMAC(z,eq0)( *alpha ) )
  {
    bli_zscalv_zen_int
    (
      BLIS_NO_CONJUGATE,
      b_n,
      beta,
      y, incy,
      cntx
    );

    return;
  }

  // If b_n is not equal to the fusing factor(4), then perform the entire
  // operation as a sequence of fringe dotxf kernel(2) and dotxv
  // kernel as per the requirement.
  if ( b_n != 4 )
  {
    dcomplex* restrict a1   = a;
    dcomplex* restrict x1   = x;
    dcomplex* restrict psi1 = y;

    if( b_n >= 2 )
    {
      bli_zdotxf_zen_int_2_avx512
      (
          conjat,
          conjx,
          m,
          (dim_t)2,
          alpha,
          a1, inca, lda,
          x1, incx,
          beta,
          psi1,   incy,
          NULL
      );

      a1 += 2*lda;
      psi1 += 2*incy;

      b_n -= 2;
    }

    if( b_n == 1 )
    {
      bli_zdotxv_zen_int_avx512
      (
        conjat,
        conjx,
        m,
        alpha,
        a1, inca,
        x1, incx,
        beta,
        psi1,
        cntx
      );
    }

    return;
  }

  // Declaring and initializing the iterator and pointers
  dim_t i = 0;

  double *restrict av[4];
  double *restrict x_temp = (double *)(x);

  av[0] = (double *)(a + 0 * lda);
  av[1] = (double *)(a + 1 * lda);
  av[2] = (double *)(a + 2 * lda);
  av[3] = (double *)(a + 3 * lda);

  // Local memory to store the dot-products
  dcomplex res[4] __attribute__((aligned(64)));
  res[0] = res[1] = res[2] = res[3] = (*bli_z0);

  // Performing XOR of conjx and conjat.
  // conj_op is set if either X or A has conjugate(not both)
  conj_t conj_op = conjx ^ conjat;

  // Computation for unit-strided case
  if (incx == 1 && inca == 1)
  {
    // Declaring 8 registers, to store partial sums over multiple loads
    // Further declaring 4 registers for load, 2 for broadcast(real and imag)
    v8df_t rhov[8], a_vec[4], xv[2];

    // Clearing the partial-sum accumulators
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();
    rhov[4].v = _mm512_setzero_pd();
    rhov[5].v = _mm512_setzero_pd();
    rhov[6].v = _mm512_setzero_pd();
    rhov[7].v = _mm512_setzero_pd();

    for (; (i + 3) < m; i += 4)
    {
      // Load 4 elements from X
      xv[0].v = _mm512_loadu_pd(x_temp);

      // Permute to duplicate the imag part for every element
      // xv[1].v = I0 I0 I1 I1 ...
      xv[1].v = _mm512_permute_pd(xv[0].v, 0xFF);

      // Permute to duplicate the real part for every element
      // xv[0].v = R0 R0 R1 R1 ...
      xv[0].v = _mm512_permute_pd(xv[0].v, 0x00);

      // Load 4 elements from first 4 columns of A
      a_vec[0].v = _mm512_loadu_pd(av[0]);
      a_vec[1].v = _mm512_loadu_pd(av[1]);
      a_vec[2].v = _mm512_loadu_pd(av[2]);
      a_vec[3].v = _mm512_loadu_pd(av[3]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm512_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[4].v = _mm512_fmadd_pd(a_vec[0].v, xv[1].v, rhov[4].v);

      rhov[1].v = _mm512_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[5].v = _mm512_fmadd_pd(a_vec[1].v, xv[1].v, rhov[5].v);

      rhov[2].v = _mm512_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
      rhov[6].v = _mm512_fmadd_pd(a_vec[2].v, xv[1].v, rhov[6].v);

      rhov[3].v = _mm512_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
      rhov[7].v = _mm512_fmadd_pd(a_vec[3].v, xv[1].v, rhov[7].v);

      // Adjust the pointers accordingly
      av[0] += 8;
      av[1] += 8;
      av[2] += 8;
      av[3] += 8;

      x_temp += 8;
    }
    if (i < m)
    {
      // Setting the mask bit based on remaining elements
      // Since each dcomplex elements corresponds to 2 doubles
      // we need to load and store 2*(m-i) elements.
      __mmask8 m_mask = (1 << 2*(m - i)) - 1;

      // Load remaining elements from X
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      xv[0].v = _mm512_maskz_loadu_pd(m_mask, x_temp);

      // Permute to duplicate the imag part for every element
      // xv[1].v = I0 I0 I1 I1 ...
      xv[1].v = _mm512_permute_pd(xv[0].v, 0xFF);

      // Permute to duplicate the real part for every element
      // xv[0].v = R0 R0 R1 R1 ...
      xv[0].v = _mm512_permute_pd(xv[0].v, 0x00);

      // Load remaining elements from first 4 columns of A
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      a_vec[0].v = _mm512_maskz_loadu_pd(m_mask, av[0]);
      a_vec[1].v = _mm512_maskz_loadu_pd(m_mask, av[1]);
      a_vec[2].v = _mm512_maskz_loadu_pd(m_mask, av[2]);
      a_vec[3].v = _mm512_maskz_loadu_pd(m_mask, av[3]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm512_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[4].v = _mm512_fmadd_pd(a_vec[0].v, xv[1].v, rhov[4].v);

      rhov[1].v = _mm512_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[5].v = _mm512_fmadd_pd(a_vec[1].v, xv[1].v, rhov[5].v);

      rhov[2].v = _mm512_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
      rhov[6].v = _mm512_fmadd_pd(a_vec[2].v, xv[1].v, rhov[6].v);

      rhov[3].v = _mm512_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
      rhov[7].v = _mm512_fmadd_pd(a_vec[3].v, xv[1].v, rhov[7].v);
    }

    // Permuting for final accumulation of real and imag parts
    rhov[4].v = _mm512_permute_pd(rhov[4].v, 0x55);
    rhov[5].v = _mm512_permute_pd(rhov[5].v, 0x55);
    rhov[6].v = _mm512_permute_pd(rhov[6].v, 0x55);
    rhov[7].v = _mm512_permute_pd(rhov[7].v, 0x55);

    // Setting 2 registers to 0 and 1
    v8df_t zero_reg, scale_one;

    zero_reg.v = _mm512_setzero_pd();
    scale_one.v = _mm512_set1_pd(1.0);

    /*
      conj_op maps to the compute as follows :
      A = (a + ib), X = (x + iy)
      -----------------------------------------------------------
      |      A       |      X       |  Real part  |  Imag Part  |
      -----------------------------------------------------------
      | No-Conjugate | No-Conjugate |   ax - by	  |   bx + ay   |
      | No-Conjugate |   Conjugate  |   ax + by   |   bx - ay   |
      |   Conjugate  | No-Conjugate |   ax + by   | -(bx - ay)  |
      |   Conjugate  |   Conjugate  |   ax - by   | -(bx + ay)  |
      -----------------------------------------------------------

      If only X or A has conjugate, fmsubadd is performed.
      Else, fmaddsub is performed.

      In the final reduction step, the imaginary part of every
      partial sum is negated if conjat is conjugate
    */

    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm512_fmaddsub_pd(scale_one.v, rhov[0].v, rhov[4].v);
      rhov[1].v = _mm512_fmaddsub_pd(scale_one.v, rhov[1].v, rhov[5].v);
      rhov[2].v = _mm512_fmaddsub_pd(scale_one.v, rhov[2].v, rhov[6].v);
      rhov[3].v = _mm512_fmaddsub_pd(scale_one.v, rhov[3].v, rhov[7].v);
    }
    else
    {
      rhov[0].v = _mm512_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[4].v);
      rhov[1].v = _mm512_fmsubadd_pd(scale_one.v, rhov[1].v, rhov[5].v);
      rhov[2].v = _mm512_fmsubadd_pd(scale_one.v, rhov[2].v, rhov[6].v);
      rhov[3].v = _mm512_fmsubadd_pd(scale_one.v, rhov[3].v, rhov[7].v);
    }

    // rhov[0 ... 3] will have the element wise product.
    // These have to be added horizontally(reduction) to get the
    // final result for every element in y.
    // If rhov[0]   = R0 I0 R1 I1 R2 I2 R3 I3
    // Then rhov[4] = R1 I1 R0 I0 R3 I2 R2 I2
    rhov[4].v = _mm512_permutex_pd(rhov[0].v, 0x4E);
    rhov[5].v = _mm512_permutex_pd(rhov[1].v, 0x4E);
    rhov[6].v = _mm512_permutex_pd(rhov[2].v, 0x4E);
    rhov[7].v = _mm512_permutex_pd(rhov[3].v, 0x4E);

    // rhov[0] = (R0 + R1) (I0 + I1) (R1 + R0) (I1 + I0)
    //           (R2 + R3) (I2 + I3) (R3 + R2) (I3 + I2)
    rhov[0].v = _mm512_add_pd(rhov[0].v, rhov[4].v);
    rhov[1].v = _mm512_add_pd(rhov[1].v, rhov[5].v);
    rhov[2].v = _mm512_add_pd(rhov[2].v, rhov[6].v);
    rhov[3].v = _mm512_add_pd(rhov[3].v, rhov[7].v);

    // 256-bit registers declared to extract 256-bit lanes
    v4df_t reduce_sum[8];

    // reduce_sum[0] = (R0 + R1) (I0 + I1) (R1 + R0) (I1 + I0)
    reduce_sum[0].v = _mm512_extractf64x4_pd(rhov[0].v, 0x00);
    reduce_sum[1].v = _mm512_extractf64x4_pd(rhov[1].v, 0x00);
    reduce_sum[2].v = _mm512_extractf64x4_pd(rhov[2].v, 0x00);
    reduce_sum[3].v = _mm512_extractf64x4_pd(rhov[3].v, 0x00);

    // reduce_sum[4] = (R2 + R3) (I2 + I3) (R3 + R2) (I3 + I2)
    reduce_sum[4].v = _mm512_extractf64x4_pd(rhov[0].v, 0x1);
    reduce_sum[5].v = _mm512_extractf64x4_pd(rhov[1].v, 0x1);
    reduce_sum[6].v = _mm512_extractf64x4_pd(rhov[2].v, 0x1);
    reduce_sum[7].v = _mm512_extractf64x4_pd(rhov[3].v, 0x1);

    // reduce_sum[0] = (R0 + R1 + R2 + R3) (I0 + I1 + I2 + I3) ...
    reduce_sum[0].v = _mm256_add_pd(reduce_sum[0].v, reduce_sum[4].v);
    reduce_sum[1].v = _mm256_add_pd(reduce_sum[1].v, reduce_sum[5].v);
    reduce_sum[2].v = _mm256_add_pd(reduce_sum[2].v, reduce_sum[6].v);
    reduce_sum[3].v = _mm256_add_pd(reduce_sum[3].v, reduce_sum[7].v);

    // The next set of shuffles, permutes and inserts are performed to store
    // all the dot-products onto one 512-bit register. This is used to perform
    // aligned stores onto the stack memory.
    reduce_sum[4].v = _mm256_shuffle_pd(reduce_sum[0].v, reduce_sum[1].v, 0xC);
    reduce_sum[5].v = _mm256_shuffle_pd(reduce_sum[2].v, reduce_sum[3].v, 0xC);

    reduce_sum[6].v = _mm256_permutex_pd(reduce_sum[4].v, 0xD8);
    reduce_sum[7].v = _mm256_permutex_pd(reduce_sum[5].v, 0xD8);

    rhov[0].v = _mm512_insertf64x4(rhov[0].v, reduce_sum[6].v, 0x00);
    rhov[0].v = _mm512_insertf64x4(rhov[0].v, reduce_sum[7].v, 0x01);

    // Negate the sign bit of imaginary part of dot-products if conjat is conjugate
    if ( bli_is_conj( conjat ) )
    {
      rhov[0].v = _mm512_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[0].v);
    }

    /*
      Computed dot product result is being stored
      in temp buffer r for further computation.
    */
    _mm512_store_pd((double *)res, rhov[0].v);
  }

  // This section will have the whole of compute when incx != 1 || inca != 1
  else
  {
    // Declaring 128-bit registers, for element by element computation
    v2df_t rhov[8], a_vec[4], xv[2];

    // Clearing the partial-sum accumulators
    rhov[0].v = _mm_setzero_pd();
    rhov[1].v = _mm_setzero_pd();
    rhov[2].v = _mm_setzero_pd();
    rhov[3].v = _mm_setzero_pd();
    rhov[4].v = _mm_setzero_pd();
    rhov[5].v = _mm_setzero_pd();
    rhov[6].v = _mm_setzero_pd();
    rhov[7].v = _mm_setzero_pd();

    for (dim_t i = 0; i < m; i++)
    {
      // Load from X
      xv[0].v = _mm_loadu_pd(x_temp);

      // Permute to duplicate the imag part for every element
      xv[1].v = _mm_permute_pd(xv[0].v, 0b11);

      // Permute to duplicate the real part for every element
      xv[0].v = _mm_permute_pd(xv[0].v, 0b00);

      // Load elements from first 4 columns of A
      a_vec[0].v = _mm_loadu_pd(av[0]);
      a_vec[1].v = _mm_loadu_pd(av[1]);
      a_vec[2].v = _mm_loadu_pd(av[2]);
      a_vec[3].v = _mm_loadu_pd(av[3]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[4].v = _mm_fmadd_pd(a_vec[0].v, xv[1].v, rhov[4].v);

      rhov[1].v = _mm_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[5].v = _mm_fmadd_pd(a_vec[1].v, xv[1].v, rhov[5].v);

      rhov[2].v = _mm_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
      rhov[6].v = _mm_fmadd_pd(a_vec[2].v, xv[1].v, rhov[6].v);

      rhov[3].v = _mm_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
      rhov[7].v = _mm_fmadd_pd(a_vec[3].v, xv[1].v, rhov[7].v);

      av[0] += 2 * inca;
      av[1] += 2 * inca;
      av[2] += 2 * inca;
      av[3] += 2 * inca;

      x_temp += 2 * incx;
    }

    // Permuting to help with final reduction
    rhov[4].v = _mm_permute_pd(rhov[4].v, 0b01);
    rhov[5].v = _mm_permute_pd(rhov[5].v, 0b01);
    rhov[6].v = _mm_permute_pd(rhov[6].v, 0b01);
    rhov[7].v = _mm_permute_pd(rhov[7].v, 0b01);

    v2df_t zero_reg, scale_one;

    zero_reg.v = _mm_setzero_pd();
    scale_one.v = _mm_set1_pd(1.0);

    // Reduction based on conj_op
    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm_addsub_pd(rhov[0].v, rhov[4].v);
      rhov[1].v = _mm_addsub_pd(rhov[1].v, rhov[5].v);
      rhov[2].v = _mm_addsub_pd(rhov[2].v, rhov[6].v);
      rhov[3].v = _mm_addsub_pd(rhov[3].v, rhov[7].v);
    }
    else
    {
      rhov[0].v = _mm_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[4].v);
      rhov[1].v = _mm_fmsubadd_pd(scale_one.v, rhov[1].v, rhov[5].v);
      rhov[2].v = _mm_fmsubadd_pd(scale_one.v, rhov[2].v, rhov[6].v);
      rhov[3].v = _mm_fmsubadd_pd(scale_one.v, rhov[3].v, rhov[7].v);
    }
    if( bli_is_conj( conjat ) )
    {
      rhov[0].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[0].v);
      rhov[1].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[1].v);
      rhov[2].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[2].v);
      rhov[3].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[3].v);
    }

    // Storing onto stack memory
    _mm_storeu_pd((double *)res, rhov[0].v);
    _mm_storeu_pd((double *)(res + 1), rhov[1].v);
    _mm_storeu_pd((double *)(res + 2), rhov[2].v);
    _mm_storeu_pd((double *)(res + 3), rhov[3].v);

  }

  // Scaling by alpha
  // Registers to load partial sums, stored in static memory
  v8df_t rhov, temp;

  rhov.v = _mm512_loadu_pd((double *)res);

  if ( !bli_zeq1( *alpha ) )
  {
    __m512d alphaRv, alphaIv;
    alphaRv = _mm512_set1_pd((*alpha).real);
    alphaIv = _mm512_set1_pd((*alpha).imag);

    temp.v = _mm512_permute_pd(rhov.v, 0x55);

    // Scaling with imag part of alpha
    temp.v = _mm512_mul_pd(temp.v, alphaIv);

    // Scaling with real part of alpha, and addsub
    rhov.v = _mm512_fmaddsub_pd(rhov.v, alphaRv, temp.v);
  }
  // When 'beta' is not zero we need to multiply scale 'y' by 'beta'
  v8df_t yv;

  yv.v = _mm512_setzero_pd();

  if (!PASTEMAC(z, eq0)(*beta))
  {
    __m512d betaRv, betaIv;

    betaRv = _mm512_set1_pd((*beta).real);
    betaIv = _mm512_set1_pd((*beta).imag);

    if (incy == 1)
    {
      yv.v = _mm512_loadu_pd((double *)(y));
    }
    else
    {
      /*
        This can be done using SSE instructions
        but has been kept as scalar code to avoid
        mixing SSE with AVX
      */
      yv.d[0] = (*(y + 0 * incy)).real;
      yv.d[1] = (*(y + 0 * incy)).imag;
      yv.d[2] = (*(y + 1 * incy)).real;
      yv.d[3] = (*(y + 1 * incy)).imag;
      yv.d[4] = (*(y + 2 * incy)).real;
      yv.d[5] = (*(y + 2 * incy)).imag;
      yv.d[6] = (*(y + 3 * incy)).real;
      yv.d[7] = (*(y + 3 * incy)).imag;

    }

    temp.v = _mm512_permute_pd(yv.v, 0x55);

    // Scaling with imag part of alpha
    temp.v = _mm512_mul_pd(temp.v, betaIv);

    // Scaling with real part of alpha, and addsub
    yv.v = _mm512_fmaddsub_pd(yv.v, betaRv, temp.v);
  }

  // Adding alpha*A*x to beta*Y
  yv.v = _mm512_add_pd(yv.v, rhov.v);

  if (incy == 1)
  {
    _mm512_storeu_pd((double *)y, yv.v);
  }
  else
  {
    (*(y + 0 * incy)).real = yv.d[0];
    (*(y + 0 * incy)).imag = yv.d[1];
    (*(y + 1 * incy)).real = yv.d[2];
    (*(y + 1 * incy)).imag = yv.d[3];

    (*(y + 2 * incy)).real = yv.d[4];
    (*(y + 2 * incy)).imag = yv.d[5];
    (*(y + 3 * incy)).real = yv.d[6];
    (*(y + 3 * incy)).imag = yv.d[7];

  }

}

void bli_zdotxf_zen_int_8_avx512
  (
     conj_t conjat,
     conj_t conjx,
     dim_t m,
     dim_t b_n,
     dcomplex* restrict alpha,
     dcomplex* restrict a, inc_t inca, inc_t lda,
     dcomplex* restrict x, inc_t incx,
     dcomplex* restrict beta,
     dcomplex* restrict y, inc_t incy,
     cntx_t* restrict cntx
   )
{
  /* If vectors are empty or if alpha is zero, scale y by beta and return */
  if ( bli_zero_dim1( m ) || PASTEMAC(z,eq0)( *alpha ) )
  {
    bli_zscalv_zen_int
    (
      BLIS_NO_CONJUGATE,
      b_n,
      beta,
      y, incy,
      cntx
    );

    return;
  }

  // If b_n is not equal to the fusing factor(8), then perform the entire
  // operation as a sequence of fringe dotxf kernels(4 and 2) and dotxv
  // kernel as per the requirement.
  if ( b_n != 8 )
  {
    dcomplex* restrict a1   = a;
    dcomplex* restrict x1   = x;
    dcomplex* restrict psi1 = y;

    if( b_n >= 4 )
    {
      bli_zdotxf_zen_int_4_avx512
      (
          conjat,
          conjx,
          m,
          (dim_t)4,
          alpha,
          a1, inca, lda,
          x1, incx,
          beta,
          psi1,   incy,
          NULL
      );

      a1 += 4*lda;
      psi1 += 4*incy;

      b_n -= 4;
    }

    if( b_n >= 2 )
    {
      bli_zdotxf_zen_int_2_avx512
      (
          conjat,
          conjx,
          m,
          (dim_t)2,
          alpha,
          a1, inca, lda,
          x1, incx,
          beta,
          psi1,   incy,
          NULL
      );

      a1 += 2*lda;
      psi1 += 2*incy;

      b_n -= 2;
    }

    if( b_n == 1 )
    {
      bli_zdotxv_zen_int_avx512
      (
        conjat,
        conjx,
        m,
        alpha,
        a1, inca,
        x1, incx,
        beta,
        psi1,
        cntx
      );
    }

    return;
  }

  // Declaring and initializing the iterator and pointers
  dim_t i = 0;

  double *restrict av[8];
  double *restrict x_temp = (double *)(x);

  av[0] = (double *)(a + 0 * lda);
  av[1] = (double *)(a + 1 * lda);
  av[2] = (double *)(a + 2 * lda);
  av[3] = (double *)(a + 3 * lda);
  av[4] = (double *)(a + 4 * lda);
  av[5] = (double *)(a + 5 * lda);
  av[6] = (double *)(a + 6 * lda);
  av[7] = (double *)(a + 7 * lda);

  // Local memory to store the dot-products
  dcomplex res[8] __attribute__((aligned(64)));
  res[0] = res[1] = res[2] = res[3] = res[4] = res[5] = res[6] = res[7] = (*bli_z0);

  // Performing XOR of conjx and conjat.
  // conj_op is set if either X or A has conjugate(not both)
  conj_t conj_op = conjx ^ conjat;

  // Computation for unit-strided case
  if (incx == 1 && inca == 1)
  {
    // Declaring 16 registers, to store partial sums over multiple loads
    // Further declaring 8 registers for load, 2 for broadcast(real and imag)
    v8df_t rhov[16], a_vec[8], xv[2];

    // Clearing the partial-sum accumulators
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();
    rhov[4].v = _mm512_setzero_pd();
    rhov[5].v = _mm512_setzero_pd();
    rhov[6].v = _mm512_setzero_pd();
    rhov[7].v = _mm512_setzero_pd();
    rhov[8].v = _mm512_setzero_pd();
    rhov[9].v = _mm512_setzero_pd();
    rhov[10].v = _mm512_setzero_pd();
    rhov[11].v = _mm512_setzero_pd();
    rhov[12].v = _mm512_setzero_pd();
    rhov[13].v = _mm512_setzero_pd();
    rhov[14].v = _mm512_setzero_pd();
    rhov[15].v = _mm512_setzero_pd();

    for (; (i + 3) < m; i += 4)
    {
      // Load 4 elements from X
      xv[0].v = _mm512_loadu_pd(x_temp);

      // Permute to duplicate the imag part for every element
      // xv[1].v = I0 I0 I1 I1 ...
      xv[1].v = _mm512_permute_pd(xv[0].v, 0xFF);

      // Permute to duplicate the real part for every element
      // xv[0].v = R0 R0 R1 R1 ...
      xv[0].v = _mm512_permute_pd(xv[0].v, 0x00);

      // Load 4 elements from first 4 columns of A
      a_vec[0].v = _mm512_loadu_pd(av[0]);
      a_vec[1].v = _mm512_loadu_pd(av[1]);
      a_vec[2].v = _mm512_loadu_pd(av[2]);
      a_vec[3].v = _mm512_loadu_pd(av[3]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm512_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[8].v = _mm512_fmadd_pd(a_vec[0].v, xv[1].v, rhov[8].v);

      rhov[1].v = _mm512_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[9].v = _mm512_fmadd_pd(a_vec[1].v, xv[1].v, rhov[9].v);

      rhov[2].v = _mm512_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
      rhov[10].v = _mm512_fmadd_pd(a_vec[2].v, xv[1].v, rhov[10].v);

      rhov[3].v = _mm512_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
      rhov[11].v = _mm512_fmadd_pd(a_vec[3].v, xv[1].v, rhov[11].v);

      // Load 4 elements from next 4 columns of A
      a_vec[4].v = _mm512_loadu_pd(av[4]);
      a_vec[5].v = _mm512_loadu_pd(av[5]);
      a_vec[6].v = _mm512_loadu_pd(av[6]);
      a_vec[7].v = _mm512_loadu_pd(av[7]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[4].v = _mm512_fmadd_pd(a_vec[4].v, xv[0].v, rhov[4].v);
      rhov[12].v = _mm512_fmadd_pd(a_vec[4].v, xv[1].v, rhov[12].v);

      rhov[5].v = _mm512_fmadd_pd(a_vec[5].v, xv[0].v, rhov[5].v);
      rhov[13].v = _mm512_fmadd_pd(a_vec[5].v, xv[1].v, rhov[13].v);

      rhov[6].v = _mm512_fmadd_pd(a_vec[6].v, xv[0].v, rhov[6].v);
      rhov[14].v = _mm512_fmadd_pd(a_vec[6].v, xv[1].v, rhov[14].v);

      rhov[7].v = _mm512_fmadd_pd(a_vec[7].v, xv[0].v, rhov[7].v);
      rhov[15].v = _mm512_fmadd_pd(a_vec[7].v, xv[1].v, rhov[15].v);

      // Adjust the pointers accordingly
      av[0] += 8;
      av[1] += 8;
      av[2] += 8;
      av[3] += 8;
      av[4] += 8;
      av[5] += 8;
      av[6] += 8;
      av[7] += 8;

      x_temp += 8;
    }
    if (i < m)
    {
      // Setting the mask bit based on remaining elements
      // Since each dcomplex elements corresponds to 2 doubles
      // we need to load and store 2*(m-i) elements.
      __mmask8 m_mask = (1 << 2*(m - i)) - 1;

      // Load remaining elements from X
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      xv[0].v = _mm512_maskz_loadu_pd(m_mask, x_temp);

      // Permute to duplicate the imag part for every element
      // xv[1].v = I0 I0 I1 I1 ...
      xv[1].v = _mm512_permute_pd(xv[0].v, 0xFF);

      // Permute to duplicate the real part for every element
      // xv[0].v = R0 R0 R1 R1 ...
      xv[0].v = _mm512_permute_pd(xv[0].v, 0x00);

      // Load remaining elements from first 4 columns of A
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      a_vec[0].v = _mm512_maskz_loadu_pd(m_mask, av[0]);
      a_vec[1].v = _mm512_maskz_loadu_pd(m_mask, av[1]);
      a_vec[2].v = _mm512_maskz_loadu_pd(m_mask, av[2]);
      a_vec[3].v = _mm512_maskz_loadu_pd(m_mask, av[3]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm512_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[8].v = _mm512_fmadd_pd(a_vec[0].v, xv[1].v, rhov[8].v);

      rhov[1].v = _mm512_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[9].v = _mm512_fmadd_pd(a_vec[1].v, xv[1].v, rhov[9].v);

      rhov[2].v = _mm512_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
      rhov[10].v = _mm512_fmadd_pd(a_vec[2].v, xv[1].v, rhov[10].v);

      rhov[3].v = _mm512_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
      rhov[11].v = _mm512_fmadd_pd(a_vec[3].v, xv[1].v, rhov[11].v);

      // Load remaining elements from next 4 columns of A
      // Maskz_load is used to ensure the unloaded elements are 0
      // Else, it affects the accumulation and final reduction
      a_vec[4].v = _mm512_maskz_loadu_pd(m_mask, av[4]);
      a_vec[5].v = _mm512_maskz_loadu_pd(m_mask, av[5]);
      a_vec[6].v = _mm512_maskz_loadu_pd(m_mask, av[6]);
      a_vec[7].v = _mm512_maskz_loadu_pd(m_mask, av[7]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[4].v = _mm512_fmadd_pd(a_vec[4].v, xv[0].v, rhov[4].v);
      rhov[12].v = _mm512_fmadd_pd(a_vec[4].v, xv[1].v, rhov[12].v);

      rhov[5].v = _mm512_fmadd_pd(a_vec[5].v, xv[0].v, rhov[5].v);
      rhov[13].v = _mm512_fmadd_pd(a_vec[5].v, xv[1].v, rhov[13].v);

      rhov[6].v = _mm512_fmadd_pd(a_vec[6].v, xv[0].v, rhov[6].v);
      rhov[14].v = _mm512_fmadd_pd(a_vec[6].v, xv[1].v, rhov[14].v);

      rhov[7].v = _mm512_fmadd_pd(a_vec[7].v, xv[0].v, rhov[7].v);
      rhov[15].v = _mm512_fmadd_pd(a_vec[7].v, xv[1].v, rhov[15].v);
    }

    // Permuting for final accumulation of real and imag parts
    rhov[8].v = _mm512_permute_pd(rhov[8].v, 0x55);
    rhov[9].v = _mm512_permute_pd(rhov[9].v, 0x55);
    rhov[10].v = _mm512_permute_pd(rhov[10].v, 0x55);
    rhov[11].v = _mm512_permute_pd(rhov[11].v, 0x55);
    rhov[12].v = _mm512_permute_pd(rhov[12].v, 0x55);
    rhov[13].v = _mm512_permute_pd(rhov[13].v, 0x55);
    rhov[14].v = _mm512_permute_pd(rhov[14].v, 0x55);
    rhov[15].v = _mm512_permute_pd(rhov[15].v, 0x55);

    // Setting 2 registers to 0 and 1
    v8df_t zero_reg, scale_one;

    zero_reg.v = _mm512_setzero_pd();
    scale_one.v = _mm512_set1_pd(1.0);

    /*
      conj_op maps to the compute as follows :
      A = (a + ib), X = (x + iy)
      -----------------------------------------------------------
      |      A       |      X       |  Real part  |  Imag Part  |
      -----------------------------------------------------------
      | No-Conjugate | No-Conjugate |   ax - by	  |   bx + ay   |
      | No-Conjugate |   Conjugate  |   ax + by   |   bx - ay   |
      |   Conjugate  | No-Conjugate |   ax + by   | -(bx - ay)  |
      |   Conjugate  |   Conjugate  |   ax - by   | -(bx + ay)  |
      -----------------------------------------------------------

      If only X or A has conjugate, fmsubadd is performed.
      Else, fmaddsub is performed.

      In the final reduction step, the imaginary part of every
      partial sum is negated if conjat is conjugate
    */
    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm512_fmaddsub_pd(scale_one.v, rhov[0].v, rhov[8].v);
      rhov[1].v = _mm512_fmaddsub_pd(scale_one.v, rhov[1].v, rhov[9].v);
      rhov[2].v = _mm512_fmaddsub_pd(scale_one.v, rhov[2].v, rhov[10].v);
      rhov[3].v = _mm512_fmaddsub_pd(scale_one.v, rhov[3].v, rhov[11].v);
      rhov[4].v = _mm512_fmaddsub_pd(scale_one.v, rhov[4].v, rhov[12].v);
      rhov[5].v = _mm512_fmaddsub_pd(scale_one.v, rhov[5].v, rhov[13].v);
      rhov[6].v = _mm512_fmaddsub_pd(scale_one.v, rhov[6].v, rhov[14].v);
      rhov[7].v = _mm512_fmaddsub_pd(scale_one.v, rhov[7].v, rhov[15].v);
    }
    else
    {
      rhov[0].v = _mm512_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[8].v);
      rhov[1].v = _mm512_fmsubadd_pd(scale_one.v, rhov[1].v, rhov[9].v);
      rhov[2].v = _mm512_fmsubadd_pd(scale_one.v, rhov[2].v, rhov[10].v);
      rhov[3].v = _mm512_fmsubadd_pd(scale_one.v, rhov[3].v, rhov[11].v);
      rhov[4].v = _mm512_fmsubadd_pd(scale_one.v, rhov[4].v, rhov[12].v);
      rhov[5].v = _mm512_fmsubadd_pd(scale_one.v, rhov[5].v, rhov[13].v);
      rhov[6].v = _mm512_fmsubadd_pd(scale_one.v, rhov[6].v, rhov[14].v);
      rhov[7].v = _mm512_fmsubadd_pd(scale_one.v, rhov[7].v, rhov[15].v);
    }

    // rhov[0 ... 7] will have the element wise product.
    // These have to be added horizontally(reduction) to get the
    // final result for every element in y.
    // If rhov[0]   = R0 I0 R1 I1 R2 I2 R3 I3
    // Then rhov[8] = R1 I1 R0 I0 R3 I2 R2 I2
    rhov[8].v = _mm512_permutex_pd(rhov[0].v, 0x4E);
    rhov[9].v = _mm512_permutex_pd(rhov[1].v, 0x4E);
    rhov[10].v = _mm512_permutex_pd(rhov[2].v, 0x4E);
    rhov[11].v = _mm512_permutex_pd(rhov[3].v, 0x4E);
    rhov[12].v = _mm512_permutex_pd(rhov[4].v, 0x4E);
    rhov[13].v = _mm512_permutex_pd(rhov[5].v, 0x4E);
    rhov[14].v = _mm512_permutex_pd(rhov[6].v, 0x4E);
    rhov[15].v = _mm512_permutex_pd(rhov[7].v, 0x4E);

    // rhov[0] = (R0 + R1) (I0 + I1) (R1 + R0) (I1 + I0)
    //           (R2 + R3) (I2 + I3) (R3 + R2) (I3 + I2)
    rhov[0].v = _mm512_add_pd(rhov[0].v, rhov[8].v);
    rhov[1].v = _mm512_add_pd(rhov[1].v, rhov[9].v);
    rhov[2].v = _mm512_add_pd(rhov[2].v, rhov[10].v);
    rhov[3].v = _mm512_add_pd(rhov[3].v, rhov[11].v);
    rhov[4].v = _mm512_add_pd(rhov[4].v, rhov[12].v);
    rhov[5].v = _mm512_add_pd(rhov[5].v, rhov[13].v);
    rhov[6].v = _mm512_add_pd(rhov[6].v, rhov[14].v);
    rhov[7].v = _mm512_add_pd(rhov[7].v, rhov[15].v);

    // 256-bit registers declared to extract 256-bit lanes
    v4df_t reduce_sum[16];

    // reduce_sum[0] = (R0 + R1) (I0 + I1) (R1 + R0) (I1 + I0)
    reduce_sum[0].v = _mm512_extractf64x4_pd(rhov[0].v, 0x00);
    reduce_sum[1].v = _mm512_extractf64x4_pd(rhov[1].v, 0x00);
    reduce_sum[2].v = _mm512_extractf64x4_pd(rhov[2].v, 0x00);
    reduce_sum[3].v = _mm512_extractf64x4_pd(rhov[3].v, 0x00);
    reduce_sum[4].v = _mm512_extractf64x4_pd(rhov[4].v, 0x00);
    reduce_sum[5].v = _mm512_extractf64x4_pd(rhov[5].v, 0x00);
    reduce_sum[6].v = _mm512_extractf64x4_pd(rhov[6].v, 0x00);
    reduce_sum[7].v = _mm512_extractf64x4_pd(rhov[7].v, 0x00);

    // reduce_sum[8] = (R2 + R3) (I2 + I3) (R3 + R2) (I3 + I2)
    reduce_sum[8].v = _mm512_extractf64x4_pd(rhov[0].v, 0x1);
    reduce_sum[9].v = _mm512_extractf64x4_pd(rhov[1].v, 0x1);
    reduce_sum[10].v = _mm512_extractf64x4_pd(rhov[2].v, 0x1);
    reduce_sum[11].v = _mm512_extractf64x4_pd(rhov[3].v, 0x1);
    reduce_sum[12].v = _mm512_extractf64x4_pd(rhov[4].v, 0x1);
    reduce_sum[13].v = _mm512_extractf64x4_pd(rhov[5].v, 0x1);
    reduce_sum[14].v = _mm512_extractf64x4_pd(rhov[6].v, 0x1);
    reduce_sum[15].v = _mm512_extractf64x4_pd(rhov[7].v, 0x1);

    // reduce_sum[0] = (R0 + R1 + R2 + R3) (I0 + I1 + I2 + I3) ...
    reduce_sum[0].v = _mm256_add_pd(reduce_sum[0].v, reduce_sum[8].v);
    reduce_sum[1].v = _mm256_add_pd(reduce_sum[1].v, reduce_sum[9].v);
    reduce_sum[2].v = _mm256_add_pd(reduce_sum[2].v, reduce_sum[10].v);
    reduce_sum[3].v = _mm256_add_pd(reduce_sum[3].v, reduce_sum[11].v);
    reduce_sum[4].v = _mm256_add_pd(reduce_sum[4].v, reduce_sum[12].v);
    reduce_sum[5].v = _mm256_add_pd(reduce_sum[5].v, reduce_sum[13].v);
    reduce_sum[6].v = _mm256_add_pd(reduce_sum[6].v, reduce_sum[14].v);
    reduce_sum[7].v = _mm256_add_pd(reduce_sum[7].v, reduce_sum[15].v);

    // The next set of shuffles, permutes and inserts are performed to store
    // all the dot-products onto two 512 registers. They are used to perform
    // aligned stores onto the stack memory.
    reduce_sum[8].v = _mm256_shuffle_pd(reduce_sum[0].v, reduce_sum[1].v, 0xC);
    reduce_sum[9].v = _mm256_shuffle_pd(reduce_sum[2].v, reduce_sum[3].v, 0xC);
    reduce_sum[10].v = _mm256_shuffle_pd(reduce_sum[4].v, reduce_sum[5].v, 0xC);
    reduce_sum[11].v = _mm256_shuffle_pd(reduce_sum[6].v, reduce_sum[7].v, 0xC);

    reduce_sum[12].v = _mm256_permutex_pd(reduce_sum[8].v, 0xD8);
    reduce_sum[13].v = _mm256_permutex_pd(reduce_sum[9].v, 0xD8);
    reduce_sum[14].v = _mm256_permutex_pd(reduce_sum[10].v, 0xD8);
    reduce_sum[15].v = _mm256_permutex_pd(reduce_sum[11].v, 0xD8);

    rhov[0].v = _mm512_insertf64x4(rhov[0].v, reduce_sum[12].v, 0x00);
    rhov[0].v = _mm512_insertf64x4(rhov[0].v, reduce_sum[13].v, 0x01);
    rhov[1].v = _mm512_insertf64x4(rhov[1].v, reduce_sum[14].v, 0x00);
    rhov[1].v = _mm512_insertf64x4(rhov[1].v, reduce_sum[15].v, 0x01);

    // Negate the sign bit of imaginary part of dot-products if conjat is conjugate
    if ( bli_is_conj( conjat ) )
    {
      rhov[0].v = _mm512_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[0].v);
      rhov[1].v = _mm512_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[1].v);
    }

    /*
      Computed dot product result is being stored
      in temp buffer r for further computation.
    */
    _mm512_store_pd((double *)res, rhov[0].v);
    _mm512_store_pd((double *)(res + 4), rhov[1].v);
  }

  // This section will have the whole of compute when incx != 1 || inca != 1
  else
  {
    // Declaring 128-bit registers, for element by element computation
    v2df_t rhov[16], a_vec[8], xv[2];

    // Clearing the partial-sum accumulators
    rhov[0].v = _mm_setzero_pd();
    rhov[1].v = _mm_setzero_pd();
    rhov[2].v = _mm_setzero_pd();
    rhov[3].v = _mm_setzero_pd();
    rhov[4].v = _mm_setzero_pd();
    rhov[5].v = _mm_setzero_pd();
    rhov[6].v = _mm_setzero_pd();
    rhov[7].v = _mm_setzero_pd();
    rhov[8].v = _mm_setzero_pd();
    rhov[9].v = _mm_setzero_pd();
    rhov[10].v = _mm_setzero_pd();
    rhov[11].v = _mm_setzero_pd();
    rhov[12].v = _mm_setzero_pd();
    rhov[13].v = _mm_setzero_pd();
    rhov[14].v = _mm_setzero_pd();
    rhov[15].v = _mm_setzero_pd();

    for (dim_t i = 0; i < m; i++)
    {
      // Load from X
      xv[0].v = _mm_loadu_pd(x_temp);

      // Permute to duplicate the imag part for every element
      xv[1].v = _mm_permute_pd(xv[0].v, 0b11);

      // Permute to duplicate the real part for every element
      xv[0].v = _mm_permute_pd(xv[0].v, 0b00);

      // Load elements from first 4 columns of A
      a_vec[0].v = _mm_loadu_pd(av[0]);
      a_vec[1].v = _mm_loadu_pd(av[1]);
      a_vec[2].v = _mm_loadu_pd(av[2]);
      a_vec[3].v = _mm_loadu_pd(av[3]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[0].v = _mm_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
      rhov[8].v = _mm_fmadd_pd(a_vec[0].v, xv[1].v, rhov[8].v);

      rhov[1].v = _mm_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
      rhov[9].v = _mm_fmadd_pd(a_vec[1].v, xv[1].v, rhov[9].v);

      rhov[2].v = _mm_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
      rhov[10].v = _mm_fmadd_pd(a_vec[2].v, xv[1].v, rhov[10].v);

      rhov[3].v = _mm_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
      rhov[11].v = _mm_fmadd_pd(a_vec[3].v, xv[1].v, rhov[11].v);

      // Load elements from next 4 columns of A
      a_vec[4].v = _mm_loadu_pd(av[4]);
      a_vec[5].v = _mm_loadu_pd(av[5]);
      a_vec[6].v = _mm_loadu_pd(av[6]);
      a_vec[7].v = _mm_loadu_pd(av[7]);

      // Perform: rhov[i].v += a_vec[i].v * xv[0];
      //          rhov[i + 8].v += a_vec[i].v * xv[1];
      // This stores the partial sums due to real and
      // imag components separately
      rhov[4].v = _mm_fmadd_pd(a_vec[4].v, xv[0].v, rhov[4].v);
      rhov[12].v = _mm_fmadd_pd(a_vec[4].v, xv[1].v, rhov[12].v);

      rhov[5].v = _mm_fmadd_pd(a_vec[5].v, xv[0].v, rhov[5].v);
      rhov[13].v = _mm_fmadd_pd(a_vec[5].v, xv[1].v, rhov[13].v);

      rhov[6].v = _mm_fmadd_pd(a_vec[6].v, xv[0].v, rhov[6].v);
      rhov[14].v = _mm_fmadd_pd(a_vec[6].v, xv[1].v, rhov[14].v);

      rhov[7].v = _mm_fmadd_pd(a_vec[7].v, xv[0].v, rhov[7].v);
      rhov[15].v = _mm_fmadd_pd(a_vec[7].v, xv[1].v, rhov[15].v);

      // Adjust the pointers accordingly
      av[0] += 2 * inca;
      av[1] += 2 * inca;
      av[2] += 2 * inca;
      av[3] += 2 * inca;
      av[4] += 2 * inca;
      av[5] += 2 * inca;
      av[6] += 2 * inca;
      av[7] += 2 * inca;

      x_temp += 2 * incx;
    }

    // Permuting to help with final reduction
    rhov[8].v = _mm_permute_pd(rhov[8].v, 0b01);
    rhov[9].v = _mm_permute_pd(rhov[9].v, 0b01);
    rhov[10].v = _mm_permute_pd(rhov[10].v, 0b01);
    rhov[11].v = _mm_permute_pd(rhov[11].v, 0b01);
    rhov[12].v = _mm_permute_pd(rhov[12].v, 0b01);
    rhov[13].v = _mm_permute_pd(rhov[13].v, 0b01);
    rhov[14].v = _mm_permute_pd(rhov[14].v, 0b01);
    rhov[15].v = _mm_permute_pd(rhov[15].v, 0b01);

    v2df_t zero_reg, scale_one;

    zero_reg.v = _mm_setzero_pd();
    scale_one.v = _mm_set1_pd(1.0);

    // Reduction based on conj_op
    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm_addsub_pd(rhov[0].v, rhov[8].v);
      rhov[1].v = _mm_addsub_pd(rhov[1].v, rhov[9].v);
      rhov[2].v = _mm_addsub_pd(rhov[2].v, rhov[10].v);
      rhov[3].v = _mm_addsub_pd(rhov[3].v, rhov[11].v);
      rhov[4].v = _mm_addsub_pd(rhov[4].v, rhov[12].v);
      rhov[5].v = _mm_addsub_pd(rhov[5].v, rhov[13].v);
      rhov[6].v = _mm_addsub_pd(rhov[6].v, rhov[14].v);
      rhov[7].v = _mm_addsub_pd(rhov[7].v, rhov[15].v);
    }
    else
    {
      rhov[0].v = _mm_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[8].v);
      rhov[1].v = _mm_fmsubadd_pd(scale_one.v, rhov[1].v, rhov[9].v);
      rhov[2].v = _mm_fmsubadd_pd(scale_one.v, rhov[2].v, rhov[10].v);
      rhov[3].v = _mm_fmsubadd_pd(scale_one.v, rhov[3].v, rhov[11].v);
      rhov[4].v = _mm_fmsubadd_pd(scale_one.v, rhov[4].v, rhov[12].v);
      rhov[5].v = _mm_fmsubadd_pd(scale_one.v, rhov[5].v, rhov[13].v);
      rhov[6].v = _mm_fmsubadd_pd(scale_one.v, rhov[6].v, rhov[14].v);
      rhov[7].v = _mm_fmsubadd_pd(scale_one.v, rhov[7].v, rhov[15].v);
    }
    if( bli_is_conj( conjat ) )
    {
      rhov[0].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[0].v);
      rhov[1].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[1].v);
      rhov[2].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[2].v);
      rhov[3].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[3].v);
      rhov[4].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[4].v);
      rhov[5].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[5].v);
      rhov[6].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[6].v);
      rhov[7].v = _mm_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[7].v);
    }

    // Storing onto stack memory
    _mm_storeu_pd((double *)res, rhov[0].v);
    _mm_storeu_pd((double *)(res + 1), rhov[1].v);
    _mm_storeu_pd((double *)(res + 2), rhov[2].v);
    _mm_storeu_pd((double *)(res + 3), rhov[3].v);
    _mm_storeu_pd((double *)(res + 4), rhov[4].v);
    _mm_storeu_pd((double *)(res + 5), rhov[5].v);
    _mm_storeu_pd((double *)(res + 6), rhov[6].v);
    _mm_storeu_pd((double *)(res + 7), rhov[7].v);

  }

  // Scaling by alpha
  // Registers to load dot-products from res
  v8df_t rhov[2], temp[2];

  rhov[0].v = _mm512_load_pd((double *)res);
  rhov[1].v = _mm512_load_pd((double *)(res + 4));

  if ( !bli_zeq1( *alpha ) )
  {
    __m512d alphaRv, alphaIv;
    alphaRv = _mm512_set1_pd((*alpha).real);
    alphaIv = _mm512_set1_pd((*alpha).imag);

    temp[0].v = _mm512_permute_pd(rhov[0].v, 0x55);
    temp[1].v = _mm512_permute_pd(rhov[1].v, 0x55);

    // Scaling with imag part of alpha
    temp[0].v = _mm512_mul_pd(temp[0].v, alphaIv);
    temp[1].v = _mm512_mul_pd(temp[1].v, alphaIv);

    // Scaling with real part of alpha, and addsub
    rhov[0].v = _mm512_fmaddsub_pd(rhov[0].v, alphaRv, temp[0].v);
    rhov[1].v = _mm512_fmaddsub_pd(rhov[1].v, alphaRv, temp[1].v);
  }

  // When 'beta' is not zero we need to scale 'y' by 'beta'
  v8df_t yv[2];

  yv[0].v = _mm512_setzero_pd();
  yv[1].v = _mm512_setzero_pd();

  if (!PASTEMAC(z, eq0)(*beta))
  {
    __m512d betaRv, betaIv;

    betaRv = _mm512_set1_pd((*beta).real);
    betaIv = _mm512_set1_pd((*beta).imag);

    if (incy == 1)
    {
      yv[0].v = _mm512_loadu_pd((double *)(y));
      yv[1].v = _mm512_loadu_pd((double *)(y + 4));
    }
    else
    {
      /*
        This can be done using SSE instructions
        but has been kept as scalar code to avoid
        mixing SSE with AVX
      */
      yv[0].d[0] = (*(y + 0 * incy)).real;
      yv[0].d[1] = (*(y + 0 * incy)).imag;
      yv[0].d[2] = (*(y + 1 * incy)).real;
      yv[0].d[3] = (*(y + 1 * incy)).imag;
      yv[0].d[4] = (*(y + 2 * incy)).real;
      yv[0].d[5] = (*(y + 2 * incy)).imag;
      yv[0].d[6] = (*(y + 3 * incy)).real;
      yv[0].d[7] = (*(y + 3 * incy)).imag;

      yv[1].d[0] = (*(y + 4 * incy)).real;
      yv[1].d[1] = (*(y + 4 * incy)).imag;
      yv[1].d[2] = (*(y + 5 * incy)).real;
      yv[1].d[3] = (*(y + 5 * incy)).imag;
      yv[1].d[4] = (*(y + 6 * incy)).real;
      yv[1].d[5] = (*(y + 6 * incy)).imag;
      yv[1].d[6] = (*(y + 7 * incy)).real;
      yv[1].d[7] = (*(y + 7 * incy)).imag;
    }

    temp[0].v = _mm512_permute_pd(yv[0].v, 0x55);
    temp[1].v = _mm512_permute_pd(yv[1].v, 0x55);

    // Scaling with imag part of alpha
    temp[0].v = _mm512_mul_pd(temp[0].v, betaIv);
    temp[1].v = _mm512_mul_pd(temp[1].v, betaIv);

    // Scaling with real part of alpha, and addsub
    yv[0].v = _mm512_fmaddsub_pd(yv[0].v, betaRv, temp[0].v);
    yv[1].v = _mm512_fmaddsub_pd(yv[1].v, betaRv, temp[1].v);
  }

  // Adding alpha*A*x to beta*Y
  yv[0].v = _mm512_add_pd(yv[0].v, rhov[0].v);
  yv[1].v = _mm512_add_pd(yv[1].v, rhov[1].v);

  if (incy == 1)
  {
    _mm512_storeu_pd((double *)y, yv[0].v);
    _mm512_storeu_pd((double *)(y + 4), yv[1].v);
  }
  else
  {
    (*(y + 0 * incy)).real = yv[0].d[0];
    (*(y + 0 * incy)).imag = yv[0].d[1];
    (*(y + 1 * incy)).real = yv[0].d[2];
    (*(y + 1 * incy)).imag = yv[0].d[3];

    (*(y + 2 * incy)).real = yv[0].d[4];
    (*(y + 2 * incy)).imag = yv[0].d[5];
    (*(y + 3 * incy)).real = yv[0].d[6];
    (*(y + 3 * incy)).imag = yv[0].d[7];

    (*(y + 4 * incy)).real = yv[1].d[0];
    (*(y + 4 * incy)).imag = yv[1].d[1];
    (*(y + 5 * incy)).real = yv[1].d[2];
    (*(y + 5 * incy)).imag = yv[1].d[3];

    (*(y + 6 * incy)).real = yv[1].d[4];
    (*(y + 6 * incy)).imag = yv[1].d[5];
    (*(y + 7 * incy)).real = yv[1].d[6];
    (*(y + 7 * incy)).imag = yv[1].d[7];
  }

}
