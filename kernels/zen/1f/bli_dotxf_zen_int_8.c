/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2017 - 21, Advanced Micro Devices, Inc. All rights reserved.

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

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

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

// -----------------------------------------------------------------------------

void bli_sdotxf_zen_int_8
     (
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float*  restrict alpha,
       float*  restrict a, inc_t inca, inc_t lda,
       float*  restrict x, inc_t incx,
       float*  restrict beta,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	const dim_t fuse_fac       = 8;
	const dim_t n_elem_per_reg = 8;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if ( bli_zero_dim1( b_n ) ) return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if ( bli_zero_dim1( m ) || PASTEMAC(s,eq0)( *alpha ) )
	{

		bli_sscalv_zen_int10
		(
		  BLIS_NO_CONJUGATE,
		  b_n,
		  beta,
		  y, incy,
		  cntx
		);
		return;
	}

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if ( b_n != fuse_fac )
	{
		for ( dim_t i = 0; i < b_n; ++i )
		{
			float* a1   = a + (0  )*inca + (i  )*lda;
			float* x1   = x + (0  )*incx;
			float* psi1 = y + (i  )*incy;

			bli_sdotxv_zen_int
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

	// Intermediate variables to hold the completed dot products
    float rho0 = 0, rho1 = 0, rho2 = 0, rho3 = 0,
	      rho4 = 0, rho5 = 0, rho6 = 0, rho7 = 0;

	if ( inca == 1 && incx == 1 )
	{
		const dim_t n_iter_unroll = 1;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_elem_per_reg * n_iter_unroll );

		// Set up pointers for x and the b_n columns of A (rows of A^T).
		float* restrict x0 = x;
		float* restrict a0 = a + 0*lda;
		float* restrict a1 = a + 1*lda;
		float* restrict a2 = a + 2*lda;
		float* restrict a3 = a + 3*lda;
		float* restrict a4 = a + 4*lda;
		float* restrict a5 = a + 5*lda;
		float* restrict a6 = a + 6*lda;
		float* restrict a7 = a + 7*lda;

		// Initialize b_n rho vector accumulators to zero.
		v8sf_t rho0v; rho0v.v = _mm256_setzero_ps();
		v8sf_t rho1v; rho1v.v = _mm256_setzero_ps();
		v8sf_t rho2v; rho2v.v = _mm256_setzero_ps();
		v8sf_t rho3v; rho3v.v = _mm256_setzero_ps();
		v8sf_t rho4v; rho4v.v = _mm256_setzero_ps();
		v8sf_t rho5v; rho5v.v = _mm256_setzero_ps();
		v8sf_t rho6v; rho6v.v = _mm256_setzero_ps();
		v8sf_t rho7v; rho7v.v = _mm256_setzero_ps();

		v8sf_t x0v;
		v8sf_t a0v, a1v, a2v, a3v, a4v, a5v, a6v, a7v;

		// If there are vectorized iterations, perform them with vector
		// instructions.
		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			x0v.v = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

			a0v.v = _mm256_loadu_ps( a0 + 0*n_elem_per_reg );
			a1v.v = _mm256_loadu_ps( a1 + 0*n_elem_per_reg );
			a2v.v = _mm256_loadu_ps( a2 + 0*n_elem_per_reg );
			a3v.v = _mm256_loadu_ps( a3 + 0*n_elem_per_reg );
			a4v.v = _mm256_loadu_ps( a4 + 0*n_elem_per_reg );
			a5v.v = _mm256_loadu_ps( a5 + 0*n_elem_per_reg );
			a6v.v = _mm256_loadu_ps( a6 + 0*n_elem_per_reg );
			a7v.v = _mm256_loadu_ps( a7 + 0*n_elem_per_reg );

			// perform: rho?v += a?v * x0v;
			rho0v.v = _mm256_fmadd_ps( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_ps( a1v.v, x0v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_ps( a2v.v, x0v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_ps( a3v.v, x0v.v, rho3v.v );
			rho4v.v = _mm256_fmadd_ps( a4v.v, x0v.v, rho4v.v );
			rho5v.v = _mm256_fmadd_ps( a5v.v, x0v.v, rho5v.v );
			rho6v.v = _mm256_fmadd_ps( a6v.v, x0v.v, rho6v.v );
			rho7v.v = _mm256_fmadd_ps( a7v.v, x0v.v, rho7v.v );

			x0 += n_elem_per_reg * n_iter_unroll;
			a0 += n_elem_per_reg * n_iter_unroll;
			a1 += n_elem_per_reg * n_iter_unroll;
			a2 += n_elem_per_reg * n_iter_unroll;
			a3 += n_elem_per_reg * n_iter_unroll;
			a4 += n_elem_per_reg * n_iter_unroll;
			a5 += n_elem_per_reg * n_iter_unroll;
			a6 += n_elem_per_reg * n_iter_unroll;
			a7 += n_elem_per_reg * n_iter_unroll;
		}
		
		// Now we need to sum the elements within each vector.
		// Sum the elements of a given rho?v with hadd.

		rho0v.v = _mm256_hadd_ps( rho0v.v, rho1v.v);
		rho1v.v = _mm256_hadd_ps( rho2v.v, rho3v.v);
		rho2v.v = _mm256_hadd_ps( rho4v.v, rho5v.v);
		rho3v.v = _mm256_hadd_ps( rho6v.v, rho7v.v);
		rho0v.v = _mm256_hadd_ps( rho0v.v, rho0v.v);
		rho1v.v = _mm256_hadd_ps( rho1v.v, rho1v.v);
		rho2v.v = _mm256_hadd_ps( rho2v.v, rho2v.v);
		rho3v.v = _mm256_hadd_ps( rho3v.v, rho3v.v);

		// Manually add the results from above to finish the sum.
		rho0    = rho0v.f[0] + rho0v.f[4];
		rho1    = rho0v.f[1] + rho0v.f[5];
		rho2    = rho1v.f[0] + rho1v.f[4];
		rho3    = rho1v.f[1] + rho1v.f[5];
		rho4    = rho2v.f[0] + rho2v.f[4];
		rho5    = rho2v.f[1] + rho2v.f[5];
		rho6    = rho3v.f[0] + rho3v.f[4];
		rho7    = rho3v.f[1] + rho3v.f[5];

		// Adjust for scalar subproblem.
		m -= n_elem_per_reg * n_iter_unroll * m_viter;
		a += n_elem_per_reg * n_iter_unroll * m_viter /* * inca */;
		x += n_elem_per_reg * n_iter_unroll * m_viter /* * incx */;
	}
	else if ( lda == 1 )
	{
		const dim_t n_iter_unroll = 4;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_iter_unroll );

		// Initialize pointers for x and A.
		float* restrict x0 = x;
		float* restrict a0 = a;

		// Initialize rho vector accumulators to zero.
		v8sf_t rho0v; rho0v.v = _mm256_setzero_ps();
		v8sf_t rho1v; rho1v.v = _mm256_setzero_ps();
		v8sf_t rho2v; rho2v.v = _mm256_setzero_ps();
		v8sf_t rho3v; rho3v.v = _mm256_setzero_ps();

		v8sf_t x0v, x1v, x2v, x3v;
		v8sf_t a0v, a1v, a2v, a3v;

		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			a0v.v = _mm256_loadu_ps( a0 + 0*inca );
			a1v.v = _mm256_loadu_ps( a0 + 1*inca );
			a2v.v = _mm256_loadu_ps( a0 + 2*inca );
			a3v.v = _mm256_loadu_ps( a0 + 3*inca );

			x0v.v = _mm256_broadcast_ss( x0 + 0*incx );
			x1v.v = _mm256_broadcast_ss( x0 + 1*incx );
			x2v.v = _mm256_broadcast_ss( x0 + 2*incx );
			x3v.v = _mm256_broadcast_ss( x0 + 3*incx );

			// perform : rho?v += a?v * x?v;
			rho0v.v = _mm256_fmadd_ps( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_ps( a1v.v, x1v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_ps( a2v.v, x2v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_ps( a3v.v, x3v.v, rho3v.v );

			x0 += incx * n_iter_unroll;
			a0 += inca * n_iter_unroll;
		}

		// Combine the 8 accumulators into one vector register.
		rho0v.v = _mm256_add_ps( rho0v.v, rho1v.v );
		rho2v.v = _mm256_add_ps( rho2v.v, rho3v.v );
		rho0v.v = _mm256_add_ps( rho0v.v, rho2v.v );

		// Write vector components to scalar values.
		rho0 = rho0v.f[0];
		rho1 = rho0v.f[1];
		rho2 = rho0v.f[2];
		rho3 = rho0v.f[3];
		rho4 = rho0v.f[4];
		rho5 = rho0v.f[5];
		rho6 = rho0v.f[6];
		rho7 = rho0v.f[7];

		// Adjust for scalar subproblem.
		m -= n_iter_unroll * m_viter;
		a += n_iter_unroll * m_viter * inca;
		x += n_iter_unroll * m_viter * incx;
	}
	else
	{
		// No vectorization possible; use scalar iterations for the entire
		// problem.
	}

	// Scalar edge case.
	{
		// Initialize pointers for x and the b_n columns of A (rows of A^T).
		float* restrict x0 = x;
		float* restrict a0 = a + 0*lda;
		float* restrict a1 = a + 1*lda;
		float* restrict a2 = a + 2*lda;
		float* restrict a3 = a + 3*lda;
		float* restrict a4 = a + 4*lda;
		float* restrict a5 = a + 5*lda;
		float* restrict a6 = a + 6*lda;
		float* restrict a7 = a + 7*lda;

		// If there are leftover iterations, perform them with scalar code.
		for ( dim_t i = 0; i < m ; ++i )
		{
			const float x0c = *x0;

			const float a0c = *a0;
			const float a1c = *a1;
			const float a2c = *a2;
			const float a3c = *a3;
			const float a4c = *a4;
			const float a5c = *a5;
			const float a6c = *a6;
			const float a7c = *a7;

			rho0 += a0c * x0c;
			rho1 += a1c * x0c;
			rho2 += a2c * x0c;
			rho3 += a3c * x0c;
			rho4 += a4c * x0c;
			rho5 += a5c * x0c;
			rho6 += a6c * x0c;
			rho7 += a7c * x0c;

			x0 += incx;
			a0 += inca;
			a1 += inca;
			a2 += inca;
			a3 += inca;
			a4 += inca;
			a5 += inca;
			a6 += inca;
			a7 += inca;
		}
	}

	// Now prepare the final rho values to output/accumulate back into
	// the y vector.

	v8sf_t rho0v, y0v;

	// Insert the scalar rho values into a single vector.
	rho0v.f[0] = rho0;
	rho0v.f[1] = rho1;
	rho0v.f[2] = rho2;
	rho0v.f[3] = rho3;
	rho0v.f[4] = rho4;
	rho0v.f[5] = rho5;
	rho0v.f[6] = rho6;
	rho0v.f[7] = rho7;

	// Broadcast the alpha scalar.
	v8sf_t alphav; alphav.v = _mm256_broadcast_ss( alpha );

	// We know at this point that alpha is nonzero; however, beta may still
	// be zero. If beta is indeed zero, we must overwrite y rather than scale
	// by beta (in case y contains NaN or Inf).
	if ( PASTEMAC(s,eq0)( *beta ) )
	{
		// Apply alpha to the accumulated dot product in rho:
		//   y := alpha * rho
		y0v.v = _mm256_mul_ps( alphav.v, rho0v.v );
	}
	else
	{
		// Broadcast the beta scalar.
		v8sf_t betav; betav.v = _mm256_broadcast_ss( beta );

		// Load y.
		if ( incy == 1 )
		{
			y0v.v = _mm256_loadu_ps( y + 0*n_elem_per_reg );
		}
		else
		{
			y0v.f[0] = *(y + 0*incy); y0v.f[1] = *(y + 1*incy);
			y0v.f[2] = *(y + 2*incy); y0v.f[3] = *(y + 3*incy);
			y0v.f[4] = *(y + 4*incy); y0v.f[5] = *(y + 5*incy);
			y0v.f[6] = *(y + 6*incy); y0v.f[7] = *(y + 7*incy);
		}

		// Apply beta to y and alpha to the accumulated dot product in rho:
		//   y := beta * y + alpha * rho
		y0v.v = _mm256_mul_ps( betav.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( alphav.v, rho0v.v, y0v.v );
	}

	// Store the output.
	if ( incy == 1 )
	{
		_mm256_storeu_ps( (y + 0*n_elem_per_reg), y0v.v );
	}
	else
	{
		*(y + 0*incy) = y0v.f[0]; *(y + 1*incy) = y0v.f[1];
		*(y + 2*incy) = y0v.f[2]; *(y + 3*incy) = y0v.f[3];
		*(y + 4*incy) = y0v.f[4]; *(y + 5*incy) = y0v.f[5];
		*(y + 6*incy) = y0v.f[6]; *(y + 7*incy) = y0v.f[7];
	}
}

// -----------------------------------------------------------------------------

void bli_ddotxf_zen_int_8
     (
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	const dim_t fuse_fac = 8;
	const dim_t n_elem_per_reg = 4;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if (bli_zero_dim1(b_n))
		return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if (bli_zero_dim1(m) || PASTEMAC(d, eq0)(*alpha))
	{
		bli_dscalv_zen_int10(
			BLIS_NO_CONJUGATE,
			b_n,
			beta,
			y, incy,
			cntx);
		return;
	}

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if (b_n != fuse_fac)
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

	// Intermediate variables to hold the completed dot products
        double rho0 = 0, rho1 = 0, rho2 = 0, rho3 = 0;
        double rho4 = 0, rho5 = 0, rho6 = 0, rho7 = 0;

	if (inca == 1 && incx == 1)
	{
		const dim_t n_iter_unroll = 1;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter;

		// Calculate the number of vector iterations that can occur
		// for the given unroll factors.
		m_viter = (m) / (n_elem_per_reg * n_iter_unroll);

		// Set up pointers for x and the b_n columns of A (rows of A^T).
		double *restrict x0 = x;
		double *restrict av[8];

		av[0] = a + 0 * lda;
		av[1] = a + 1 * lda;
		av[2] = a + 2 * lda;
		av[3] = a + 3 * lda;
		av[4] = a + 4 * lda;
		av[5] = a + 5 * lda;
		av[6] = a + 6 * lda;
		av[7] = a + 7 * lda;

		// Initialize b_n rho vector accumulators to zero.
		v4df_t rhov[8];

		rhov[0].v = _mm256_setzero_pd();
		rhov[1].v = _mm256_setzero_pd();
		rhov[2].v = _mm256_setzero_pd();
		rhov[3].v = _mm256_setzero_pd();
		rhov[4].v = _mm256_setzero_pd();
		rhov[5].v = _mm256_setzero_pd();
		rhov[6].v = _mm256_setzero_pd();
		rhov[7].v = _mm256_setzero_pd();

		v4df_t xv;
		v4df_t avec[8];

		for (dim_t i = 0; i < m_viter; ++i)
		{
			// Load the input values.
			xv.v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[2] + 0 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[3] + 0 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv.v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv.v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv.v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv.v, rhov[3].v);

			avec[4].v = _mm256_loadu_pd(av[4] + 0 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[5] + 0 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[6] + 0 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[7] + 0 * n_elem_per_reg);

			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv.v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv.v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv.v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv.v, rhov[7].v);

			x0 += n_elem_per_reg * n_iter_unroll;
			av[0] += n_elem_per_reg * n_iter_unroll;
			av[1] += n_elem_per_reg * n_iter_unroll;
			av[2] += n_elem_per_reg * n_iter_unroll;
			av[3] += n_elem_per_reg * n_iter_unroll;
			av[4] += n_elem_per_reg * n_iter_unroll;
			av[5] += n_elem_per_reg * n_iter_unroll;
			av[6] += n_elem_per_reg * n_iter_unroll;
			av[7] += n_elem_per_reg * n_iter_unroll;
		}

		// Sum the elements of a given rho?v. This computes the sum of
		// elements within lanes and stores the sum to both elements.
		rhov[0].v = _mm256_hadd_pd(rhov[0].v, rhov[0].v);
		rhov[1].v = _mm256_hadd_pd(rhov[1].v, rhov[1].v);
		rhov[2].v = _mm256_hadd_pd(rhov[2].v, rhov[2].v);
		rhov[3].v = _mm256_hadd_pd(rhov[3].v, rhov[3].v);
		rhov[4].v = _mm256_hadd_pd(rhov[4].v, rhov[4].v);
		rhov[5].v = _mm256_hadd_pd(rhov[5].v, rhov[5].v);
		rhov[6].v = _mm256_hadd_pd(rhov[6].v, rhov[6].v);
		rhov[7].v = _mm256_hadd_pd(rhov[7].v, rhov[7].v);

		// Manually add the results from above to finish the sum.
		rho0 = rhov[0].d[0] + rhov[0].d[2];
		rho1 = rhov[1].d[0] + rhov[1].d[2];
		rho2 = rhov[2].d[0] + rhov[2].d[2];
		rho3 = rhov[3].d[0] + rhov[3].d[2];
		rho4 = rhov[4].d[0] + rhov[4].d[2];
		rho5 = rhov[5].d[0] + rhov[5].d[2];
		rho6 = rhov[6].d[0] + rhov[6].d[2];
		rho7 = rhov[7].d[0] + rhov[7].d[2];

		// Adjust for scalar subproblem.
		m -= n_elem_per_reg * n_iter_unroll * m_viter;
		a += n_elem_per_reg * n_iter_unroll * m_viter /* * inca */;
		x += n_elem_per_reg * n_iter_unroll * m_viter /* * incx */;

	}else if (lda == 1)
	{
		const dim_t n_iter_unroll = 3;
		const dim_t n_reg_per_row = 2; // fuse_fac / n_elem_per_reg;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_reg_per_row * n_iter_unroll );

		// Initialize pointers for x and A.
		double* restrict x0 = x;
		double* restrict a0 = a;

		// Initialize rho vector accumulators to zero.
		v4df_t rho0v; rho0v.v = _mm256_setzero_pd();
		v4df_t rho1v; rho1v.v = _mm256_setzero_pd();
		v4df_t rho2v; rho2v.v = _mm256_setzero_pd();
		v4df_t rho3v; rho3v.v = _mm256_setzero_pd();
		v4df_t rho4v; rho4v.v = _mm256_setzero_pd();
		v4df_t rho5v; rho5v.v = _mm256_setzero_pd();

		v4df_t x0v, x1v, x2v;
		v4df_t a0v, a1v, a2v, a3v, a4v, a5v;

		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			a0v.v = _mm256_loadu_pd( a0 + 0*inca + 0*n_elem_per_reg );
			a1v.v = _mm256_loadu_pd( a0 + 0*inca + 1*n_elem_per_reg );
			a2v.v = _mm256_loadu_pd( a0 + 1*inca + 0*n_elem_per_reg );
			a3v.v = _mm256_loadu_pd( a0 + 1*inca + 1*n_elem_per_reg );
			a4v.v = _mm256_loadu_pd( a0 + 2*inca + 0*n_elem_per_reg );
			a5v.v = _mm256_loadu_pd( a0 + 2*inca + 1*n_elem_per_reg );

			x0v.v = _mm256_broadcast_sd( x0 + 0*incx );
			x1v.v = _mm256_broadcast_sd( x0 + 1*incx );
			x2v.v = _mm256_broadcast_sd( x0 + 2*incx );

			// perform : rho?v += a?v * x?v;
			rho0v.v = _mm256_fmadd_pd( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_pd( a1v.v, x0v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_pd( a2v.v, x1v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_pd( a3v.v, x1v.v, rho3v.v );
			rho4v.v = _mm256_fmadd_pd( a4v.v, x2v.v, rho4v.v );
			rho5v.v = _mm256_fmadd_pd( a5v.v, x2v.v, rho5v.v );

			x0 += incx * n_iter_unroll;
			a0 += inca * n_iter_unroll;
		}

		// Combine the 8 accumulators into one vector register.
		rho0v.v = _mm256_add_pd( rho0v.v, rho2v.v );
		rho0v.v = _mm256_add_pd( rho0v.v, rho4v.v );
		rho1v.v = _mm256_add_pd( rho1v.v, rho3v.v );
		rho1v.v = _mm256_add_pd( rho1v.v, rho5v.v );

		// Write vector components to scalar values.
		rho0 = rho0v.d[0];
		rho1 = rho0v.d[1];
		rho2 = rho0v.d[2];
		rho3 = rho0v.d[3];
		rho4 = rho1v.d[0];
		rho5 = rho1v.d[1];
		rho6 = rho1v.d[2];
		rho7 = rho1v.d[3];

		// Adjust for scalar subproblem.
		m -= n_iter_unroll * m_viter;
		a += n_iter_unroll * m_viter * inca;
		x += n_iter_unroll * m_viter * incx;
	}

	// Initialize pointers for x and the b_n columns of A (rows of A^T).
	double *restrict x0 = x;
	double *restrict a0 = a + 0 * lda;
	double *restrict a1 = a + 1 * lda;
	double *restrict a2 = a + 2 * lda;
	double *restrict a3 = a + 3 * lda;
	double *restrict a4 = a + 4 * lda;
	double *restrict a5 = a + 5 * lda;
	double *restrict a6 = a + 6 * lda;
	double *restrict a7 = a + 7 * lda;

	// If there are leftover iterations, perform them with scalar code.
	for (dim_t i = 0; i < m; ++i)
	{
		const double x0c = *x0;

		const double a0c = *a0;
		const double a1c = *a1;
		const double a2c = *a2;
		const double a3c = *a3;
		const double a4c = *a4;
		const double a5c = *a5;
		const double a6c = *a6;
		const double a7c = *a7;

		rho0 += a0c * x0c;
		rho1 += a1c * x0c;
		rho2 += a2c * x0c;
		rho3 += a3c * x0c;
		rho4 += a4c * x0c;
		rho5 += a5c * x0c;
		rho6 += a6c * x0c;
		rho7 += a7c * x0c;

		x0 += incx;
		a0 += inca;
		a1 += inca;
		a2 += inca;
		a3 += inca;
		a4 += inca;
		a5 += inca;
		a6 += inca;
		a7 += inca;
	}

	// Now prepare the final rho values to output/accumulate back into
	// the y vector.

	v4df_t rho0v, rho1v, y0v, y1v;

	// Insert the scalar rho values into a single vector.
	rho0v.d[0] = rho0;
	rho0v.d[1] = rho1;
	rho0v.d[2] = rho2;
	rho0v.d[3] = rho3;
	rho1v.d[0] = rho4;
	rho1v.d[1] = rho5;
	rho1v.d[2] = rho6;
	rho1v.d[3] = rho7;

	// Broadcast the alpha scalar.
	v4df_t alphav;
	alphav.v = _mm256_broadcast_sd(alpha);

	// We know at this point that alpha is nonzero; however, beta may still
	// be zero. If beta is indeed zero, we must overwrite y rather than scale
	// by beta (in case y contains NaN or Inf).
	if (PASTEMAC(d, eq0)(*beta))
	{
		// Apply alpha to the accumulated dot product in rho:
		//   y := alpha * rho
		y0v.v = _mm256_mul_pd(alphav.v, rho0v.v);
		y1v.v = _mm256_mul_pd(alphav.v, rho1v.v);
	}
	else
	{
		// Broadcast the beta scalar.
		v4df_t betav;
		betav.v = _mm256_broadcast_sd(beta);

		// Load y.
		if (incy == 1)
		{
			y0v.v = _mm256_loadu_pd(y + 0 * n_elem_per_reg);
			y1v.v = _mm256_loadu_pd(y + 1 * n_elem_per_reg);
		}
		else
		{
			y0v.d[0] = *(y + 0 * incy);
			y0v.d[1] = *(y + 1 * incy);
			y0v.d[2] = *(y + 2 * incy);
			y0v.d[3] = *(y + 3 * incy);
			y1v.d[0] = *(y + 4 * incy);
			y1v.d[1] = *(y + 5 * incy);
			y1v.d[2] = *(y + 6 * incy);
			y1v.d[3] = *(y + 7 * incy);
		}

		// Apply beta to y and alpha to the accumulated dot product in rho:
		//   y := beta * y + alpha * rho
		y0v.v = _mm256_mul_pd(betav.v, y0v.v);
		y1v.v = _mm256_mul_pd(betav.v, y1v.v);
		y0v.v = _mm256_fmadd_pd(alphav.v, rho0v.v, y0v.v);
		y1v.v = _mm256_fmadd_pd(alphav.v, rho1v.v, y1v.v);
	}

	if (incy == 1)
	{
		// Store the output.
		_mm256_storeu_pd((y + 0 * n_elem_per_reg), y0v.v);
		_mm256_storeu_pd((y + 1 * n_elem_per_reg), y1v.v);
	}
	else
	{
		*(y + 0 * incy) = y0v.d[0];
		*(y + 1 * incy) = y0v.d[1];
		*(y + 2 * incy) = y0v.d[2];
		*(y + 3 * incy) = y0v.d[3];
		*(y + 4 * incy) = y1v.d[0];
		*(y + 5 * incy) = y1v.d[1];
		*(y + 6 * incy) = y1v.d[2];
		*(y + 7 * incy) = y1v.d[3];
	}
}


void bli_ddotxf_zen_int_4
	(
		conj_t conjat,
		conj_t conjx,
		dim_t m,
		dim_t b_n,
		double *restrict alpha,
		double *restrict a, inc_t inca, inc_t lda,
		double *restrict x, inc_t incx,
		double *restrict beta,
		double *restrict y, inc_t incy,
		cntx_t *restrict cntx
	)
{
	const dim_t fuse_fac = 4;
	const dim_t n_elem_per_reg = 4;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if (bli_zero_dim1(b_n))
		return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if (bli_zero_dim1(m) || PASTEMAC(d, eq0)(*alpha))
	{
		bli_dscalv_zen_int10(
			BLIS_NO_CONJUGATE,
			b_n,
			beta,
			y, incy,
			cntx);
		return;
	}

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if (b_n != fuse_fac)
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

	// Intermediate variables to hold the completed dot products
	double rho0 = 0, rho1 = 0, rho2 = 0, rho3 = 0;

	if (inca == 1 && incx == 1)
	{
		const dim_t n_iter_unroll[4] = {4, 3, 2, 1};

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter[4], m_left = m, i;

		// Calculate the number of vector iterations that can occur for
		// various unroll factors.
		for (i = 0; i < 4; ++i)
		{
			m_viter[i] = (m_left) / (n_elem_per_reg * n_iter_unroll[i]);
			m_left = (m_left) % (n_elem_per_reg * n_iter_unroll[i]);
		}

		// Set up pointers for x and the b_n columns of A (rows of A^T).
		double *restrict x0 = x;
		double *restrict av[4];

		av[0] = a + 0 * lda;
		av[1] = a + 1 * lda;
		av[2] = a + 2 * lda;
		av[3] = a + 3 * lda;

		// Initialize b_n rho vector accumulators to zero.
		v4df_t rhov[8];

		rhov[0].v = _mm256_setzero_pd();
		rhov[1].v = _mm256_setzero_pd();
		rhov[2].v = _mm256_setzero_pd();
		rhov[3].v = _mm256_setzero_pd();
		rhov[4].v = _mm256_setzero_pd();
		rhov[5].v = _mm256_setzero_pd();
		rhov[6].v = _mm256_setzero_pd();
		rhov[7].v = _mm256_setzero_pd();

		v4df_t xv[4];
		v4df_t avec[16];

		// If there are vectorized iterations, perform them with vector
		// instructions.
		for (i = 0; i < m_viter[0]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);
			xv[2].v = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);
			xv[3].v = _mm256_loadu_pd(x0 + 3 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[2] + 0 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[3] + 0 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[0].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[0].v, rhov[3].v);

			avec[4].v = _mm256_loadu_pd(av[0] + 1 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[1] + 1 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[2] + 1 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[3] + 1 * n_elem_per_reg);

			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv[1].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv[1].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv[1].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv[1].v, rhov[7].v);

			avec[8].v = _mm256_loadu_pd(av[0] + 2 * n_elem_per_reg);
			avec[9].v = _mm256_loadu_pd(av[1] + 2 * n_elem_per_reg);
			avec[10].v = _mm256_loadu_pd(av[2] + 2 * n_elem_per_reg);
			avec[11].v = _mm256_loadu_pd(av[3] + 2 * n_elem_per_reg);

			rhov[0].v = _mm256_fmadd_pd(avec[8].v, xv[2].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[9].v, xv[2].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[10].v, xv[2].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[11].v, xv[2].v, rhov[3].v);

			avec[12].v = _mm256_loadu_pd(av[0] + 3 * n_elem_per_reg);
			avec[13].v = _mm256_loadu_pd(av[1] + 3 * n_elem_per_reg);
			avec[14].v = _mm256_loadu_pd(av[2] + 3 * n_elem_per_reg);
			avec[15].v = _mm256_loadu_pd(av[3] + 3 * n_elem_per_reg);

			rhov[4].v = _mm256_fmadd_pd(avec[12].v, xv[3].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[13].v, xv[3].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[14].v, xv[3].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[15].v, xv[3].v, rhov[7].v);

			x0 += n_elem_per_reg * n_iter_unroll[0];
			av[0] += n_elem_per_reg * n_iter_unroll[0];
			av[1] += n_elem_per_reg * n_iter_unroll[0];
			av[2] += n_elem_per_reg * n_iter_unroll[0];
			av[3] += n_elem_per_reg * n_iter_unroll[0];
		}

		for (i = 0; i < m_viter[1]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);
			xv[2].v = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[2] + 0 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[3] + 0 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[0].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[0].v, rhov[3].v);

			avec[4].v = _mm256_loadu_pd(av[0] + 1 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[1] + 1 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[2] + 1 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[3] + 1 * n_elem_per_reg);

			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv[1].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv[1].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv[1].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv[1].v, rhov[7].v);

			avec[8].v = _mm256_loadu_pd(av[0] + 2 * n_elem_per_reg);
			avec[9].v = _mm256_loadu_pd(av[1] + 2 * n_elem_per_reg);
			avec[10].v = _mm256_loadu_pd(av[2] + 2 * n_elem_per_reg);
			avec[11].v = _mm256_loadu_pd(av[3] + 2 * n_elem_per_reg);

			rhov[0].v = _mm256_fmadd_pd(avec[8].v, xv[2].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[9].v, xv[2].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[10].v, xv[2].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[11].v, xv[2].v, rhov[3].v);

			x0 += n_elem_per_reg * n_iter_unroll[1];
			av[0] += n_elem_per_reg * n_iter_unroll[1];
			av[1] += n_elem_per_reg * n_iter_unroll[1];
			av[2] += n_elem_per_reg * n_iter_unroll[1];
			av[3] += n_elem_per_reg * n_iter_unroll[1];
		}

		for (i = 0; i < m_viter[2]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[2] + 0 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[3] + 0 * n_elem_per_reg);

			avec[4].v = _mm256_loadu_pd(av[0] + 1 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[1] + 1 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[2] + 1 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[3] + 1 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[0].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[0].v, rhov[3].v);

			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv[1].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv[1].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv[1].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv[1].v, rhov[7].v);

			x0 += n_elem_per_reg * n_iter_unroll[2];
			av[0] += n_elem_per_reg * n_iter_unroll[2];
			av[1] += n_elem_per_reg * n_iter_unroll[2];
			av[2] += n_elem_per_reg * n_iter_unroll[2];
			av[3] += n_elem_per_reg * n_iter_unroll[2];
		}

		for (i = 0; i < m_viter[3]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[2] + 0 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[3] + 0 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[0].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[0].v, rhov[3].v);

			x0 += n_elem_per_reg * n_iter_unroll[3];
			av[0] += n_elem_per_reg * n_iter_unroll[3];
			av[1] += n_elem_per_reg * n_iter_unroll[3];
			av[2] += n_elem_per_reg * n_iter_unroll[3];
			av[3] += n_elem_per_reg * n_iter_unroll[3];
		}

		// Sum the elements of a given rho?v. This computes the sum of
		// elements within lanes and stores the sum to both elements.
		rhov[0].v = _mm256_add_pd(rhov[0].v, rhov[4].v);
		rhov[1].v = _mm256_add_pd(rhov[1].v, rhov[5].v);
		rhov[2].v = _mm256_add_pd(rhov[2].v, rhov[6].v);
		rhov[3].v = _mm256_add_pd(rhov[3].v, rhov[7].v);

		rhov[0].v = _mm256_hadd_pd(rhov[0].v, rhov[0].v);
		rhov[1].v = _mm256_hadd_pd(rhov[1].v, rhov[1].v);
		rhov[2].v = _mm256_hadd_pd(rhov[2].v, rhov[2].v);
		rhov[3].v = _mm256_hadd_pd(rhov[3].v, rhov[3].v);

		// Manually add the results from above to finish the sum.
		rho0 = rhov[0].d[0] + rhov[0].d[2];
		rho1 = rhov[1].d[0] + rhov[1].d[2];
		rho2 = rhov[2].d[0] + rhov[2].d[2];
		rho3 = rhov[3].d[0] + rhov[3].d[2];

		// Adjust for scalar subproblem.
		for (i = 0; i < 4; ++i)
		{
			m -= n_elem_per_reg * n_iter_unroll[i] * m_viter[i];
			a += n_elem_per_reg * n_iter_unroll[i] * m_viter[i] /* * inca */;
			x += n_elem_per_reg * n_iter_unroll[i] * m_viter[i] /* * incx */;
		}
	}

	// Initialize pointers for x and the b_n columns of A (rows of A^T).
	double *restrict x0 = x;
	double *restrict a0 = a + 0 * lda;
	double *restrict a1 = a + 1 * lda;
	double *restrict a2 = a + 2 * lda;
	double *restrict a3 = a + 3 * lda;

	// If there are leftover iterations, perform them with scalar code.
	for (dim_t i = 0; i < m; ++i)
	{
		const double x0c = *x0;

		const double a0c = *a0;
		const double a1c = *a1;
		const double a2c = *a2;
		const double a3c = *a3;

		rho0 += a0c * x0c;
		rho1 += a1c * x0c;
		rho2 += a2c * x0c;
		rho3 += a3c * x0c;

		x0 += incx;
		a0 += inca;
		a1 += inca;
		a2 += inca;
		a3 += inca;
	}

	// Now prepare the final rho values to output/accumulate back into
	// the y vector.

	v4df_t rho0v, y0v;

	// Insert the scalar rho values into a single vector.
	rho0v.d[0] = rho0;
	rho0v.d[1] = rho1;
	rho0v.d[2] = rho2;
	rho0v.d[3] = rho3;

	// Broadcast the alpha scalar.
	v4df_t alphav;
	alphav.v = _mm256_broadcast_sd(alpha);

	// We know at this point that alpha is nonzero; however, beta may still
	// be zero. If beta is indeed zero, we must overwrite y rather than scale
	// by beta (in case y contains NaN or Inf).
	if (PASTEMAC(d, eq0)(*beta))
	{
		// Apply alpha to the accumulated dot product in rho:
		//   y := alpha * rho
		y0v.v = _mm256_mul_pd(alphav.v, rho0v.v);
	}
	else
	{
		// Broadcast the beta scalar.
		v4df_t betav;
		betav.v = _mm256_broadcast_sd(beta);

		// Load y.
		if (incy == 1)
		{
			y0v.v = _mm256_loadu_pd(y + 0 * n_elem_per_reg);
		}
		else
		{
			y0v.d[0] = *(y + 0 * incy);
			y0v.d[1] = *(y + 1 * incy);
			y0v.d[2] = *(y + 2 * incy);
			y0v.d[3] = *(y + 3 * incy);
		}

		// Apply beta to y and alpha to the accumulated dot product in rho:
		//   y := beta * y + alpha * rho
		y0v.v = _mm256_mul_pd(betav.v, y0v.v);
		y0v.v = _mm256_fmadd_pd(alphav.v, rho0v.v, y0v.v);
	}

	if (incy == 1)
	{
		// Store the output.
		_mm256_storeu_pd((y + 0 * n_elem_per_reg), y0v.v);
	}
	else
	{
		*(y + 0 * incy) = y0v.d[0];
		*(y + 1 * incy) = y0v.d[1];
		*(y + 2 * incy) = y0v.d[2];
		*(y + 3 * incy) = y0v.d[3];
	}
}

void bli_ddotxf_zen_int_2
	(
		conj_t conjat,
		conj_t conjx,
		dim_t m,
		dim_t b_n,
		double *restrict alpha,
		double *restrict a, inc_t inca, inc_t lda,
		double *restrict x, inc_t incx,
		double *restrict beta,
		double *restrict y, inc_t incy,
		cntx_t *restrict cntx
	)
{
	const dim_t fuse_fac = 2;
	const dim_t n_elem_per_reg = 4;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if (bli_zero_dim1(b_n))
		return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if (bli_zero_dim1(m) || PASTEMAC(d, eq0)(*alpha))
	{
		bli_dscalv_zen_int10(
			BLIS_NO_CONJUGATE,
			b_n,
			beta,
			y, incy,
			cntx);
		return;
	}

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if (b_n != fuse_fac)
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

	// Intermediate variables to hold the completed dot products
	double rho0 = 0, rho1 = 0;

	if (inca == 1 && incx == 1)
	{
		const dim_t n_iter_unroll[4] = {8, 4, 2, 1};

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter[4], i, m_left = m;

		for (i = 0; i < 4; ++i)
		{
			m_viter[i] = (m_left) / (n_elem_per_reg * n_iter_unroll[i]);
			m_left = (m_left) % (n_elem_per_reg * n_iter_unroll[i]);
		}

		// Set up pointers for x and the b_n columns of A (rows of A^T).
		double *restrict x0 = x;
		double *restrict av[2];

		av[0] = a + 0 * lda;
		av[1] = a + 1 * lda;

		// Initialize b_n rho vector accumulators to zero.
		v4df_t rhov[8];

		rhov[0].v = _mm256_setzero_pd();
		rhov[1].v = _mm256_setzero_pd();
		rhov[2].v = _mm256_setzero_pd();
		rhov[3].v = _mm256_setzero_pd();
		rhov[4].v = _mm256_setzero_pd();
		rhov[5].v = _mm256_setzero_pd();
		rhov[6].v = _mm256_setzero_pd();
		rhov[7].v = _mm256_setzero_pd();

		v4df_t xv[4];
		v4df_t avec[8];

		for (i = 0; i < m_viter[0]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);
			xv[2].v = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);
			xv[3].v = _mm256_loadu_pd(x0 + 3 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[0] + 1 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[1] + 1 * n_elem_per_reg);
			avec[4].v = _mm256_loadu_pd(av[0] + 2 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[1] + 2 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[0] + 3 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[1] + 3 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[1].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[1].v, rhov[3].v);
			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv[2].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv[2].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv[3].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv[3].v, rhov[7].v);

			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 4 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 5 * n_elem_per_reg);
			xv[2].v = _mm256_loadu_pd(x0 + 6 * n_elem_per_reg);
			xv[3].v = _mm256_loadu_pd(x0 + 7 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 4 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 4 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[0] + 5 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[1] + 5 * n_elem_per_reg);
			avec[4].v = _mm256_loadu_pd(av[0] + 6 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[1] + 6 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[0] + 7 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[1] + 7 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[1].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[1].v, rhov[3].v);
			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv[2].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv[2].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv[3].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv[3].v, rhov[7].v);

			x0 += n_elem_per_reg * n_iter_unroll[0];
			av[0] += n_elem_per_reg * n_iter_unroll[0];
			av[1] += n_elem_per_reg * n_iter_unroll[0];
		}

		for (i = 0; i < m_viter[1]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);
			xv[2].v = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);
			xv[3].v = _mm256_loadu_pd(x0 + 3 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[0] + 1 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[1] + 1 * n_elem_per_reg);
			avec[4].v = _mm256_loadu_pd(av[0] + 2 * n_elem_per_reg);
			avec[5].v = _mm256_loadu_pd(av[1] + 2 * n_elem_per_reg);
			avec[6].v = _mm256_loadu_pd(av[0] + 3 * n_elem_per_reg);
			avec[7].v = _mm256_loadu_pd(av[1] + 3 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[1].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[1].v, rhov[3].v);
			rhov[4].v = _mm256_fmadd_pd(avec[4].v, xv[2].v, rhov[4].v);
			rhov[5].v = _mm256_fmadd_pd(avec[5].v, xv[2].v, rhov[5].v);
			rhov[6].v = _mm256_fmadd_pd(avec[6].v, xv[3].v, rhov[6].v);
			rhov[7].v = _mm256_fmadd_pd(avec[7].v, xv[3].v, rhov[7].v);

			x0 += n_elem_per_reg * n_iter_unroll[1];
			av[0] += n_elem_per_reg * n_iter_unroll[1];
			av[1] += n_elem_per_reg * n_iter_unroll[1];
		}

		rhov[0].v = _mm256_add_pd(rhov[0].v, rhov[4].v);
		rhov[1].v = _mm256_add_pd(rhov[1].v, rhov[5].v);
		rhov[2].v = _mm256_add_pd(rhov[2].v, rhov[6].v);
		rhov[3].v = _mm256_add_pd(rhov[3].v, rhov[7].v);

		for (i = 0; i < m_viter[2]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
			xv[1].v = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);
			avec[2].v = _mm256_loadu_pd(av[0] + 1 * n_elem_per_reg);
			avec[3].v = _mm256_loadu_pd(av[1] + 1 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);
			rhov[2].v = _mm256_fmadd_pd(avec[2].v, xv[1].v, rhov[2].v);
			rhov[3].v = _mm256_fmadd_pd(avec[3].v, xv[1].v, rhov[3].v);

			x0 += n_elem_per_reg * n_iter_unroll[2];
			av[0] += n_elem_per_reg * n_iter_unroll[2];
			av[1] += n_elem_per_reg * n_iter_unroll[2];
		}

		rhov[0].v = _mm256_add_pd(rhov[0].v, rhov[2].v);
		rhov[1].v = _mm256_add_pd(rhov[1].v, rhov[3].v);

		for (i = 0; i < m_viter[3]; ++i)
		{
			// Load the input values.
			xv[0].v = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);

			avec[0].v = _mm256_loadu_pd(av[0] + 0 * n_elem_per_reg);
			avec[1].v = _mm256_loadu_pd(av[1] + 0 * n_elem_per_reg);

			// perform: rho?v += a?v * x0v;
			rhov[0].v = _mm256_fmadd_pd(avec[0].v, xv[0].v, rhov[0].v);
			rhov[1].v = _mm256_fmadd_pd(avec[1].v, xv[0].v, rhov[1].v);

			x0 += n_elem_per_reg * n_iter_unroll[3];
			av[0] += n_elem_per_reg * n_iter_unroll[3];
			av[1] += n_elem_per_reg * n_iter_unroll[3];
		}

		// Sum the elements of a given rho?v. This computes the sum of
		// elements within lanes and stores the sum to both elements.
		rhov[0].v = _mm256_hadd_pd(rhov[0].v, rhov[0].v);
		rhov[1].v = _mm256_hadd_pd(rhov[1].v, rhov[1].v);

		// Manually add the results from above to finish the sum.
		rho0 = rhov[0].d[0] + rhov[0].d[2];
		rho1 = rhov[1].d[0] + rhov[1].d[2];

		// Adjust for scalar subproblem.
		for (i = 0; i < 4; ++i)
		{
			m -= n_elem_per_reg * n_iter_unroll[i] * m_viter[i];
			a += n_elem_per_reg * n_iter_unroll[i] * m_viter[i] /* * inca */;
			x += n_elem_per_reg * n_iter_unroll[i] * m_viter[i] /* * incx */;
		}
	}

	// Initialize pointers for x and the b_n columns of A (rows of A^T).
	double *restrict x0 = x;
	double *restrict a0 = a + 0 * lda;
	double *restrict a1 = a + 1 * lda;

	// If there are leftover iterations, perform them with scalar code.
	for (dim_t i = 0; i < m; ++i)
	{
		const double x0c = *x0;

		const double a0c = *a0;
		const double a1c = *a1;

		rho0 += a0c * x0c;
		rho1 += a1c * x0c;

		x0 += incx;
		a0 += inca;
		a1 += inca;
	}

	// Now prepare the final rho values to output/accumulate back into
	// the y vector.

	v2df_t rho0v, y0v;

	// Insert the scalar rho values into a single vector.
	rho0v.d[0] = rho0;
	rho0v.d[1] = rho1;

	// Broadcast the alpha scalar.
	v2df_t alphav;

	alphav.v = _mm_load1_pd(alpha);

	// We know at this point that alpha is nonzero; however, beta may still
	// be zero. If beta is indeed zero, we must overwrite y rather than scale
	// by beta (in case y contains NaN or Inf).
	if (PASTEMAC(d, eq0)(*beta))
	{
		// Apply alpha to the accumulated dot product in rho:
		//   y := alpha * rho
		y0v.v = _mm_mul_pd(alphav.v, rho0v.v);
	}
	else
	{
		// Broadcast the beta scalar.
		v2df_t betav;
		betav.v = _mm_load1_pd(beta);

		// Load y.
		if (incy == 1)
		{
			y0v.v = _mm_loadu_pd(y + 0 * 2);
		}
		else
		{
			y0v.d[0] = *(y + 0 * incy);
			y0v.d[1] = *(y + 1 * incy);
		}

		// Apply beta to y and alpha to the accumulated dot product in rho:
		//   y := beta * y + alpha * rho
		y0v.v = _mm_mul_pd(betav.v, y0v.v);
		y0v.v = _mm_fmadd_pd(alphav.v, rho0v.v, y0v.v);
	}

	if (incy == 1)
	{
		// Store the output.
		_mm_storeu_pd((y + 0 * 2), y0v.v);
	}
	else
	{
		*(y + 0 * incy) = y0v.d[0];
		*(y + 1 * incy) = y0v.d[1];
	}
}


