/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2017 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * Performs dotxf operation on dcomplex.
 * x and y are vectors and a is the matrix.
 * Computation is done on 6 columns at a time
 * Marches through vectors in multiple of 2.
 */
void bli_zdotxf_zen_int_6
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

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if ( b_n != 6 )
	{
		for ( dim_t i = 0; i < b_n; ++i )
		{
			dcomplex* restrict a1   = a + (0  )*inca + (i  )*lda;
			dcomplex* restrict x1   = x + (0  )*incx;
			dcomplex* restrict psi1 = y + (i  )*incy;

			bli_zdotxv_zen_int
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

	dim_t rem = m;

	double *restrict av[6];
	double *restrict x_temp = (double *)(x);

	av[0] = (double *)(a + 0 * lda);
	av[1] = (double *)(a + 1 * lda);
	av[2] = (double *)(a + 2 * lda);
	av[3] = (double *)(a + 3 * lda);
	av[4] = (double *)(a + 4 * lda);
	av[5] = (double *)(a + 5 * lda);

	dcomplex res[6];

	res[0] = res[1] = res[2] = res[3] = res[4] = res[5] = (*bli_z0);

	conj_t conjx_use = conjx;

	if (bli_is_conj(conjat))
	{
		bli_toggle_conj(&conjx_use);
	}

	if (incx == 1 && inca == 1)
	{
		rem = m % 2;
		v4df_t rhov[12], a_vec[6], xv[2], conj_mul;

		rhov[0].v = _mm256_setzero_pd();
		rhov[1].v = _mm256_setzero_pd();
		rhov[2].v = _mm256_setzero_pd();
		rhov[3].v = _mm256_setzero_pd();
		rhov[4].v = _mm256_setzero_pd();
		rhov[5].v = _mm256_setzero_pd();
		rhov[6].v = _mm256_setzero_pd();
		rhov[7].v = _mm256_setzero_pd();
		rhov[8].v = _mm256_setzero_pd();
		rhov[9].v = _mm256_setzero_pd();
		rhov[10].v = _mm256_setzero_pd();
		rhov[11].v = _mm256_setzero_pd();

		for (dim_t i = 0; (i + 1) < m; i += 2)
		{
			// Load 2 dcomplex elements from vector x
			xv[0].v = _mm256_loadu_pd(x_temp);

			// xv[1].v - R0 I0 R1 I1 => I0 I0 I1 I1
			xv[1].v = _mm256_permute_pd(xv[0].v, 15);

			// xv[0].v - R0 I0 R1 I1 => R0 R0 R1 R1
			xv[0].v = _mm256_permute_pd(xv[0].v, 0);

			a_vec[0].v = _mm256_loadu_pd((double *)(av[0]));
			a_vec[1].v = _mm256_loadu_pd((double *)(av[1]));
			a_vec[2].v = _mm256_loadu_pd((double *)(av[2]));
			a_vec[3].v = _mm256_loadu_pd((double *)(av[3]));
			a_vec[4].v = _mm256_loadu_pd((double *)(av[4]));
			a_vec[5].v = _mm256_loadu_pd((double *)(av[5]));

			// perform: rho?v += a?v * xv[0];
			rhov[0].v = _mm256_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
			rhov[6].v = _mm256_fmadd_pd(a_vec[0].v, xv[1].v, rhov[6].v);

			rhov[1].v = _mm256_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
			rhov[7].v = _mm256_fmadd_pd(a_vec[1].v, xv[1].v, rhov[7].v);

			rhov[2].v = _mm256_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
			rhov[8].v = _mm256_fmadd_pd(a_vec[2].v, xv[1].v, rhov[8].v);

			rhov[3].v = _mm256_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
			rhov[9].v = _mm256_fmadd_pd(a_vec[3].v, xv[1].v, rhov[9].v);

			rhov[4].v = _mm256_fmadd_pd(a_vec[4].v, xv[0].v, rhov[4].v);
			rhov[10].v = _mm256_fmadd_pd(a_vec[4].v, xv[1].v, rhov[10].v);

			rhov[5].v = _mm256_fmadd_pd(a_vec[5].v, xv[0].v, rhov[5].v);
			rhov[11].v = _mm256_fmadd_pd(a_vec[5].v, xv[1].v, rhov[11].v);

			av[0] += 4;
			av[1] += 4;
			av[2] += 4;
			av[3] += 4;
			av[4] += 4;
			av[5] += 4;

			x_temp += 4;
		}

		if (bli_is_noconj(conjx_use))
		{
			conj_mul.v = _mm256_setr_pd(-1, 1, -1, 1);
		}
		else
		{
			conj_mul.v = _mm256_setr_pd(1, -1, 1, -1);
		}

		/*Swapping position of real and imag component
		 * for horizontal addition to get the final
		 * dot product computation
		 * rho register are holding computation which needs
		 * to be arranged in following manner.
		 * Ra0*Ix0 | Ia0*Ix0 | Ra1*Ix1 | Ia1*Ix1
		 *             ||
		 *             \/
		 * Ia0*Ix0 | Ra0*Ix0 | Ia1*Ix1 | Ra1*Ix1
		 */
		rhov[6].v = _mm256_permute_pd(rhov[6].v, 0x05);
		rhov[7].v = _mm256_permute_pd(rhov[7].v, 0x05);
		rhov[8].v = _mm256_permute_pd(rhov[8].v, 0x05);
		rhov[9].v = _mm256_permute_pd(rhov[9].v, 0x05);
		rhov[10].v = _mm256_permute_pd(rhov[10].v, 0x05);
		rhov[11].v = _mm256_permute_pd(rhov[11].v, 0x05);

		/*
			Modifying the imag sign according to the conj value
		*/
		rhov[6].v = _mm256_mul_pd(rhov[6].v, conj_mul.v);
		rhov[7].v = _mm256_mul_pd(rhov[7].v, conj_mul.v);
		rhov[8].v = _mm256_mul_pd(rhov[8].v, conj_mul.v);
		rhov[9].v = _mm256_mul_pd(rhov[9].v, conj_mul.v);
		rhov[10].v = _mm256_mul_pd(rhov[10].v, conj_mul.v);
		rhov[11].v = _mm256_mul_pd(rhov[11].v, conj_mul.v);

		rhov[0].v = _mm256_add_pd(rhov[0].v, rhov[6].v);
		rhov[1].v = _mm256_add_pd(rhov[1].v, rhov[7].v);
		rhov[2].v = _mm256_add_pd(rhov[2].v, rhov[8].v);
		rhov[3].v = _mm256_add_pd(rhov[3].v, rhov[9].v);
		rhov[4].v = _mm256_add_pd(rhov[4].v, rhov[10].v);
		rhov[5].v = _mm256_add_pd(rhov[5].v, rhov[11].v);

		/*rho0, rho1, rho2 holds final dot product
		 * result of 6 dcomplex elements.
		 */
		rhov[0].d[0] += rhov[0].d[2];
		rhov[0].d[1] += rhov[0].d[3];

		rhov[0].d[2] = rhov[1].d[0] + rhov[1].d[2];
		rhov[0].d[3] = rhov[1].d[1] + rhov[1].d[3];

		rhov[1].d[0] = rhov[2].d[0] + rhov[2].d[2];
		rhov[1].d[1] = rhov[2].d[1] + rhov[2].d[3];

		rhov[1].d[2] = rhov[3].d[0] + rhov[3].d[2];
		rhov[1].d[3] = rhov[3].d[1] + rhov[3].d[3];

		rhov[2].d[0] = rhov[4].d[0] + rhov[4].d[2];
		rhov[2].d[1] = rhov[4].d[1] + rhov[4].d[3];

		rhov[2].d[2] = rhov[5].d[0] + rhov[5].d[2];
		rhov[2].d[3] = rhov[5].d[1] + rhov[5].d[3];

		/*
			Computed dot product result is being stored
			in temp buffer r for further computation.
		*/
		_mm256_storeu_pd((double *)res, rhov[0].v);
		_mm256_storeu_pd((double *)(res + 2), rhov[1].v);
		_mm256_storeu_pd((double *)(res + 4), rhov[2].v);
	}

	// This section will have the whole of compute when incx != 1 || inca != 1
	if (rem)
	{
		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur later,
		// especially if BLIS is compiled with -mfpmath=sse).
		_mm256_zeroupper();

		v2df_t rhov[12], a_vec[6], xv[2], conj_mul;

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

		for (dim_t i = 0; i < rem; i++)
		{
			// Load 2 dcomplex elements from vector x
			xv[0].v = _mm_loadu_pd(x_temp);

			// xv[1].v - R0 I0 R1 I1 => I0 I0 I1 I1
			xv[1].v = _mm_permute_pd(xv[0].v, 0b11);

			// xv[0].v - R0 I0 R1 I1 => R0 R0 R1 R1
			xv[0].v = _mm_permute_pd(xv[0].v, 0b00);

			a_vec[0].v = _mm_loadu_pd((double *)(av[0]));
			a_vec[1].v = _mm_loadu_pd((double *)(av[1]));
			a_vec[2].v = _mm_loadu_pd((double *)(av[2]));
			a_vec[3].v = _mm_loadu_pd((double *)(av[3]));
			a_vec[4].v = _mm_loadu_pd((double *)(av[4]));
			a_vec[5].v = _mm_loadu_pd((double *)(av[5]));

			// perform: rho?v += a?v * xv[0];
			rhov[0].v = _mm_fmadd_pd(a_vec[0].v, xv[0].v, rhov[0].v);
			rhov[6].v = _mm_fmadd_pd(a_vec[0].v, xv[1].v, rhov[6].v);

			rhov[1].v = _mm_fmadd_pd(a_vec[1].v, xv[0].v, rhov[1].v);
			rhov[7].v = _mm_fmadd_pd(a_vec[1].v, xv[1].v, rhov[7].v);

			rhov[2].v = _mm_fmadd_pd(a_vec[2].v, xv[0].v, rhov[2].v);
			rhov[8].v = _mm_fmadd_pd(a_vec[2].v, xv[1].v, rhov[8].v);

			rhov[3].v = _mm_fmadd_pd(a_vec[3].v, xv[0].v, rhov[3].v);
			rhov[9].v = _mm_fmadd_pd(a_vec[3].v, xv[1].v, rhov[9].v);

			rhov[4].v = _mm_fmadd_pd(a_vec[4].v, xv[0].v, rhov[4].v);
			rhov[10].v = _mm_fmadd_pd(a_vec[4].v, xv[1].v, rhov[10].v);

			rhov[5].v = _mm_fmadd_pd(a_vec[5].v, xv[0].v, rhov[5].v);
			rhov[11].v = _mm_fmadd_pd(a_vec[5].v, xv[1].v, rhov[11].v);

			av[0] += 2 * inca;
			av[1] += 2 * inca;
			av[2] += 2 * inca;
			av[3] += 2 * inca;
			av[4] += 2 * inca;
			av[5] += 2 * inca;

			x_temp += 2 * incx;
		}

		if (bli_is_noconj(conjx_use))
		{
			conj_mul.v = _mm_setr_pd(-1, 1);
		}
		else
		{
			conj_mul.v = _mm_setr_pd(1, -1);
		}

		rhov[6].v = _mm_permute_pd(rhov[6].v, 0b01);
		rhov[7].v = _mm_permute_pd(rhov[7].v, 0b01);
		rhov[8].v = _mm_permute_pd(rhov[8].v, 0b01);
		rhov[9].v = _mm_permute_pd(rhov[9].v, 0b01);
		rhov[10].v = _mm_permute_pd(rhov[10].v, 0b01);
		rhov[11].v = _mm_permute_pd(rhov[11].v, 0b01);

		/*
			Modifying the imag sign according to the conj value
		*/
		rhov[6].v = _mm_mul_pd(rhov[6].v, conj_mul.v);
		rhov[7].v = _mm_mul_pd(rhov[7].v, conj_mul.v);
		rhov[8].v = _mm_mul_pd(rhov[8].v, conj_mul.v);
		rhov[9].v = _mm_mul_pd(rhov[9].v, conj_mul.v);
		rhov[10].v = _mm_mul_pd(rhov[10].v, conj_mul.v);
		rhov[11].v = _mm_mul_pd(rhov[11].v, conj_mul.v);

		rhov[0].v = _mm_add_pd(rhov[0].v, rhov[6].v);
		rhov[1].v = _mm_add_pd(rhov[1].v, rhov[7].v);
		rhov[2].v = _mm_add_pd(rhov[2].v, rhov[8].v);
		rhov[3].v = _mm_add_pd(rhov[3].v, rhov[9].v);
		rhov[4].v = _mm_add_pd(rhov[4].v, rhov[10].v);
		rhov[5].v = _mm_add_pd(rhov[5].v, rhov[11].v);

		rhov[6].v = _mm_loadu_pd((double *)(res));
		rhov[7].v = _mm_loadu_pd((double *)(res + 1));
		rhov[8].v = _mm_loadu_pd((double *)(res + 2));
		rhov[9].v = _mm_loadu_pd((double *)(res + 3));
		rhov[10].v = _mm_loadu_pd((double *)(res + 4));
		rhov[11].v = _mm_loadu_pd((double *)(res + 5));

		rhov[0].v = _mm_add_pd(rhov[0].v, rhov[6].v);
		rhov[1].v = _mm_add_pd(rhov[1].v, rhov[7].v);
		rhov[2].v = _mm_add_pd(rhov[2].v, rhov[8].v);
		rhov[3].v = _mm_add_pd(rhov[3].v, rhov[9].v);
		rhov[4].v = _mm_add_pd(rhov[4].v, rhov[10].v);
		rhov[5].v = _mm_add_pd(rhov[5].v, rhov[11].v);

		/*
			Computed dot product result is being stored
			in temp buffer r for further computation.
		*/
		_mm_storeu_pd((double *)res, rhov[0].v);
		_mm_storeu_pd((double *)(res + 1), rhov[1].v);
		_mm_storeu_pd((double *)(res + 2), rhov[2].v);
		_mm_storeu_pd((double *)(res + 3), rhov[3].v);
		_mm_storeu_pd((double *)(res + 4), rhov[4].v);
		_mm_storeu_pd((double *)(res + 5), rhov[5].v);

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur later,
		// especially if BLIS is compiled with -mfpmath=sse).
		_mm256_zeroupper();
	}

	// Multiplying 'A' * 'x' by 'alpha'
	__m256d alpha_r, alpha_i, temp_v[3];
	v4df_t rhov[3];

	rhov[0].v = _mm256_loadu_pd((double *)(res));
	rhov[1].v = _mm256_loadu_pd((double *)(res + 2));
	rhov[2].v = _mm256_loadu_pd((double *)(res + 4));

	if (bli_is_conj(conjat))
	{
		__m256d conj_mul = _mm256_setr_pd(1, -1, 1, -1);

		rhov[0].v = _mm256_mul_pd(rhov[0].v, conj_mul);
		rhov[1].v = _mm256_mul_pd(rhov[1].v, conj_mul);
		rhov[2].v = _mm256_mul_pd(rhov[2].v, conj_mul);
	}

	alpha_r = _mm256_broadcast_sd(&((*alpha).real));
	alpha_i = _mm256_broadcast_sd(&((*alpha).imag));

	temp_v[0] = _mm256_mul_pd(rhov[0].v, alpha_i);
	temp_v[1] = _mm256_mul_pd(rhov[1].v, alpha_i);
	temp_v[2] = _mm256_mul_pd(rhov[2].v, alpha_i);

	temp_v[0] = _mm256_permute_pd(temp_v[0], 0b0101);
	temp_v[1] = _mm256_permute_pd(temp_v[1], 0b0101);
	temp_v[2] = _mm256_permute_pd(temp_v[2], 0b0101);

	rhov[0].v = _mm256_fmaddsub_pd(rhov[0].v, alpha_r, temp_v[0]);
	rhov[1].v = _mm256_fmaddsub_pd(rhov[1].v, alpha_r, temp_v[1]);
	rhov[2].v = _mm256_fmaddsub_pd(rhov[2].v, alpha_r, temp_v[2]);

	// When 'beta' is not zero we need to multiply scale 'y' by 'beta'
	if (!PASTEMAC(z, eq0)(*beta))
	{
		v4df_t yv[3];
		__m256d beta_r, beta_i;

		beta_r = _mm256_broadcast_sd(&((*beta).real));
		beta_i = _mm256_broadcast_sd(&((*beta).imag));

		if (incy == 1)
		{
			yv[0].v = _mm256_loadu_pd((double *)(y));
			yv[1].v = _mm256_loadu_pd((double *)(y + 2));
			yv[2].v = _mm256_loadu_pd((double *)(y + 4));
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

			yv[1].d[0] = (*(y + 2 * incy)).real;
			yv[1].d[1] = (*(y + 2 * incy)).imag;
			yv[1].d[2] = (*(y + 3 * incy)).real;
			yv[1].d[3] = (*(y + 3 * incy)).imag;

			yv[2].d[0] = (*(y + 4 * incy)).real;
			yv[2].d[1] = (*(y + 4 * incy)).imag;
			yv[2].d[2] = (*(y + 5 * incy)).real;
			yv[2].d[3] = (*(y + 5 * incy)).imag;
		}

		temp_v[0] = _mm256_mul_pd(yv[0].v, beta_i);
		temp_v[1] = _mm256_mul_pd(yv[1].v, beta_i);
		temp_v[2] = _mm256_mul_pd(yv[2].v, beta_i);

		temp_v[0] = _mm256_permute_pd(temp_v[0], 0b0101);
		temp_v[1] = _mm256_permute_pd(temp_v[1], 0b0101);
		temp_v[2] = _mm256_permute_pd(temp_v[2], 0b0101);

		yv[0].v = _mm256_fmaddsub_pd(yv[0].v, beta_r, temp_v[0]);
		yv[1].v = _mm256_fmaddsub_pd(yv[1].v, beta_r, temp_v[1]);
		yv[2].v = _mm256_fmaddsub_pd(yv[2].v, beta_r, temp_v[2]);

		// Here we 'rhov' has 'alpha' * 'A' * 'x' that is added with 'y'
		rhov[0].v = _mm256_add_pd(yv[0].v, rhov[0].v);
		rhov[1].v = _mm256_add_pd(yv[1].v, rhov[1].v);
		rhov[2].v = _mm256_add_pd(yv[2].v, rhov[2].v);
	}

	if (incy == 1)
	{
		_mm256_storeu_pd((double *)y, rhov[0].v);
		_mm256_storeu_pd((double *)(y + 2), rhov[1].v);
		_mm256_storeu_pd((double *)(y + 4), rhov[2].v);
	}
	else
	{
		(*(y + 0 * incy)).real = rhov[0].d[0];
		(*(y + 0 * incy)).imag = rhov[0].d[1];
		(*(y + 1 * incy)).real = rhov[0].d[2];
		(*(y + 1 * incy)).imag = rhov[0].d[3];

		(*(y + 2 * incy)).real = rhov[1].d[0];
		(*(y + 2 * incy)).imag = rhov[1].d[1];
		(*(y + 3 * incy)).real = rhov[1].d[2];
		(*(y + 3 * incy)).imag = rhov[1].d[3];

		(*(y + 4 * incy)).real = rhov[2].d[0];
		(*(y + 4 * incy)).imag = rhov[2].d[1];
		(*(y + 5 * incy)).real = rhov[2].d[2];
		(*(y + 5 * incy)).imag = rhov[2].d[3];
	}
}

/**
 * Performs dotxf operation on scomplex.
 * x and y are vectors and a is the matrix.
 * Computation is done on 6 columns at a time
 * Marches through vectors in multiple of 4 and 2.
 */
void bli_cdotxf_zen_int_6
	(
		 conj_t conjat,
		 conj_t conjx,
		 dim_t m,
		 dim_t b_n,
		 scomplex* restrict alpha,
		 scomplex* restrict a, inc_t inca, inc_t lda,
		 scomplex* restrict x, inc_t incx,
		 scomplex* restrict beta,
		 scomplex* restrict y, inc_t incy,
		 cntx_t* restrict cntx
	)
{
        if ( (inca == 1) && (incx == 1) && (incy == 1) && (b_n == 6) )
        {
                /* Temporary rho buffer holds computed dot product result */
                scomplex r[ 6 ];

                /* If beta is zero, clear y. Otherwise, scale by beta. */
                if ( PASTEMAC(c,eq0)( *beta ) )
                {
                        for ( dim_t i = 0; i < 6; ++i )
			{
				PASTEMAC(c,set0s)( y[i] );
			}
                }
                else
                {
                        for ( dim_t i = 0; i < 6; ++i )
			{
				PASTEMAC(c,scals)( *beta, y[i] );
			}
                }

                /* If the vectors are empty or if alpha is zero, return early. */
                if ( bli_zero_dim1( m ) || PASTEMAC(c,eq0)( *alpha ) ) return;

                /* Initialize r vector to 0. */
                for ( dim_t i = 0; i < 6; ++i ) PASTEMAC(c,set0s)( r[i] );

                /* If a must be conjugated, we do so indirectly by first toggling the
                   effective conjugation of x and then conjugating the resulting do
                   products. */
                conj_t conjx_use = conjx;

                if ( bli_is_conj( conjat ) )
                        bli_toggle_conj( &conjx_use );

                dim_t iter = m / 2;
                dim_t iter4 = m / 4;
                dim_t rem = m % 2;
                dim_t i = 0;
                if(iter)
                {
                        if(iter4)
                        {
                                /* Setting rho vectors to 0 */
                                __m256 rho0v; rho0v = _mm256_setzero_ps();
                                __m256 rho1v; rho1v = _mm256_setzero_ps();
                                __m256 rho2v; rho2v = _mm256_setzero_ps();
                                __m256 rho3v; rho3v = _mm256_setzero_ps();
                                __m256 rho4v; rho4v = _mm256_setzero_ps();
                                __m256 rho5v; rho5v = _mm256_setzero_ps();

                                __m256 rho6v; rho6v = _mm256_setzero_ps();
                                __m256 rho7v; rho7v = _mm256_setzero_ps();
                                __m256 rho8v; rho8v = _mm256_setzero_ps();
                                __m256 rho9v; rho9v = _mm256_setzero_ps();
                                __m256 rho10v; rho10v = _mm256_setzero_ps();
                                __m256 rho11v; rho11v = _mm256_setzero_ps();
                                /* Holds 2 dcomplex element of x vector
                                 * for computing dot product with A tile
                                 */
                                __m256 x0v, x1v;
                                /* Holds 2x6 tile of matrix A */
				__m256 a0v, a1v, a2v, a3v, a4v, a5v;
				/**
				 * Since complex datatype multiplication is
				 * being held in two sets of rho vectors.
				 * Where first set holds the computaion with
				 * real part of vector x and other holds
				 * imaginary part of vector x.
				 * For final computation, based on conj sign
				 * of imaginary component needs to be toggled.
				 */
				__m256 no_conju = _mm256_setr_ps(-1, 1, -1, 1, -1, 1, -1, 1);
                                __m256 conju = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);

				// March through vectos in multiple of 4.
                                for ( ; (i+3) < m; i+=4)
                                {
                                        /*Load 4 scomplex elements from vector x*/
                                        x0v = _mm256_loadu_ps( (float *) (x + i) );
                                        /* x1v.v holds imaginary part of dcomplex
                                         * elements from vector x
                                         */
                                        x1v = _mm256_permute_ps( x0v, 0xf5 );
                                        /* x1v.v holds real part of dcomplex
                                         * elements from vector x
                                         */
                                        x0v = _mm256_permute_ps( x0v, 0xa0);
                                        /* x1v.v holds imag part of dcomplex
                                        Load 4x6 tile of matrix A*/
					a0v = _mm256_loadu_ps( (float *)(a + i + 0 * lda));
                                        a1v = _mm256_loadu_ps( (float *)(a + i + 1 * lda));
                                        a2v = _mm256_loadu_ps( (float *)(a + i + 2 * lda));
                                        a3v = _mm256_loadu_ps( (float *)(a + i + 3 * lda));
                                        a4v = _mm256_loadu_ps( (float *)(a + i + 4 * lda));
                                        a5v = _mm256_loadu_ps( (float *)(a + i + 5 * lda));

                                        // perform: rho?v += a?v * x0v;

                                        rho0v = _mm256_fmadd_ps( a0v, x0v, rho0v );
                                        rho6v = _mm256_fmadd_ps( a0v, x1v, rho6v );

                                        rho1v = _mm256_fmadd_ps( a1v, x0v, rho1v );
                                        rho7v = _mm256_fmadd_ps( a1v, x1v, rho7v );

                                        rho2v = _mm256_fmadd_ps( a2v, x0v, rho2v );
                                        rho8v = _mm256_fmadd_ps( a2v, x1v, rho8v );

                                        rho3v = _mm256_fmadd_ps( a3v, x0v, rho3v );
                                        rho9v = _mm256_fmadd_ps( a3v, x1v, rho9v );

                                        rho4v = _mm256_fmadd_ps( a4v, x0v, rho4v );
                                        rho10v = _mm256_fmadd_ps( a4v, x1v, rho10v );

                                        rho5v = _mm256_fmadd_ps( a5v, x0v, rho5v );
                                        rho11v = _mm256_fmadd_ps( a5v, x1v, rho11v );
                                }


                                /*Swapping position of real and imag component
                                 * for horizontal addition to get the final
                                 * dot product computation
				 * rho register are holding computation which needs
				 * to be arranged in following manner.
				 * Ra0*Ix0 | Ia0*Ix0 | Ra1*Ix1 | Ia1*Ix1
				 *             ||
				 *             \/
				 * Ia0*Ix0 | Ra0*Ix0 | Ia1*Ix1 | Ra1*Ix1
                                 */

                                rho6v = _mm256_permute_ps(rho6v, 0xb1);
                                rho7v = _mm256_permute_ps(rho7v, 0xb1);
                                rho8v = _mm256_permute_ps(rho8v, 0xb1);
                                rho9v = _mm256_permute_ps(rho9v, 0xb1);
                                rho10v = _mm256_permute_ps(rho10v, 0xb1);
                                rho11v = _mm256_permute_ps(rho11v, 0xb1);

                                /*Negating imaginary part for computing
                                 * the final result of dcomplex multiplication*/
                                if ( bli_is_noconj( conjx_use ) )
                                {
                                        rho6v = _mm256_mul_ps(rho6v, no_conju);
                                        rho7v = _mm256_mul_ps(rho7v, no_conju);
                                        rho8v = _mm256_mul_ps(rho8v, no_conju);
                                        rho9v = _mm256_mul_ps(rho9v, no_conju);
                                        rho10v = _mm256_mul_ps(rho10v, no_conju);
                                        rho11v = _mm256_mul_ps(rho11v, no_conju);
                                }
				else
                                {

                                        rho6v = _mm256_mul_ps(rho6v, conju);
                                        rho7v = _mm256_mul_ps(rho7v, conju);
                                        rho8v = _mm256_mul_ps(rho8v, conju);
                                        rho9v = _mm256_mul_ps(rho9v, conju);
                                        rho10v = _mm256_mul_ps(rho10v, conju);
                                        rho11v = _mm256_mul_ps(rho11v, conju);

                                }

                                rho0v = _mm256_add_ps(rho0v, rho6v);
                                rho1v = _mm256_add_ps(rho1v, rho7v);
                                rho2v = _mm256_add_ps(rho2v, rho8v);
                                rho3v = _mm256_add_ps(rho3v, rho9v);
                                rho4v = _mm256_add_ps(rho4v, rho10v);
                                rho5v = _mm256_add_ps(rho5v, rho11v);

				/**
				 * Horizontal addition of rho elements
				 * for computing final dotxf result.
				 * ptr pointer addresses all 6 rho
				 * register one by one and store the
				 * computed result into r buffer.
				 */
                                scomplex *ptr = (scomplex *)&rho0v;
                                for(dim_t i = 0; i < 4; i++)
                                {
                                        r[0].real += ptr[i].real;
                                        r[0].imag += ptr[i].imag;
                                }
                                ptr = (scomplex *)&rho1v;
                                for(dim_t i = 0; i < 4; i++)
                                {
                                        r[1].real += ptr[i].real;
                                        r[1].imag += ptr[i].imag;
                                }
                                ptr = (scomplex *)&rho2v;
                                for(dim_t i = 0; i < 4; i++)
                                {
                                        r[2].real += ptr[i].real;
                                        r[2].imag += ptr[i].imag;
                                }
                                ptr = (scomplex *)&rho3v;
                                for(dim_t i = 0; i < 4; i++)
                                {
                                        r[3].real += ptr[i].real;
                                        r[3].imag += ptr[i].imag;
                                }
                                ptr = (scomplex *)&rho4v;
                                for(dim_t i = 0; i < 4; i++)
                                {
                                        r[4].real += ptr[i].real;
                                        r[4].imag += ptr[i].imag;
                                }
                                ptr = (scomplex *)&rho5v;
                                for(dim_t i = 0; i < 4; i++)
                                {
                                        r[5].real += ptr[i].real;
                                        r[5].imag += ptr[i].imag;
                                }
                        }
			// March through vectos in multiple of 2.
                        if(i+1 < m)
                        {
                                /* Setting rho vectors to 0 */
                                __m128 rho0v; rho0v = _mm_setzero_ps();
                                __m128 rho1v; rho1v = _mm_setzero_ps();
                                __m128 rho2v; rho2v = _mm_setzero_ps();
                                __m128 rho3v; rho3v = _mm_setzero_ps();
                                __m128 rho4v; rho4v = _mm_setzero_ps();
                                __m128 rho5v; rho5v = _mm_setzero_ps();

                                __m128 rho6v; rho6v = _mm_setzero_ps();
                                __m128 rho7v; rho7v = _mm_setzero_ps();
                                __m128 rho8v; rho8v = _mm_setzero_ps();
                                __m128 rho9v; rho9v = _mm_setzero_ps();
                                __m128 rho10v; rho10v = _mm_setzero_ps();
                                __m128 rho11v; rho11v = _mm_setzero_ps();
                                /* Holds 2 dcomplex element of x vector
                                 * for computing dot product with A tile
                                 */
                                __m128 x0v, x1v;
                                /* Holds 2x6 tile of matrix A */
                                __m128 a0v, a1v, a2v, a3v, a4v, a5v;
				/**
				 * Since complex datatype multiplication is
				 * being held in two sets of rho vectors.
				 * Where first set holds the computaion with
				 * real part of vector x and other holds
				 * imaginary part of vector x.
				 * For final computation, based on conj sign
				 * of imaginary component needs to be toggled.
				 */
                                __m128 no_conju = _mm_setr_ps(-1, 1, -1, 1);
                                __m128 conju = _mm_setr_ps(1, -1, 1, -1);

                                for ( ; (i+1) < m; i+=2)
                                {
                                        /*Load 4 scomplex elements from vector x*/
                                        x0v = _mm_loadu_ps( (float *)(x + i) );
                                        /* x1v.v holds imaginary part of dcomplex
                                         * elements from vector x
                                         */
                                        x1v = _mm_permute_ps( x0v,  0xf5 );
                                        /* x1v.v holds real part of dcomplex
                                         * elements from vector x
                                         */
                                        x0v = _mm_permute_ps( x0v, 0xa0);
                                        /* x1v.v holds imag part of dcomplex
                                        Load 4x6 tile of matrix A*/

                                        a0v = _mm_loadu_ps( (float *)(a + i + 0 * lda));
                                        a1v = _mm_loadu_ps( (float *)(a + i + 1 * lda));
                                        a2v = _mm_loadu_ps( (float *)(a + i + 2 * lda));
                                        a3v = _mm_loadu_ps( (float *)(a + i + 3 * lda));
                                        a4v = _mm_loadu_ps( (float *)(a + i + 4 * lda));
                                        a5v = _mm_loadu_ps( (float *)(a + i + 5 * lda));

                                        // perform: rho?v += a?v * x0v;

                                        rho0v = _mm_fmadd_ps( a0v, x0v, rho0v );
                                        rho6v = _mm_fmadd_ps( a0v, x1v, rho6v );

                                        rho1v = _mm_fmadd_ps( a1v, x0v, rho1v );
                                        rho7v = _mm_fmadd_ps( a1v, x1v, rho7v );

                                        rho2v = _mm_fmadd_ps( a2v, x0v, rho2v );
                                        rho8v = _mm_fmadd_ps( a2v, x1v, rho8v );

                                        rho3v = _mm_fmadd_ps( a3v, x0v, rho3v );
                                        rho9v = _mm_fmadd_ps( a3v, x1v, rho9v );

                                        rho4v = _mm_fmadd_ps( a4v, x0v, rho4v );
                                        rho10v = _mm_fmadd_ps( a4v, x1v, rho10v );

                                        rho5v = _mm_fmadd_ps( a5v, x0v, rho5v );
					rho11v = _mm_fmadd_ps( a5v, x1v, rho11v );
				}
				/*Swapping position of real and imag component
				 * for horizontal addition to get the final
				 * dot product computation
				 * rho register are holding computation which needs
				 * to be arranged in following manner.
				 * Ra0*Ix0 | Ia0*Ix0 | Ra1*Ix1 | Ia1*Ix1
				 *             ||
				 *             \/
				 * Ia0*Ix0 | Ra0*Ix0 | Ia1*Ix1 | Ra1*Ix1
				 */
				rho6v = _mm_permute_ps(rho6v, 0xb1);
				rho7v = _mm_permute_ps(rho7v, 0xb1);
				rho8v = _mm_permute_ps(rho8v, 0xb1);
				rho9v = _mm_permute_ps(rho9v, 0xb1);
				rho10v = _mm_permute_ps(rho10v, 0xb1);
				rho11v = _mm_permute_ps(rho11v, 0xb1);

				/*Negating imaginary part for computing
				 * the final result of dcomplex multiplication*/
				if ( bli_is_noconj( conjx_use ) )
				{

					rho6v = _mm_mul_ps(rho6v, no_conju);
					rho7v = _mm_mul_ps(rho7v, no_conju);
					rho8v = _mm_mul_ps(rho8v, no_conju);
					rho9v = _mm_mul_ps(rho9v, no_conju);
					rho10v = _mm_mul_ps(rho10v, no_conju);
					rho11v = _mm_mul_ps(rho11v, no_conju);
				}
				else
				{
					rho6v = _mm_mul_ps(rho6v, conju);
					rho7v = _mm_mul_ps(rho7v, conju);
					rho8v = _mm_mul_ps(rho8v, conju);
					rho9v = _mm_mul_ps(rho9v, conju);
					rho10v = _mm_mul_ps(rho10v, conju);
					rho11v = _mm_mul_ps(rho11v, conju);
				}

				rho0v = _mm_add_ps(rho0v, rho6v);
				rho1v = _mm_add_ps(rho1v, rho7v);
				rho2v = _mm_add_ps(rho2v, rho8v);
				rho3v = _mm_add_ps(rho3v, rho9v);
				rho4v = _mm_add_ps(rho4v, rho10v);
				rho5v = _mm_add_ps(rho5v, rho11v);

				/**
				 * Horizontal addition of rho elements
				 * for computing final dotxf result.
				 * ptr pointer addresses all 6 rho
				 * register one by one and store the
				 * computed result into r buffer.
				 */
				scomplex *ptr = (scomplex *)&rho0v;
				for(dim_t i = 0; i < 2; i++)
				{
					r[0].real += ptr[i].real;
					r[0].imag += ptr[i].imag;
				}
				ptr = (scomplex *)&rho1v;
				for(dim_t i = 0; i < 2; i++)
				{
					r[1].real += ptr[i].real;
					r[1].imag += ptr[i].imag;
				}
				ptr = (scomplex *)&rho2v;
				for(dim_t i = 0; i < 2; i++)
				{
					r[2].real += ptr[i].real;
					r[2].imag += ptr[i].imag;
				}
				ptr = (scomplex *)&rho3v;
				for(dim_t i = 0; i < 2; i++)
				{
					r[3].real += ptr[i].real;
					r[3].imag += ptr[i].imag;
				}
				ptr = (scomplex *)&rho4v;
				for(dim_t i = 0; i < 2; i++)
				{
					r[4].real += ptr[i].real;
					r[4].imag += ptr[i].imag;
				}
				ptr = (scomplex *)&rho5v;
				for(dim_t i = 0; i < 2; i++)
				{
					r[5].real += ptr[i].real;
					r[5].imag += ptr[i].imag;
				}
			}
		}
                /*handles remainder cases*/
                if(rem)
                {
                        if ( bli_is_noconj( conjx_use ) )
                        {

                                PRAGMA_SIMD
                                        for(dim_t p = 0; p < 6 ; p++)
                                        {
                                                PASTEMAC(c,axpys)( a[i + p*lda], x[i], r[p] );
                                        }
                        }
                        else
                        {
                                PRAGMA_SIMD
                                        for(dim_t p = 0; p < 6 ; p++)
                                        {
                                                PASTEMAC(c,axpyjs)( a[i + p*lda], x[i], r[p] );
                                        }

                        }
                }

                if ( bli_is_conj( conjat ) )
		{
                        for ( dim_t i = 0; i < 6; ++i )
			{
				PASTEMAC(c,conjs)( r[i] );
			}
		}

                /*scaling dot product result with alpha and
                 * adding the result to vector
                 */
                for ( dim_t i = 0; i < 6; ++i )
                {
                        PASTEMAC(c,axpys)( *alpha, r[i], y[i] );
                }
	}
	else
	{
		/* Query the context for the kernel function pointer. */
		const num_t              dt     = PASTEMAC(c,type);
		PASTECH(c,dotxv_ker_ft) kfp_dv
			=
			bli_cntx_get_l1v_ker_dt( dt, BLIS_DOTXV_KER, cntx );

		for ( dim_t i = 0; i < b_n; ++i )
		{
			scomplex* restrict a1   = a + (0  )*inca + (i  )*lda;
	                scomplex* restrict x1   = x + (0  )*incx;
	                scomplex* restrict psi1 = y + (i  )*incy;

	                kfp_dv
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
	}
}

