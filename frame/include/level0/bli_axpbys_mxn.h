/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#ifndef BLIS_AXPBYS_MXN_H
#define BLIS_AXPBYS_MXN_H

// axpbys_mxn

BLIS_INLINE void bli_saxpbys_mxn( const dim_t m, const dim_t n, float* restrict alpha,
                                                                float* restrict x, const inc_t rs_x, const inc_t cs_x,
                                                                float* restrict beta,
                                                                float* restrict y, const inc_t rs_y, const inc_t cs_y )
{
	// If beta is zero, overwrite y with x (in case y has infs or NaNs).
	if ( bli_seq0( *beta ) )
	{
		bli_sscal2s_mxn( BLIS_NO_CONJUGATE, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y );
		return;
	}

#ifdef BLIS_ENABLE_CR_CASES
	if ( rs_x == 1 && rs_y == 1 )
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_saxpbys( *alpha, *(x + ii + jj*cs_x), *beta, *(y + ii + jj*cs_y) );
	}
	else if ( cs_x == 1 && cs_y == 1 )
	{
		for ( dim_t ii = 0; ii < m; ++ii )
		for ( dim_t jj = 0; jj < n; ++jj )
		bli_saxpbys( *alpha, *(x + ii*rs_x + jj), *beta, *(y + ii*rs_y + jj) );
	}
	else
#endif
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_saxpbys( *alpha, *(x + ii*rs_x + jj*cs_x), *beta, *(y + ii*rs_y + jj*cs_y) );
	}
}

BLIS_INLINE void bli_daxpbys_mxn( const dim_t m, const dim_t n, double* restrict alpha,
                                                                double* restrict x, const inc_t rs_x, const inc_t cs_x,
                                                                double* restrict beta,
                                                                double* restrict y, const inc_t rs_y, const inc_t cs_y )
{
	// If beta is zero, overwrite y with x (in case y has infs or NaNs).
	if ( bli_deq0( *beta ) )
	{
		bli_dscal2s_mxn( BLIS_NO_CONJUGATE, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y );
		return;
	}

#ifdef BLIS_ENABLE_CR_CASES
	if ( rs_x == 1 && rs_y == 1 )
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_daxpbys( *alpha, *(x + ii + jj*cs_x), *beta, *(y + ii + jj*cs_y) );
	}
	else if ( cs_x == 1 && cs_y == 1 )
	{
		for ( dim_t ii = 0; ii < m; ++ii )
		for ( dim_t jj = 0; jj < n; ++jj )
		bli_daxpbys( *alpha, *(x + ii*rs_x + jj), *beta, *(y + ii*rs_y + jj) );
	}
	else
#endif
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_daxpbys( *alpha, *(x + ii*rs_x + jj*cs_x), *beta, *(y + ii*rs_y + jj*cs_y) );
	}
}

BLIS_INLINE void bli_caxpbys_mxn( const dim_t m, const dim_t n, scomplex* restrict alpha,
                                                                scomplex* restrict x, const inc_t rs_x, const inc_t cs_x,
                                                                scomplex* restrict beta,
                                                                scomplex* restrict y, const inc_t rs_y, const inc_t cs_y )
{
	// If beta is zero, overwrite y with x (in case y has infs or NaNs).
	if ( bli_ceq0( *beta ) )
	{
		bli_cscal2s_mxn( BLIS_NO_CONJUGATE, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y );
		return;
	}

#ifdef BLIS_ENABLE_CR_CASES
	if ( rs_x == 1 && rs_y == 1 )
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_caxpbys( *alpha, *(x + ii + jj*cs_x), *beta, *(y + ii + jj*cs_y) );
	}
	else if ( cs_x == 1 && cs_y == 1 )
	{
		for ( dim_t ii = 0; ii < m; ++ii )
		for ( dim_t jj = 0; jj < n; ++jj )
		bli_caxpbys( *alpha, *(x + ii*rs_x + jj), *beta, *(y + ii*rs_y + jj) );
	}
	else
#endif
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_caxpbys( *alpha, *(x + ii*rs_x + jj*cs_x), *beta, *(y + ii*rs_y + jj*cs_y) );
	}
}

BLIS_INLINE void bli_zaxpbys_mxn( const dim_t m, const dim_t n, dcomplex* restrict alpha,
                                                                dcomplex* restrict x, const inc_t rs_x, const inc_t cs_x,
                                                                dcomplex* restrict beta,
                                                                dcomplex* restrict y, const inc_t rs_y, const inc_t cs_y )
{
	// If beta is zero, overwrite y with x (in case y has infs or NaNs).
	if ( bli_zeq0( *beta ) )
	{
		bli_zscal2s_mxn( BLIS_NO_CONJUGATE, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y );
		return;
	}

#ifdef BLIS_ENABLE_CR_CASES
	if ( rs_x == 1 && rs_y == 1 )
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_zaxpbys( *alpha, *(x + ii + jj*cs_x), *beta, *(y + ii + jj*cs_y) );
	}
	else if ( cs_x == 1 && cs_y == 1 )
	{
		for ( dim_t ii = 0; ii < m; ++ii )
		for ( dim_t jj = 0; jj < n; ++jj )
		bli_zaxpbys( *alpha, *(x + ii*rs_x + jj), *beta, *(y + ii*rs_y + jj) );
	}
	else
#endif
	{
		for ( dim_t jj = 0; jj < n; ++jj )
		for ( dim_t ii = 0; ii < m; ++ii )
		bli_zaxpbys( *alpha, *(x + ii*rs_x + jj*cs_x), *beta, *(y + ii*rs_y + jj*cs_y) );
	}
}


#endif
