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

#include "blis.h"

// Optional OpenMP column-parallel rank-1 update (ger). Each column of A is an
// independent axpyv (A[:,j] += alpha*conj(y[j]) * x), so we split the columns
// across threads -- disjoint output, no reduction. The per-column kernel is the
// raw axpyv micro-kernel (not the threaded tapi wrapper), so there is no
// nesting. Enabled via BLIS_ENABLE_L1_OPENMP.
#ifdef BLIS_ENABLE_L1_OPENMP
#include <omp.h>
#ifndef BLIS_L2_MT_THRESHOLD
#define BLIS_L2_MT_THRESHOLD 262144
#endif
#define BLI_GER_V2_COLS( ch, ctype, kfp_av, conjx, conjy, m, n, alpha, x, incx, y, incy, a, rs_a, cs_a, cntx ) \
{ \
	if ( ( uint64_t )(m)*( uint64_t )(n) >= ( uint64_t )BLIS_L2_MT_THRESHOLD && \
	     omp_get_active_level() == 0 && omp_get_max_threads() > 1 ) \
	{ \
		_Pragma( "omp parallel" ) \
		{ \
			const dim_t nt_ = omp_get_num_threads(), tid_ = omp_get_thread_num(); \
			const dim_t bs_ = (n) / nt_, rm_ = (n) % nt_; \
			const dim_t j0_ = tid_*bs_ + ( tid_ < rm_ ? tid_ : rm_ ); \
			const dim_t j1_ = j0_ + bs_ + ( tid_ < rm_ ? 1 : 0 ); \
			for ( dim_t j_ = j0_; j_ < j1_; ++j_ ) { \
				ctype ap_; \
				bli_tcopycjs( ch,ch, (conjy), *((y) + j_*(incy)), ap_ ); \
				bli_tscals( ch,ch,ch, *(alpha), ap_ ); \
				kfp_av( (conjx), (m), &ap_, (x), (incx), (a) + j_*(cs_a), (rs_a), (cntx) ); \
			} \
		} \
	} \
	else { \
		for ( dim_t j_ = 0; j_ < (n); ++j_ ) { \
			ctype ap_; \
			bli_tcopycjs( ch,ch, (conjy), *((y) + j_*(incy)), ap_ ); \
			bli_tscals( ch,ch,ch, *(alpha), ap_ ); \
			kfp_av( (conjx), (m), &ap_, (x), (incx), (a) + j_*(cs_a), (rs_a), (cntx) ); \
		} \
	} \
}
#else
#define BLI_GER_V2_COLS( ch, ctype, kfp_av, conjx, conjy, m, n, alpha, x, incx, y, incy, a, rs_a, cs_a, cntx ) \
{ \
	for ( dim_t j_ = 0; j_ < (n); ++j_ ) { \
		ctype ap_; \
		bli_tcopycjs( ch,ch, (conjy), *((y) + j_*(incy)), ap_ ); \
		bli_tscals( ch,ch,ch, *(alpha), ap_ ); \
		kfp_av( (conjx), (m), &ap_, (x), (incx), (a) + j_*(cs_a), (rs_a), (cntx) ); \
	} \
}
#endif

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  a1; \
	ctype*  x1; \
	ctype*  psi1; \
	ctype   alpha_psi1; \
	dim_t   j; \
\
	/* Query the context for the kernel function pointer. */ \
	axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	( void )a1; ( void )x1; ( void )psi1; ( void )alpha_psi1; ( void )j; \
	BLI_GER_V2_COLS( ch, ctype, kfp_av, conjx, conjy, m, n, alpha, x, incx, y, incy, a, rs_a, cs_a, cntx ); \
}

INSERT_GENTFUNC_BASIC( ger_unb_var2 )

