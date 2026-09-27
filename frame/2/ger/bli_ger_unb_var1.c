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

// Optional OpenMP row-parallel rank-1 update (ger, var1). Each row of A is an
// independent axpyv (A[i,:] += alpha*conj(x[i]) * y), so split the rows across
// threads -- disjoint output, no reduction, raw axpyv kernel (no nesting).
// Enabled via BLIS_ENABLE_L1_OPENMP.
#ifdef BLIS_ENABLE_L1_OPENMP
#include <omp.h>
#ifndef BLIS_L2_MT_THRESHOLD
#define BLIS_L2_MT_THRESHOLD 262144
#endif
#define BLI_GER_V1_ROWS( ch, ctype, kfp_av, conjx, conjy, m, n, alpha, x, incx, y, incy, a, rs_a, cs_a, cntx ) \
{ \
	if ( ( uint64_t )(m)*( uint64_t )(n) >= ( uint64_t )BLIS_L2_MT_THRESHOLD && \
	     omp_get_active_level() == 0 && omp_get_max_threads() > 1 ) \
	{ \
		_Pragma( "omp parallel" ) \
		{ \
			const dim_t nt_ = omp_get_num_threads(), tid_ = omp_get_thread_num(); \
			const dim_t bs_ = (m) / nt_, rm_ = (m) % nt_; \
			const dim_t i0_ = tid_*bs_ + ( tid_ < rm_ ? tid_ : rm_ ); \
			const dim_t i1_ = i0_ + bs_ + ( tid_ < rm_ ? 1 : 0 ); \
			for ( dim_t i_ = i0_; i_ < i1_; ++i_ ) { \
				ctype ac_; \
				bli_tcopycjs( ch,ch, (conjx), *((x) + i_*(incx)), ac_ ); \
				bli_tscals( ch,ch,ch, *(alpha), ac_ ); \
				kfp_av( (conjy), (n), &ac_, (y), (incy), (a) + i_*(rs_a), (cs_a), (cntx) ); \
			} \
		} \
	} \
	else { \
		for ( dim_t i_ = 0; i_ < (m); ++i_ ) { \
			ctype ac_; \
			bli_tcopycjs( ch,ch, (conjx), *((x) + i_*(incx)), ac_ ); \
			bli_tscals( ch,ch,ch, *(alpha), ac_ ); \
			kfp_av( (conjy), (n), &ac_, (y), (incy), (a) + i_*(rs_a), (cs_a), (cntx) ); \
		} \
	} \
}
#else
#define BLI_GER_V1_ROWS( ch, ctype, kfp_av, conjx, conjy, m, n, alpha, x, incx, y, incy, a, rs_a, cs_a, cntx ) \
{ \
	for ( dim_t i_ = 0; i_ < (m); ++i_ ) { \
		ctype ac_; \
		bli_tcopycjs( ch,ch, (conjx), *((x) + i_*(incx)), ac_ ); \
		bli_tscals( ch,ch,ch, *(alpha), ac_ ); \
		kfp_av( (conjy), (n), &ac_, (y), (incy), (a) + i_*(rs_a), (cs_a), (cntx) ); \
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
	ctype*  a1t; \
	ctype*  chi1; \
	ctype*  y1; \
	ctype   alpha_chi1; \
	dim_t   i; \
\
	/* Query the context for the kernel function pointer. */ \
	axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	( void )a1t; ( void )chi1; ( void )y1; ( void )alpha_chi1; ( void )i; \
	BLI_GER_V1_ROWS( ch, ctype, kfp_av, conjx, conjy, m, n, alpha, x, incx, y, incy, a, rs_a, cs_a, cntx ); \
}

INSERT_GENTFUNC_BASIC( ger_unb_var1 )

