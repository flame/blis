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

// -- Optional OpenMP row-parallel axpyf loop for gemv (no-transpose) ---------
// gemv accumulates A's columns into y, so we cannot split the column loop
// without a reduction. Instead we split y's rows (n_elem): each thread runs
// the full column loop over its own contiguous band of rows, with offset A/y
// pointers -- disjoint outputs, no reduction. Beta-scaling of y already
// happened (serially) before this loop. Enabled via BLIS_ENABLE_L1_OPENMP.
#ifdef BLIS_ENABLE_L1_OPENMP
#include <omp.h>
#ifndef BLIS_L2_MT_THRESHOLD
#define BLIS_L2_MT_THRESHOLD 262144   // min m*n to thread
#endif
#define BLI_GEMV_V2_AXPYF_LOOP( kfp_af, conja, conjx, n_elem, n_iter, b_fuse, \
                                alpha, a, rs_at, cs_at, x, incx, y, incy, cntx, mn ) \
{ \
	if ( ( uint64_t )(mn) >= ( uint64_t )BLIS_L2_MT_THRESHOLD && \
	     omp_get_active_level() == 0 && omp_get_max_threads() > 1 ) \
	{ \
		_Pragma( "omp parallel" ) \
		{ \
			const dim_t nt_ = omp_get_num_threads(), tid_ = omp_get_thread_num(); \
			const dim_t bs_ = (n_elem) / nt_, rm_ = (n_elem) % nt_; \
			const dim_t r0_ = tid_ * bs_ + ( tid_ < rm_ ? tid_ : rm_ ); \
			const dim_t rl_ = bs_ + ( tid_ < rm_ ? 1 : 0 ); \
			if ( rl_ > 0 ) { \
				dim_t i_, f_; \
				for ( i_ = 0; i_ < (n_iter); i_ += f_ ) { \
					f_ = bli_determine_blocksize_dim_f( i_, (n_iter), (b_fuse) ); \
					kfp_af( (conja), (conjx), rl_, f_, (alpha), \
					        (a) + r0_*(rs_at) + i_*(cs_at), (rs_at), (cs_at), \
					        (x) + i_*(incx), (incx), (y) + r0_*(incy), (incy), (cntx) ); \
				} \
			} \
		} \
	} \
	else { \
		dim_t i_, f_; \
		for ( i_ = 0; i_ < (n_iter); i_ += f_ ) { \
			f_ = bli_determine_blocksize_dim_f( i_, (n_iter), (b_fuse) ); \
			kfp_af( (conja), (conjx), (n_elem), f_, (alpha), \
			        (a) + i_*(cs_at), (rs_at), (cs_at), \
			        (x) + i_*(incx), (incx), (y), (incy), (cntx) ); \
		} \
	} \
}
#else
#define BLI_GEMV_V2_AXPYF_LOOP( kfp_af, conja, conjx, n_elem, n_iter, b_fuse, \
                                alpha, a, rs_at, cs_at, x, incx, y, incy, cntx, mn ) \
{ \
	dim_t i_, f_; \
	for ( i_ = 0; i_ < (n_iter); i_ += f_ ) { \
		f_ = bli_determine_blocksize_dim_f( i_, (n_iter), (b_fuse) ); \
		kfp_af( (conja), (conjx), (n_elem), f_, (alpha), \
		        (a) + i_*(cs_at), (rs_at), (cs_at), \
		        (x) + i_*(incx), (incx), (y), (incy), (cntx) ); \
	} \
}
#endif

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  zero       = PASTEMAC(ch,0); \
	ctype*  A1; \
	ctype*  x1; \
	ctype*  y1; \
	dim_t   i; \
	dim_t   b_fuse, f; \
	dim_t   n_elem, n_iter; \
	inc_t   rs_at, cs_at; \
	conj_t  conja; \
\
	bli_set_dims_incs_with_trans( transa, \
	                              m, n, rs_a, cs_a, \
	                              &n_elem, &n_iter, &rs_at, &cs_at ); \
\
	conja = bli_extract_conj( transa ); \
\
	/* If beta is zero, use setv. Otherwise, scale by beta. */ \
	if ( bli_teq0s( ch, *beta ) ) \
	{ \
		/* y = 0; */ \
		PASTEMAC(ch,setv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  n_elem, \
		  zero, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
	else \
	{ \
		/* y = beta * y; */ \
		PASTEMAC(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  n_elem, \
		  beta, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
\
	/* Query the context for the kernel function pointer and fusing factor. */ \
	axpyf_ker_ft kfp_af = bli_cntx_get_ukr_dt( dt, BLIS_AXPYF_KER, cntx ); \
	b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_AF, cntx ); \
\
	/* y = y + alpha * A * x, row-parallel when enabled (see macro above). */ \
	( void )A1; ( void )x1; ( void )y1; ( void )i; ( void )f; \
	BLI_GEMV_V2_AXPYF_LOOP( kfp_af, conja, conjx, n_elem, n_iter, b_fuse, \
	                        alpha, a, rs_at, cs_at, x, incx, y, incy, cntx, \
	                        ( uint64_t )m * ( uint64_t )n ); \
}

INSERT_GENTFUNC_BASIC( gemv_unf_var2 )

