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

typedef void (*FUNCPTR_T)(
                           conj_t conjat,
                           conj_t conja,
                           conj_t conjw,
                           conj_t conjx,
                           dim_t  m,
                           dim_t  b_n,
                           void*  alpha,
                           void*  a, inc_t inca, inc_t lda,
                           void*  w, inc_t incw,
                           void*  x, inc_t incx,
                           void*  beta,
                           void*  y, inc_t incy,
                           void*  z, inc_t incz
                         );

static FUNCPTR_T GENARRAY_MIN(ftypes,dotxaxpyf_void);


//
// Define object-based interface.
//
void bli_dotxaxpyf( obj_t*  alpha,
                    obj_t*  at,
                    obj_t*  a,
                    obj_t*  w,
                    obj_t*  x,
                    obj_t*  beta,
                    obj_t*  y,
                    obj_t*  z )
{
	num_t     dt        = bli_obj_datatype( *x );

	conj_t    conjat    = bli_obj_conj_status( *at );
	conj_t    conja     = bli_obj_conj_status( *a );
	conj_t    conjw     = bli_obj_conj_status( *w );
	conj_t    conjx     = bli_obj_conj_status( *x );

	dim_t     m         = bli_obj_vector_dim( *z );
	dim_t     b_n       = bli_obj_vector_dim( *y );

	void*     buf_a     = bli_obj_buffer_at_off( *a );
	inc_t     rs_a      = bli_obj_row_stride( *a );
	inc_t     cs_a      = bli_obj_col_stride( *a );

	void*     buf_w     = bli_obj_buffer_at_off( *w );
	inc_t     inc_w     = bli_obj_vector_inc( *w );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     inc_x     = bli_obj_vector_inc( *x );

	void*     buf_y     = bli_obj_buffer_at_off( *y );
	inc_t     inc_y     = bli_obj_vector_inc( *y );

	void*     buf_z     = bli_obj_buffer_at_off( *z );
	inc_t     inc_z     = bli_obj_vector_inc( *z );

	obj_t     alpha_local;
	void*     buf_alpha;

	obj_t     beta_local;
	void*     buf_beta;

	FUNCPTR_T f         = ftypes[dt];

	if ( bli_error_checking_is_enabled() )
	    bli_dotxaxpyf_check( alpha, at, a, w, x, beta, y, z );

	// Create local copy-casts of the scalars (and apply internal conjugation
	// if needed).
	bli_obj_scalar_init_detached_copy_of( dt,
	                                      BLIS_NO_CONJUGATE,
	                                      alpha,
	                                      &alpha_local );
	bli_obj_scalar_init_detached_copy_of( dt,
	                                      BLIS_NO_CONJUGATE,
	                                      beta,
	                                      &beta_local );

	// Extract the scalar buffers.
	buf_alpha = bli_obj_buffer_for_1x1( dt, alpha_local );
	buf_beta  = bli_obj_buffer_for_1x1( dt, beta_local );

	// Support cases where matrix A requires a transposition.
	if ( bli_obj_has_trans( *a ) ) { bli_swap_incs( rs_a, cs_a ); }

	// Invoke the void pointer-based function.
	f( conjat,
	   conja,
	   conjw,
	   conjx,
	   m,
	   b_n,
	   buf_alpha,
	   buf_a, rs_a, cs_a,
	   buf_w, inc_w,
	   buf_x, inc_x,
	   buf_beta,
	   buf_y, inc_y,
	   buf_z, inc_z );
}


//
// Define BLAS-like interfaces with void pointer operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjat, \
                          conj_t conja, \
                          conj_t conjw, \
                          conj_t conjx, \
                          dim_t  m, \
                          dim_t  b_n, \
                          void*  alpha, \
                          void*  a, inc_t inca, inc_t lda, \
                          void*  w, inc_t incw, \
                          void*  x, inc_t incx, \
                          void*  beta, \
                          void*  y, inc_t incy, \
                          void*  z, inc_t incz  \
                        ) \
{ \
	PASTEMAC(ch,kername)( conjat, \
	                      conja, \
	                      conjw, \
	                      conjx, \
	                      m, \
	                      b_n, \
	                      alpha, \
	                      a, inca, lda, \
	                      w, incw, \
	                      x, incx, \
	                      beta, \
	                      y, incy, \
	                      z, incz ); \
}

INSERT_GENTFUNC_BASIC( dotxaxpyf_void, dotxaxpyf )


//
// Define BLAS-like interfaces with typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjat, \
                          conj_t conja, \
                          conj_t conjw, \
                          conj_t conjx, \
                          dim_t  m, \
                          dim_t  b_n, \
                          ctype* alpha, \
                          ctype* a, inc_t inca, inc_t lda, \
                          ctype* w, inc_t incw, \
                          ctype* x, inc_t incx, \
                          ctype* beta, \
                          ctype* y, inc_t incy, \
                          ctype* z, inc_t incz  \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
	cntx_t*     cntx; \
\
	PASTECH2(ch,opname,_ker_t) f; \
\
	PASTEMAC(opname,_cntx_init)( &cntx ); \
\
	f = bli_cntx_get_l1f_ker_dt( dt, kerid, &cntx ); \
\
	f( conjat, \
	   conja, \
	   conjw, \
	   conjx, \
	   m, \
	   b_n, \
	   alpha, \
	   a, inca, lda, \
	   w, incw, \
	   x, incx, \
	   beta, \
	   y, incy, \
	   z, incz ); \
\
	PASTEMAC(opname,_cntx_finalize)( &cntx ); \
}

INSERT_GENTFUNC_BASIC( dotxaxpyf, BLIS_DOTXAXPYF_KER )


