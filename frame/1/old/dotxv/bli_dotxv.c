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
                           conj_t conjx,
                           conj_t conjy,
                           dim_t  n,
                           void*  alpha,
                           void*  x, inc_t incx,
                           void*  y, inc_t incy,
                           void*  beta,
                           void*  rho
                         );

static FUNCPTR_T GENARRAY_MIN(ftypes,dotxv_void);


//
// Define object-based interface.
//
void bli_dotxv( obj_t*  alpha,
                obj_t*  x,
                obj_t*  y,
                obj_t*  beta,
                obj_t*  rho )
{
	num_t     dt        = bli_obj_datatype( *x );

	conj_t    conjx     = bli_obj_conj_status( *x );
	conj_t    conjy     = bli_obj_conj_status( *y );

	dim_t     n         = bli_obj_vector_dim( *x );

	inc_t     inc_x     = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	inc_t     inc_y     = bli_obj_vector_inc( *y );
	void*     buf_y     = bli_obj_buffer_at_off( *y );

	void*     buf_rho   = bli_obj_buffer_at_off( *rho );

	obj_t     alpha_local;
	void*     buf_alpha;

	obj_t     beta_local;
	void*     buf_beta;

	FUNCPTR_T f         = ftypes[dt];

	if ( bli_error_checking_is_enabled() )
	    bli_dotxv_check( alpha, x, y, beta, rho );

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

	// Invoke the void pointer-based function.
	f( conjx,
	   n,
	   buf_alpha,
	   buf_x, inc_x,
	   buf_y, inc_y,
	   buf_beta,
	   buf_rho );
}


//
// Define BLAS-like interfaces with void pointer operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjx, \
                          conj_t conjy, \
                          dim_t  n, \
                          void*  alpha, \
                          void*  x, inc_t incx, \
                          void*  y, inc_t incy, \
                          void*  beta, \
                          void*  rho  \
                        ) \
{ \
	PASTEMAC(ch,kername)( conjx, \
	                      conjy, \
	                      n, \
	                      alpha, \
	                      x, incx, \
	                      y, incy, \
	                      beta, \
	                      rho ); \
}

INSERT_GENTFUNC_BASIC( dotxv_void, dotxv )


//
// Define BLAS-like interfaces with typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjx, \
                          conj_t conjy, \
                          dim_t  n, \
                          ctype* alpha, \
                          ctype* x, inc_t incx, \
                          ctype* y, inc_t incy, \
                          ctype* beta, \
                          ctype* rho  \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
	cntx_t*     cntx; \
\
	PASTECH2(ch,opname,_ker_t) f; \
\
	PASTEMAC(opname,_cntx_init)( &cntx ); \
\
	f = bli_cntx_get_l1v_ker_dt( dt, kerid, &cntx ); \
\
	f( conjx, \
	   conjy, \
	   n, \
	   alpha, \
	   x, incx, \
	   y, incy, \
	   beta, \
	   rho ); \
\
	PASTEMAC(opname,_cntx_finalize)( &cntx ); \
}

INSERT_GENTFUNC_BASIC( dotxv, BLIS_DOTXV_KER )


