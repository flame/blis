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
                           void*  alphax,
                           void*  alphay,
                           void*  x, inc_t incx,
                           void*  y, inc_t incy,
                           void*  z, inc_t incz
                         );

static FUNCPTR_T GENARRAY_MIN(ftypes,axpy2v_void);


//
// Define object-based interface.
//
void bli_axpy2v( obj_t*  alphax,
                 obj_t*  alphay,
                 obj_t*  x,
                 obj_t*  y,
                 obj_t*  z )
{
	num_t     dt        = bli_obj_datatype( *x );

	conj_t    conjx     = bli_obj_conj_status( *x );
	conj_t    conjy     = bli_obj_conj_status( *y );

	dim_t     n         = bli_obj_vector_dim( *x );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     inc_x     = bli_obj_vector_inc( *x );

	void*     buf_y     = bli_obj_buffer_at_off( *y );
	inc_t     inc_y     = bli_obj_vector_inc( *y );

	void*     buf_z     = bli_obj_buffer_at_off( *z );
	inc_t     inc_z     = bli_obj_vector_inc( *z );

	obj_t     alphax_local;
	void*     buf_alphax;

	obj_t     alphay_local;
	void*     buf_alphay;

	FUNCPTR_T f         = ftypes[dt];

	if ( bli_error_checking_is_enabled() )
	    bli_axpy2v_check( alphax, alphay, x, y, z );

	// Create local copy-casts of the scalars (and apply internal conjugation
	// if needed).
	bli_obj_scalar_init_detached_copy_of( dt,
	                                      BLIS_NO_CONJUGATE,
	                                      alphax,
	                                      &alphax_local );
	bli_obj_scalar_init_detached_copy_of( dt,
	                                      BLIS_NO_CONJUGATE,
	                                      alphay,
	                                      &alphay_local );

	// Extract the scalar buffers.
	buf_alphax = bli_obj_buffer_for_1x1( dt, alphax_local );
	buf_alphay = bli_obj_buffer_for_1x1( dt, alphay_local );

	// Invoke the void pointer-based function.
	f( conjx,
	   conjy,
	   n,
	   buf_alphax,
	   buf_alphay,
	   buf_x, inc_x,
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
                          conj_t conjx, \
                          conj_t conjy, \
                          dim_t  n, \
                          void*  alphax, \
                          void*  alphay, \
                          void*  x, inc_t incx, \
                          void*  y, inc_t incy, \
                          void*  z, inc_t incz  \
                        ) \
{ \
	PASTEMAC(ch,kername)( conjx, \
	                      conjy, \
	                      n, \
	                      alphax, \
	                      alphay, \
	                      x, incx, \
	                      y, incy, \
	                      z, incz ); \
}

INSERT_GENTFUNC_BASIC( axpy2v_void, axpy2v )


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
                          ctype* alphax, \
                          ctype* alphay, \
                          ctype* x, inc_t incx, \
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
	f( conjx, \
	   conjy, \
	   n, \
	   alphax, \
	   alphay, \
	   x, incx, \
	   y, incy, \
	   z, incz ); \
\
	PASTEMAC(opname,_cntx_finalize)( &cntx ); \
}

INSERT_GENTFUNC_BASIC( axpy2v, BLIS_AXPY2V_KER )


