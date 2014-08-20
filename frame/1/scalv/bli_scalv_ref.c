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

/*
#define FUNCPTR_T scalv_fp

typedef void (*FUNCPTR_T)(
                           conj_t conjbeta,
                           dim_t  n,
                           void*  beta,
                           void*  x, inc_t incx
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,scalv_ref);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,scalv_ref);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,scalv_ref);
#endif
#endif


void bli_scalv_ref( obj_t*  beta,
                         obj_t*  x )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	conj_t    conjbeta  = bli_obj_conj_status( *beta );

	dim_t     n         = bli_obj_vector_dim( *x );

	inc_t     inc_x     = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	num_t     dt_beta;
	void*     buf_beta;

	FUNCPTR_T f;

	// If beta is a scalar constant, use dt_x to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the beta object and extract the buffer at the beta offset.
	bli_set_scalar_dt_buffer( beta, dt_x, dt_beta, buf_beta );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_beta][dt_x];

	// Invoke the function.
	f( conjbeta,
	   n,
	   buf_beta,
	   buf_x, inc_x );
}
*/


#undef  GENTFUNC2
#define GENTFUNC2( ctype_b, ctype_x, chb, chx, varname, setvker ) \
\
void PASTEMAC2(chb,chx,varname) \
     ( \
       conj_t            conjbeta, \
       dim_t             n, \
       ctype_b* restrict beta, \
       ctype_x* restrict x, inc_t incx \
     ) \
{ \
	ctype_b* beta_cast = beta; \
	ctype_x* x_cast    = x; \
	ctype_x* chi1; \
	ctype_b  beta_conj; \
	dim_t    i; \
\
	if ( bli_zero_dim1( n ) ) return; \
\
	/* If beta is one, return. */ \
	if ( PASTEMAC(chb,eq1)( *beta_cast ) ) return; \
\
	/* If beta is zero, use setv. */ \
	if ( PASTEMAC(chb,eq0)( *beta_cast ) ) \
	{ \
		ctype_x* zero = PASTEMAC(chb,0); \
\
		PASTEMAC2(chx,chx,setvker)( n, \
		                            zero, \
		                            x, incx ); \
		return; \
	} \
\
	PASTEMAC(chb,copycjs)( conjbeta, *beta_cast, beta_conj ); \
\
	chi1 = x_cast; \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		PASTEMAC2(chb,chx,scals)( beta_conj, *chi1 ); \
\
		chi1 += incx; \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( scalv_ref, SETV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( scalv_ref, SETV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( scalv_ref, SETV_KERNEL )
#endif

