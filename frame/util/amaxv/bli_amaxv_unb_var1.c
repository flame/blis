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

#define FUNCPTR_T amaxv_fp

typedef void (*FUNCPTR_T)(
                           dim_t  n,
                           void*  x, inc_t incx,
                           void*  abmax_i
                         );

static FUNCPTR_T GENARRAY(ftypes,amaxv_unb_var1);


void bli_amaxv_unb_var1( obj_t*  x,
                         obj_t*  abmax_i )
{
	num_t     dt_x        = bli_obj_datatype( *x );

	dim_t     n           = bli_obj_vector_dim( *x );

	inc_t     inc_x       = bli_obj_vector_inc( *x );
	void*     buf_x       = bli_obj_buffer_at_off( *x );

	void*     buf_abmax_i = bli_obj_buffer_at_off( *abmax_i );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x];

	// Invoke the function.
	f( n,
	   buf_x, inc_x,
	   buf_abmax_i );
}


#undef  GENTFUNCRI
#define GENTFUNCRI( ctype_x, ctype_xr, ctype_i, chx, chxr, chi, varname ) \
\
void PASTEMAC(chx,varname)( \
                            dim_t  n, \
                            void*  x, inc_t incx, \
                            void*  abmax_i  \
                          ) \
{ \
	ctype_x*  x_cast       = x; \
	ctype_i*  abmax_i_cast = abmax_i; \
	ctype_xr* minus_one    = PASTEMAC(chxr,m1); \
	ctype_i*  zero_i       = PASTEMAC(chi,0); \
\
	ctype_x*  chi1; \
	ctype_xr  chi1_r; \
	ctype_xr  chi1_i; \
	ctype_xr  abs_chi1; \
	ctype_xr  abs_chi1_max; \
	ctype_i   i_max; \
	dim_t     i; \
\
	/* If the vector is empty, return an index of zero. */ \
	if ( bli_zero_dim1( n ) ) \
	{ \
		PASTEMAC2(chi,chi,copys)( *zero_i, *abmax_i_cast ); \
		return; \
	} \
\
	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC2(chi,chi,copys)( *zero_i, i_max ); \
\
	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */ \
	PASTEMAC2(chxr,chxr,copys)( *minus_one, abs_chi1_max ); \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		chi1 = x_cast + (i  )*incx; \
\
		/* Get the real and imaginary components of chi1. */ \
		PASTEMAC2(chx,chxr,gets)( *chi1, chi1_r, chi1_i ); \
\
		/* Replace chi1_r and chi1_i with their absolute values. */ \
		PASTEMAC2(chxr,chxr,abval2s)( chi1_r, chi1_r ); \
		PASTEMAC2(chxr,chxr,abval2s)( chi1_i, chi1_i ); \
\
		/* Add the real and imaginary absolute values together. */ \
		PASTEMAC(chxr,set0s)( abs_chi1 ); \
		PASTEMAC2(chxr,chxr,adds)( chi1_r, abs_chi1 ); \
		PASTEMAC2(chxr,chxr,adds)( chi1_i, abs_chi1 ); \
\
		/* If the absolute value of the current element exceeds that of
		   the previous largest, save it and its index. If NaN is
		   encountered, then treat it the same as if it were a valid
		   value that was smaller than any previously seen. This
		   behavior mimics that of LAPACK's ?lange(). */ \
		if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
		{ \
			PASTEMAC2(chxr,chxr,copys)( abs_chi1, abs_chi1_max ); \
			PASTEMAC2(chi,chi,copys)( i, i_max ); \
		} \
	} \
\
	/* Store final index to output variable. */ \
	PASTEMAC2(chi,chi,copys)( i_max, *abmax_i_cast ); \
}

INSERT_GENTFUNCRI_BASIC0( amaxv_unb_var1 )

