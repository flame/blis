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

#define FUNCPTR_T mktrim_fp

typedef void (*FUNCPTR_T)(
                           uplo_t  uploa,
                           dim_t   m,
                           void*   a, inc_t rs_a, inc_t cs_a
                         );

static FUNCPTR_T GENARRAY(ftypes,mktrim_unb_var1);


void bli_mktrim_unb_var1( obj_t* a )
{
	num_t     dt_a      = bli_obj_datatype( *a );

	uplo_t    uploa     = bli_obj_uplo( *a );

	dim_t     m         = bli_obj_length( *a );

	void*     buf_a     = bli_obj_buffer_at_off( *a );
	inc_t     rs_a      = bli_obj_row_stride( *a );
	inc_t     cs_a      = bli_obj_col_stride( *a );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_a];

	// Invoke the function.
	f( uploa,
	   m,
	   buf_a, rs_a, cs_a );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           uplo_t  uploa, \
                           dim_t   m, \
                           void*   a, inc_t rs_a, inc_t cs_a \
                          ) \
{ \
	ctype*  a_cast     = a; \
	ctype*  zero       = PASTEMAC(ch,0); \
	doff_t  diagoffa; \
\
	/* If the dimension is zero, return early. */ \
	if ( bli_zero_dim1( m ) ) return; \
\
	/* Toggle uplo so that it refers to the unstored triangle. */ \
	bli_toggle_uplo( uploa ); \
\
	/* In order to avoid the main diagonal, we must nudge the diagonal either
	   up or down by one, depending on which triangle is to be zeroed. */ \
	if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
	else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
	/* Set the unstored triangle to zero. */ \
	PASTEMAC2(ch,ch,setm)( diagoffa, \
	                       BLIS_NONUNIT_DIAG, \
	                       uploa, \
	                       m, \
	                       m, \
	                       zero, \
	                       a_cast, rs_a, cs_a ); \
}


INSERT_GENTFUNC_BASIC0( mktrim_unb_var1 )

