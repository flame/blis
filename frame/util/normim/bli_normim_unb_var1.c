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

#define FUNCPTR_T normim_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffx,
                           diag_t  diagx,
                           uplo_t  uplox,
                           dim_t   m,
                           dim_t   n,
                           void*   x, inc_t rs_x, inc_t cs_x,
                           void*   norm
                         );

static FUNCPTR_T GENARRAY(ftypes,normim_unb_var1);


void bli_normim_unb_var1( obj_t* x,
                          obj_t* norm )
{
	num_t     dt_x     = bli_obj_datatype( *x );

	doff_t    diagoffx = bli_obj_diag_offset( *x );
	uplo_t    diagx    = bli_obj_diag( *x );
	uplo_t    uplox    = bli_obj_uplo( *x );

	dim_t     m        = bli_obj_length( *x );
	dim_t     n        = bli_obj_width( *x );

	void*     buf_x    = bli_obj_buffer_at_off( *x );
	inc_t     rs_x     = bli_obj_row_stride( *x );
	inc_t     cs_x     = bli_obj_col_stride( *x );

	void*     buf_norm = bli_obj_buffer_at_off( *norm );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x];

	// Invoke the function.
	f( diagoffx,
	   diagx,
	   uplox,
	   m,
	   n,
	   buf_x, rs_x, cs_x,
	   buf_norm );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype_x, ctype_xr, chx, chxr, varname, kername ) \
\
void PASTEMAC(chx,varname)( \
                            doff_t  diagoffx, \
                            diag_t  diagx, \
                            uplo_t  uplox, \
                            dim_t   m, \
                            dim_t   n, \
                            void*   x, inc_t rs_x, inc_t cs_x, \
                            void*   norm  \
                          ) \
{ \
	/* Induce a transposition so that rows become columns. */ \
	bli_swap_dims( m, n ); \
	bli_swap_incs( rs_x, cs_x ); \
	bli_toggle_uplo( uplox ); \
	bli_negate_diag_offset( diagoffx ); \
\
	/* Now we can simply compute the 1-norm of this transposed matrix,
	   which will be equivalent to the infinity-norm of the original
	   matrix. */ \
	PASTEMAC(chx,kername)( diagoffx, \
	                       diagx, \
	                       uplox, \
	                       m, \
	                       n, \
	                       x, rs_x, cs_x, \
	                       norm ); \
}


INSERT_GENTFUNCR_BASIC( normim_unb_var1, norm1m_unb_var1 )

