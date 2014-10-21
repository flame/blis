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

#undef  FUNCPTR_T
#define FUNCPTR_T gemmtrsm_ukr_fp

typedef void (*FUNCPTR_T)(
                           dim_t      k,
                           void*      alpha,
                           void*      a1x,
                           void*      a11,
                           void*      bx1,
                           void*      b11,
                           void*      c11, inc_t rs_c, inc_t cs_c,
                           auxinfo_t* data
                         );

static FUNCPTR_T GENARRAY(ftypes_l,gemmtrsm_l_ukernel_void);
static FUNCPTR_T GENARRAY(ftypes_u,gemmtrsm_u_ukernel_void);


void bli_gemmtrsm_ukernel( obj_t*  alpha,
                           obj_t*  a1x,
                           obj_t*  a11,
                           obj_t*  bx1,
                           obj_t*  b11,
                           obj_t*  c11 )
{
	dim_t     k         = bli_obj_width( *a1x );

	num_t     dt        = bli_obj_datatype( *c11 );

	void*     buf_a1x   = bli_obj_buffer_at_off( *a1x );

	void*     buf_a11   = bli_obj_buffer_at_off( *a11 );

	void*     buf_bx1   = bli_obj_buffer_at_off( *bx1 );

	void*     buf_b11   = bli_obj_buffer_at_off( *b11 );

	void*     buf_c11   = bli_obj_buffer_at_off( *c11 );
	inc_t     rs_c      = bli_obj_row_stride( *c11 );
	inc_t     cs_c      = bli_obj_col_stride( *c11 );

	void*     buf_alpha = bli_obj_buffer_for_1x1( dt, *alpha );

	FUNCPTR_T f;

	auxinfo_t data;


	// Fill the auxinfo_t struct in case the micro-kernel uses it.
	if ( bli_obj_is_lower( *a11 ) )
	{ bli_auxinfo_set_next_a( buf_a1x, data ); }
	else
	{ bli_auxinfo_set_next_a( buf_a11, data ); }
	bli_auxinfo_set_next_b( buf_bx1, data );

	// Index into the type combination array to extract the correct
	// function pointer.
	if ( bli_obj_is_lower( *a11 ) ) f = ftypes_l[dt];
	else                            f = ftypes_u[dt];

	// Invoke the function.
	f( k,
	   buf_alpha,
	   buf_a1x,
	   buf_a11,
	   buf_bx1,
	   buf_b11,
	   buf_c11, rs_c, cs_c,
	   &data );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, ukrname ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t      k, \
                           void*      alpha, \
                           void*      a1x, \
                           void*      a11, \
                           void*      bx1, \
                           void*      b11, \
                           void*      c11, inc_t rs_c, inc_t cs_c, \
                           auxinfo_t* data  \
                         ) \
{ \
	PASTEMAC(ch,ukrname)( k, \
	                      alpha, \
	                      a1x, \
	                      a11, \
	                      bx1, \
	                      b11, \
	                      c11, rs_c, cs_c, \
	                      data ); \
}

INSERT_GENTFUNC_BASIC( gemmtrsm_l_ukernel_void, GEMMTRSM_L_UKERNEL )
INSERT_GENTFUNC_BASIC( gemmtrsm_u_ukernel_void, GEMMTRSM_U_UKERNEL )

