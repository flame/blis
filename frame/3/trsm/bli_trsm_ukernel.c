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
#define FUNCPTR_T trsm_ukr_fp

typedef void (*FUNCPTR_T)(
                           void*      a,
                           void*      b,
                           void*      c, inc_t rs_c, inc_t cs_c,
                           auxinfo_t* data
                         );

static FUNCPTR_T GENARRAY(ftypes_l,trsm_l_ukernel_void);
static FUNCPTR_T GENARRAY(ftypes_u,trsm_u_ukernel_void);


void bli_trsm_ukernel( obj_t*  a,
                       obj_t*  b,
                       obj_t*  c )
{
	num_t     dt        = bli_obj_datatype( *c );

	void*     buf_a     = bli_obj_buffer_at_off( *a );

	void*     buf_b     = bli_obj_buffer_at_off( *b );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	FUNCPTR_T f;

	auxinfo_t data;


	// Fill the auxinfo_t struct in case the micro-kernel uses it.
	bli_auxinfo_set_next_a( buf_a, data );
	bli_auxinfo_set_next_b( buf_b, data );

	// Index into the type combination array to extract the correct
	// function pointer.
	if ( bli_obj_is_lower( *a ) ) f = ftypes_l[dt];
	else                          f = ftypes_u[dt];

	// Invoke the function.
	f( buf_a,
	   buf_b,
	   buf_c, rs_c, cs_c,
	   &data );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, ukrname ) \
\
void PASTEMAC(ch,varname)( \
                           void*      a, \
                           void*      b, \
                           void*      c, inc_t rs_c, inc_t cs_c, \
                           auxinfo_t* data  \
                         ) \
{ \
	PASTEMAC(ch,ukrname)( a, \
	                      b, \
	                      c, rs_c, cs_c, \
	                      data ); \
}

INSERT_GENTFUNC_BASIC( trsm_l_ukernel_void, TRSM_L_UKERNEL )
INSERT_GENTFUNC_BASIC( trsm_u_ukernel_void, TRSM_U_UKERNEL )

