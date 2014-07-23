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

#define FUNCPTR_T packv_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           void*   c, inc_t incc,
                           void*   p, inc_t incp
                         );

static FUNCPTR_T GENARRAY(ftypes,packv_unb_var1);


void bli_packv_unb_var1( obj_t*   c,
                         obj_t*   p,
                         packv_t* cntl )
{
	num_t     dt_cp     = bli_obj_datatype( *c );

	dim_t     dim_p     = bli_obj_vector_dim( *p );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     incc      = bli_obj_vector_inc( *c );

	void*     buf_p     = bli_obj_buffer_at_off( *p );
	inc_t     incp      = bli_obj_vector_inc( *p );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_cp];

	// Invoke the function.
	f( dim_p,
	   buf_c, incc,
	   buf_p, incp );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, kername ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t   m, \
                           void*   c, inc_t incc, \
                           void*   p, inc_t incp \
                         ) \
{ \
	ctype* c_cast = c; \
	ctype* p_cast = p; \
\
	PASTEMAC2(ch,ch,kername)( BLIS_NO_CONJUGATE, \
	                          m, \
	                          c_cast, incc, \
	                          p_cast, incp ); \
}

INSERT_GENTFUNC_BASIC( packv_unb_var1, COPYV_KERNEL )

