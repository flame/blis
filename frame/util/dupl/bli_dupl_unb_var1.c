/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

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

#define FUNCPTR_T dupl_fp

typedef void (*FUNCPTR_T)(
                           dim_t   k,
                           void*   b,
                           void*   bd
                         );

static FUNCPTR_T GENARRAY(ftypes,dupl_unb_var1);


void bli_dupl_unb_var1( obj_t* b,
                        obj_t* bd )
{
	num_t     dt_b      = bli_obj_datatype( *b );

	dim_t     k;

	void*     buf_b     = bli_obj_buffer_at_off( *b );

	void*     buf_bd    = bli_obj_buffer_at_off( *bd );

	FUNCPTR_T f;

	// The k dimension is the one that is "perpendicular" to the
	// storage dimension. 
	if ( bli_obj_is_row_stored( *b ) ) k = bli_obj_length( *b );
	else                               k = bli_obj_width( *b );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_b];

	// Invoke the function.
	f( k,
	   buf_b,
	   buf_bd );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, kername ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t  n, \
                           void*  b, \
                           void*  bd \
                         ) \
{ \
	ctype*      b_cast  = b; \
	ctype*      bd_cast = bd; \
\
	const dim_t NDUP    = PASTEMAC(ch,ndup); \
	const dim_t NR      = PASTEMAC(ch,nr); \
	const dim_t PACKNR  = PASTEMAC(ch,packnr); \
\
	dim_t       i, j, el, d; \
\
	for ( el = 0; el < n; ++el ) \
	{ \
		i = el / NR; \
		j = el % NR; \
\
		for ( d = 0; d < NDUP; ++d ) \
		{ \
			*(bd_cast + el*NDUP + d) = *(b_cast + i*PACKNR + j); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( dupl_unb_var1, dupl_unb_var1 )

