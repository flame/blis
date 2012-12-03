/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#include "blis2.h"

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   a, inc_t rs_a, inc_t cs_a,
                           void*   b, inc_t rs_b, inc_t cs_b,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T GENARRAY(ftypes,gemm_unb_var2);


void bl2_gemm_unb_var2( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  b,
                        obj_t*  beta,
                        obj_t*  c,
                        gemm_t* cntl )
{
	num_t     dt_exec   = bl2_obj_execution_datatype( *c );

	dim_t     m         = bl2_obj_length( *c );
	dim_t     n         = bl2_obj_width( *c );
	dim_t     k         = bl2_obj_width( *a );

	void*     buf_a     = bl2_obj_buffer_at_off( *a );
	inc_t     rs_a      = bl2_obj_row_stride( *a );
	inc_t     cs_a      = bl2_obj_col_stride( *a );

	void*     buf_b     = bl2_obj_buffer_at_off( *b );
	inc_t     rs_b      = bl2_obj_row_stride( *b );
	inc_t     cs_b      = bl2_obj_col_stride( *b );

	void*     buf_c     = bl2_obj_buffer_at_off( *c );
	inc_t     rs_c      = bl2_obj_row_stride( *c );
	inc_t     cs_c      = bl2_obj_col_stride( *c );

	FUNCPTR_T f;

/*
	// Handle the special case where c and a are complex and b is real.
	// Note that this is the ONLY case allowed by the inner kernel whereby
	// the datatypes of a and b differ. In this situation, the execution
	// datatype is real, so we need to inflate the m and leading dimensions
	// by a factor of two.
	if ( bl2_obj_is_complex( *a ) && bl2_obj_is_real( *b ) )
	{
		m    *= 2;
		cs_c *= 2;
		cs_a *= 2;
	}
*/

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( m,
	   n,
	   k,
	   buf_a, rs_a, cs_a,
	   buf_b, rs_b, cs_b,
	   buf_c, rs_c, cs_c );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   a, inc_t rs_a, inc_t cs_a, \
                           void*   b, inc_t rs_b, inc_t cs_b, \
                           void*   c, inc_t rs_c, inc_t cs_c \
                         ) \
{ \
	ctype* a1; \
	ctype* b1; \
	ctype* c1; \
	ctype* alpha11; \
	ctype* beta11; \
	ctype* gamma11; \
	ctype  rho; \
	dim_t  i, j, h; \
\
	if ( bl2_zero_dim3( m, n, k ) ) return; \
\
	c1 = c; \
	b1 = b; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		gamma11 = c1; \
		a1      = a; \
\
		for ( i = 0; i < m; ++i ) \
		{ \
			/* gamma11 = c1 + (i  )*rs_c + (j  )*cs_c; */ \
\
			alpha11 = a1; \
			beta11  = b1; \
\
			PASTEMAC(ch,set0)( rho ); \
\
			for ( h = 0; h < k; ++h ) \
			{ \
				/* alpha11 = a1 + (i  )*rs_a + (h  )*cs_a; */ \
				/* beta11  = b1 + (h  )*rs_b + (j  )*cs_b; */ \
\
				PASTEMAC(ch,dots)( *alpha11, *beta11, rho ); \
\
				alpha11 += cs_a; \
				beta11  += rs_b; \
			} \
\
			PASTEMAC(ch,adds)( rho, *gamma11 ); \
\
			gamma11 += rs_c; \
			a1      += rs_a; \
		} \
\
		c1 += cs_c; \
		b1 += cs_b; \
	} \
}

INSERT_GENTFUNC_BASIC( gemm, gemm_unb_var2 )

