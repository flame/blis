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

#define FUNCPTR_T fprintm_fp

typedef void (*FUNCPTR_T)(
                           FILE*  file,
                           char*  s1,
                           dim_t  m,
                           dim_t  n,
                           void*  x, inc_t rs_x, inc_t cs_x,
                           char*  format,
                           char*  s2
                         );

static FUNCPTR_T GENARRAY_I(ftypes,fprintm);


void bli_fprintm( FILE* file, char* s1, obj_t* x, char* format, char* s2 )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	dim_t     m         = bli_obj_length( *x );
	dim_t     n         = bli_obj_width( *x );

	inc_t     rs_x      = bli_obj_row_stride( *x );
	inc_t     cs_x      = bli_obj_col_stride( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	FUNCPTR_T f;

	if ( bli_error_checking_is_enabled() )
		bli_fprintm_check( file, s1, x, format, s2 );

	// Handle constants up front.
	if ( dt_x == BLIS_CONSTANT )
	{
		float*    sp = bli_obj_buffer_for_const( BLIS_FLOAT,    *x );
		double*   dp = bli_obj_buffer_for_const( BLIS_DOUBLE,   *x );
		scomplex* cp = bli_obj_buffer_for_const( BLIS_SCOMPLEX, *x );
		dcomplex* zp = bli_obj_buffer_for_const( BLIS_DCOMPLEX, *x );
		gint_t*   ip = bli_obj_buffer_for_const( BLIS_INT,      *x );

		fprintf( file, "%s\n", s1 );
		fprintf( file, " float:     %9.2e\n",         bli_sreal( *sp ) );
		fprintf( file, " double:    %9.2e\n",         bli_dreal( *dp ) );
		fprintf( file, " scomplex:  %9.2e + %9.2e\n", bli_creal( *cp ), bli_cimag( *cp ) );
		fprintf( file, " dcomplex:  %9.2e + %9.2e\n", bli_zreal( *zp ), bli_zimag( *zp ) );
		fprintf( file, " int:       %ld\n",           *ip );
		fprintf( file, "\n" );
		return;
	}

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x];

	// Invoke the function.
	f( file,
	   s1,
	   m,
	   n,
	   buf_x, rs_x, cs_x,
	   format,
	   s2 );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       FILE*  file, \
       char*  s1, \
       dim_t  m, \
       dim_t  n, \
       void*  x, inc_t rs_x, inc_t cs_x, \
       char*  format, \
       char*  s2  \
     ) \
{ \
	dim_t  i, j; \
	ctype* chi1; \
	char   default_spec[32] = PASTEMAC(ch,formatspec)(); \
\
	if ( format == NULL ) format = default_spec; \
\
	fprintf( file, "%s\n", s1 ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		for ( j = 0; j < n; ++j ) \
		{ \
			chi1 = (( ctype* ) x) + i*rs_x + j*cs_x; \
\
			PASTEMAC(ch,fprints)( file, format, *chi1 ); \
			fprintf( file, " " ); \
		} \
\
		fprintf( file, "\n" ); \
	} \
\
	fprintf( file, "%s\n", s2 ); \
	fflush( file ); \
}

INSERT_GENTFUNC_BASIC_I( fprintm, fprintm )

