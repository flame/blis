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

#define FUNCPTR_T machval_fp

typedef void (*FUNCPTR_T)(
                           machval_t machval,
                           void*     val
                         );

// Manually initialize a function pointer array.
static FUNCPTR_T ftypes[BLIS_NUM_FP_TYPES] = 
{
	bl2_smachval,
	NULL,
	bl2_dmachval,
	NULL
};


//
// Define object-based interface.
//
void bl2_machval( machval_t machval,
                  obj_t*    v )
{
	num_t     dt_v  = bl2_obj_datatype( *v );

	void*     buf_v = bl2_obj_buffer_at_off( *v );

	FUNCPTR_T f;

	// Index into the function pointer array.
	f = ftypes[dt_v];

	// Invoke the function.
	f( machval,
	   buf_v );
}


//
// Define BLAS-like interfaces.
//
#undef  GENTFUNC
#define GENTFUNC3( ctype, ctype_r, ch, chr, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          machvar_t machval, \
                          ctype*    val, \
                        ) \
{ \
	static ctype_r pvals[ BLIS_NUM_MACH_PARAMS ]; \
	static bool_t  first_time = TRUE; \
	dim_t          val_i      = machval - BLIS_MACH_PARAM_FIRST; \
\
	/* If this is the first time through, call the underlying
	   code to discover each machine parameter. */ \
	if ( first_time ) \
	{ \
		char  lapack_machval; \
		dim_t i; \
\
		for( m = BLIS_MACH_PARAM_FIRST, i = 0; \
		     m <= BLIS_MACH_PARAM_LAST; \
		     ++m, ++i ) \
		{ \
			bl2_param_map_to_netlib_machval( m, &lapack_machval ); \
\
			/*printf( "bl2_machval: querying %u %c\n", m, lapack_machval );*/ \
\
			pvals[i] = PASTEMAC(chr,varname)( &lapack_machval, 1 ); \
\
			/*printf( "bl2_machval: got back %34.29e\n", pvals[i] ); */ \
		} \
\
		/* Store epsilon^2 in the last element. */ \
		pvals[i] = pvals[0] * pvals[0]; \
\
		first_time = FALSE; \
	} \
\
	/* Copy the requested parameter value to the output buffer, which
	   may involve a demotion from the complex to real domain. */ \
	PASTEMAC2(chr,ch,copys)( pvals[ val_i ], \
	                         *val ); \
}


GENTFUNC( float,    float,  s, s, machval, lamch )
GENTFUNC( double,   double, d, d, machval, lamch )
GENTFUNC( scomplex, float,  c, s, machval, lamch )
GENTFUNC( dcomplex, double, z, d, machval, lamch )

