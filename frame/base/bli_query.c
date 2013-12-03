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

bool_t bli_obj_equals( obj_t* a,
                       obj_t* b )
{
	bool_t r_val = FALSE;
	num_t  dt_a;
	num_t  dt_b;
	num_t  dt;
	void*  buf_a;
	void*  buf_b;

	// The function is not yet implemented for vectors and matrices.
	if ( !bli_obj_is_1x1( *a ) ||
	     !bli_obj_is_1x1( *b ) )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
/*
bli_printm( "a:", a, "%9.2e", "" );
bli_printm( "b:", b, "%9.2e", "" );
*/
	dt_a = bli_obj_datatype( *a );
	dt_b = bli_obj_datatype( *b );

	//if      ( dt_a != BLIS_CONSTANT && dt_b != BLIS_CONSTANT ) dt = dt_a;
	//else if ( dt_a != BLIS_CONSTANT && dt_b == BLIS_CONSTANT ) dt = dt_a;
	//else if ( dt_a == BLIS_CONSTANT && dt_b != BLIS_CONSTANT ) dt = dt_b;
	//else if ( dt_a == BLIS_CONSTANT && dt_b == BLIS_CONSTANT ) dt = dt_a;

	if ( dt_b == BLIS_CONSTANT ) dt = dt_a;
	else                         dt = dt_b;

	buf_a = bli_obj_buffer_for_1x1( dt, *a );
	buf_b = bli_obj_buffer_for_1x1( dt, *b );
/*
printf( "dt:   %u\n", dt );
printf( "dt_a: %u\n", dt_a );
printf( "dt_b: %u\n", dt_b );
printf( "bufa: %p\n", buf_a );
printf( "bufb: %p\n", buf_b );
*/
	if      ( dt == BLIS_CONSTANT )
	{
		float*    ap_s = bli_obj_buffer_for_const( BLIS_FLOAT,    *a );
		double*   ap_d = bli_obj_buffer_for_const( BLIS_DOUBLE,   *a );
		scomplex* ap_c = bli_obj_buffer_for_const( BLIS_SCOMPLEX, *a );
		dcomplex* ap_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, *a );

		float*    bp_s = bli_obj_buffer_for_const( BLIS_FLOAT,    *b );
		double*   bp_d = bli_obj_buffer_for_const( BLIS_DOUBLE,   *b );
		scomplex* bp_c = bli_obj_buffer_for_const( BLIS_SCOMPLEX, *b );
		dcomplex* bp_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, *b );

		r_val = r_val || bli_seqa( ap_s, bp_s );
		r_val = r_val || bli_deqa( ap_d, bp_d );
		r_val = r_val || bli_ceqa( ap_c, bp_c );
		r_val = r_val || bli_zeqa( ap_z, bp_z );
	}
	else if ( dt == BLIS_FLOAT )    r_val = bli_seqa( buf_a, buf_b );
	else if ( dt == BLIS_DOUBLE )   r_val = bli_deqa( buf_a, buf_b );
	else if ( dt == BLIS_SCOMPLEX ) r_val = bli_ceqa( buf_a, buf_b );
	else if ( dt == BLIS_DCOMPLEX ) r_val = bli_zeqa( buf_a, buf_b );
	else if ( dt == BLIS_INT )      r_val = bli_ieqa( buf_a, buf_b );

	return r_val;
}

