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

bool_t bl2_obj_scalar_equals( obj_t* a,
                              obj_t* b )
{
	bool_t r_val = FALSE;
	num_t  dt_a;
	num_t  dt_b;
	num_t  dt;
	void*  buf_a;
	void*  buf_b;
/*
bl2_printm( "a:", a, "%9.2e", "" );
bl2_printm( "b:", b, "%9.2e", "" );
*/
	dt_a = bl2_obj_datatype( *a );
	dt_b = bl2_obj_datatype( *b );

	if ( dt_a != BLIS_CONSTANT ) dt = dt_a;
	else                         dt = dt_b;

	buf_a = bl2_obj_scalar_buffer( dt, *a );
	buf_b = bl2_obj_scalar_buffer( dt, *b );
/*
printf( "dt:   %u\n", dt );
printf( "dt_a: %u\n", dt_a );
printf( "dt_b: %u\n", dt_b );
printf( "bufa: %p\n", buf_a );
printf( "bufb: %p\n", buf_b );
*/
	if      ( dt == BLIS_CONSTANT )
	{
		r_val = r_val || ( *BLIS_CONST_S_PTR( *a ) == *BLIS_CONST_S_PTR( *b ) );
		r_val = r_val || ( *BLIS_CONST_D_PTR( *a ) == *BLIS_CONST_D_PTR( *b ) );
		r_val = r_val || (  BLIS_CONST_C_PTR( *a )->real == BLIS_CONST_C_PTR( *b )->real &&
                            BLIS_CONST_C_PTR( *a )->imag == BLIS_CONST_C_PTR( *b )->imag );
		r_val = r_val || (  BLIS_CONST_Z_PTR( *a )->real == BLIS_CONST_Z_PTR( *b )->real &&
                            BLIS_CONST_Z_PTR( *a )->imag == BLIS_CONST_Z_PTR( *b )->imag );
	}
	else if ( dt == BLIS_FLOAT )    r_val = bl2_seqa( buf_a, buf_b );
	else if ( dt == BLIS_DOUBLE )   r_val = bl2_deqa( buf_a, buf_b );
	else if ( dt == BLIS_SCOMPLEX ) r_val = bl2_ceqa( buf_a, buf_b );
	else if ( dt == BLIS_DCOMPLEX ) r_val = bl2_zeqa( buf_a, buf_b );
	else if ( dt == BLIS_INT )      r_val = bl2_ieqa( buf_a, buf_b );

	return r_val;
}

