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

bool_t bli_obj_equals( obj_t* a,
                       obj_t* b )
{
	bool_t r_val = FALSE;
	num_t  dt_a;
	num_t  dt_b;
	num_t  dt;

	// The function is not yet implemented for vectors and matrices.
	if ( !bli_obj_is_1x1( *a ) ||
	     !bli_obj_is_1x1( *b ) )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	dt_a = bli_obj_datatype( *a );
	dt_b = bli_obj_datatype( *b );

	// If B is BLIS_CONSTANT, then we need to test equality based on the
	// datatype of A--this works even if A is also BLIS_CONSTANT. If B
	// is a regular non-constant type, then we should use its datatype
	// to test equality.
	if ( dt_b == BLIS_CONSTANT ) dt = dt_a;
	else                         dt = dt_b;

	// Now test equality based on the chosen datatype.
	if ( dt == BLIS_CONSTANT )
	{
		dcomplex* ap_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, *a );
		dcomplex* bp_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, *b );

		// We only test equality for one datatype (double complex) since
		// we expect either all fields within the constant to be equal or
		// none to be equal. Therefore, we can just test one of them.
		r_val = bli_zeqa( ap_z, bp_z );
	}
	else
	{
		void* buf_a = bli_obj_buffer_for_1x1( dt, *a );
		void* buf_b = bli_obj_buffer_for_1x1( dt, *b );

		if      ( dt == BLIS_FLOAT )    r_val = bli_seqa( buf_a, buf_b );
		else if ( dt == BLIS_DOUBLE )   r_val = bli_deqa( buf_a, buf_b );
		else if ( dt == BLIS_SCOMPLEX ) r_val = bli_ceqa( buf_a, buf_b );
		else if ( dt == BLIS_DCOMPLEX ) r_val = bli_zeqa( buf_a, buf_b );
		else if ( dt == BLIS_INT )      r_val = bli_ieqa( buf_a, buf_b );
	}

	return r_val;
}

