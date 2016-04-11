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

void bli_her2k_basic_check( obj_t*   alpha,
                            obj_t*   a,
                            obj_t*   bh,
                            obj_t*   alpha_conj,
                            obj_t*   b,
                            obj_t*   ah,
                            obj_t*   beta,
                            obj_t*   c )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( bh );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( alpha_conj );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( ah );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( c );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( ah );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( bh );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( c );
	bli_check_error_code( e_val );

	e_val = bli_check_level3_dims( a, bh, c );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( alpha_conj );
	bli_check_error_code( e_val );

	e_val = bli_check_level3_dims( b, ah, c );
	bli_check_error_code( e_val );

	// Check matrix structure.

	e_val = bli_check_general_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_general_object( bh );
	bli_check_error_code( e_val );

	e_val = bli_check_general_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_general_object( ah );
	bli_check_error_code( e_val );
}

void bli_her2k_check( obj_t*   alpha,
                      obj_t*   a,
                      obj_t*   b,
                      obj_t*   beta,
                      obj_t*   c )
{
	err_t e_val;
	obj_t ah, bh;

	// Alias A and B to A^H and B^H so we can perform dimension checks.
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, *a, ah );
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, *b, bh );

	// Check basic properties of the operation.

	bli_her2k_basic_check( alpha, a, &bh, alpha, b, &ah, beta, c );

	// Check for real-valued beta.

	e_val = bli_check_real_valued_object( beta );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( c );
	bli_check_error_code( e_val );

	// Check matrix structure.

	e_val = bli_check_hermitian_object( c );
	bli_check_error_code( e_val );
}

#if 0
void bli_her2k_int_check( obj_t*   alpha,
                          obj_t*   a,
                          obj_t*   bh,
                          obj_t*   alpha_conj,
                          obj_t*   b,
                          obj_t*   ah,
                          obj_t*   beta,
                          obj_t*   c,
                          her2k_t* cntl )
{
	err_t e_val;

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( bh );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( alpha_conj );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( b );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( ah );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( c );
	bli_check_error_code( e_val );

	// Check basic properties of the operation.

	bli_her2k_basic_check( alpha, a, bh, alpha_conj, b, ah, beta, c );

	// Check control tree pointer

	e_val = bli_check_valid_cntl( ( void* )cntl );
	bli_check_error_code( e_val );
}
#endif
