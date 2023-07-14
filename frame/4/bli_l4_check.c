/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#ifdef BLIS_ENABLE_LEVEL4

void bli_chol_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	bool is_herm      = bli_obj_is_hermitian( a );
	bool is_real_symm = bli_obj_is_symmetric( a ) && bli_obj_is_real( a );

	if ( !is_herm && !is_real_symm )
	{
		e_val = BLIS_EXPECTED_HERMITIAN_OBJECT;
		bli_check_error_code( e_val );
	}

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}

void bli_trinv_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.
#if 0
	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );
#endif

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}

void bli_ttmm_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.
#if 0
	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );
#endif

	// Check object buffers (for non-NULLness).
	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}

void bli_hpdinv_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	bool is_herm      = bli_obj_is_hermitian( a );
	bool is_real_symm = bli_obj_is_symmetric( a ) && bli_obj_is_real( a );

	if ( !is_herm && !is_real_symm )
	{
		e_val = BLIS_EXPECTED_HERMITIAN_OBJECT;
		bli_check_error_code( e_val );
	}

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}

void bli_hevd_check
     (
       const obj_t*  a,
       const obj_t*  v,
       const obj_t*  e,
       const cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( v );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( e );
	bli_check_error_code( e_val );

	e_val = bli_check_real_object( e );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, v );
	bli_check_error_code( e_val );

	e_val = bli_check_object_real_proj_of( v, e );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( v );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( e );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_square_object( v );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_conformal_dims( a, v );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( e, bli_obj_length( v ) );
	bli_check_error_code( e_val );

	// Check object structure.

	bool is_herm      = bli_obj_is_hermitian( a );
	bool is_real_symm = bli_obj_is_symmetric( a ) && bli_obj_is_real( a );

	if ( !is_herm && !is_real_symm )
	{
		e_val = BLIS_EXPECTED_HERMITIAN_OBJECT;
		bli_check_error_code( e_val );
	}

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( v );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( e );
	bli_check_error_code( e_val );
}

void bli_rhevd_check
     (
       const obj_t*  v,
       const obj_t*  e,
       const obj_t*  a,
       const cntx_t* cntx
     )
{
	bli_hevd_check( a, v, e, cntx );
}

void bli_hevpinv_check
     (
             double  thresh,
       const obj_t*  a,
       const obj_t*  p,
       const cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( p );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, p );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( p );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_square_object( p );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_conformal_dims( a, p );
	bli_check_error_code( e_val );

	// Check object structure.

	bool is_herm      = bli_obj_is_hermitian( a );
	bool is_real_symm = bli_obj_is_symmetric( a ) && bli_obj_is_real( a );

	if ( !is_herm && !is_real_symm )
	{
		e_val = BLIS_EXPECTED_HERMITIAN_OBJECT;
		bli_check_error_code( e_val );
	}

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( p );
	bli_check_error_code( e_val );
}

#endif
