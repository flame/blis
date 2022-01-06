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

void bli_gemm_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	//err_t e_val;

	// Check basic properties of the operation.

	bli_gemm_basic_check( alpha, a, b, beta, c, cntx );

	// Check object structure.

	// NOTE: Can't perform these checks as long as bli_gemm_check() is called
	// from bli_l3_int(), which is in the execution path for structured
	// level-3 operations such as hemm.

	//e_val = bli_check_general_object( a );
	//bli_check_error_code( e_val );

	//e_val = bli_check_general_object( b );
	//bli_check_error_code( e_val );
}

void bli_gemmt_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Check basic properties of the operation.

	bli_gemmt_basic_check( alpha, a, b, beta, c, cntx );

	// Check matrix squareness.

	e_val = bli_check_square_object( c );
	bli_check_error_code( e_val );
}

void bli_hemm_check
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform checks common to hemm/symm/trmm/trsm.

	bli_hemm_basic_check( side, alpha, a, b, beta, c, cntx );

	// Check object structure.

	e_val = bli_check_hermitian_object( a );
	bli_check_error_code( e_val );
}

void bli_herk_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;
	obj_t ah;

	// Alias A to A^H so we can perform dimension checks.
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, &ah );

	// Check basic properties of the operation.

	bli_herk_basic_check( alpha, a, &ah, beta, c, cntx );

	// Check for real-valued alpha and beta.

	e_val = bli_check_real_valued_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_real_valued_object( beta );
	bli_check_error_code( e_val );

	// Check matrix structure.

	e_val = bli_check_hermitian_object( c );
	bli_check_error_code( e_val );
}

void bli_her2k_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;
	obj_t ah, bh;

	// Alias A and B to A^H and B^H so we can perform dimension checks.
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, &ah );
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, b, &bh );

	// Check basic properties of the operation.

	bli_her2k_basic_check( alpha, a, &bh, b, &ah, beta, c, cntx );

	// Check for real-valued beta.

	e_val = bli_check_real_valued_object( beta );
	bli_check_error_code( e_val );

	// Check matrix structure.

	e_val = bli_check_hermitian_object( c );
	bli_check_error_code( e_val );
}

void bli_symm_check
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Check basic properties of the operation.

	bli_hemm_basic_check( side, alpha, a, b, beta, c, cntx );

	// Check object structure.

	e_val = bli_check_symmetric_object( a );
	bli_check_error_code( e_val );
}

void bli_syrk_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;
	obj_t at;

	// Alias A to A^T so we can perform dimension checks.
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, a, &at );

	// Check basic properties of the operation.

	bli_herk_basic_check( alpha, a, &at, beta, c, cntx );

	// Check matrix structure.

	e_val = bli_check_symmetric_object( c );
	bli_check_error_code( e_val );
}

void bli_syr2k_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;
	obj_t at, bt;

	// Alias A and B to A^T and B^T so we can perform dimension checks.
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, a, &at );
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, b, &bt );

	// Check basic properties of the operation.

	bli_her2k_basic_check( alpha, a, &bt, b, &at, beta, c, cntx );

	// Check matrix structure.

	e_val = bli_check_symmetric_object( c );
	bli_check_error_code( e_val );
}

void bli_trmm3_check
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform checks common to hemm/symm/trmm/trsm.

	bli_hemm_basic_check( side, alpha, a, b, beta, c, cntx );

	// Check object structure.

	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );
}

void bli_trmm_check
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform checks common to hemm/symm/trmm/trsm.

	bli_hemm_basic_check( side, alpha, a, b, &BLIS_ZERO, b, cntx );

	// Check object structure.

	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );
}

void bli_trsm_check
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform checks common to hemm/symm/trmm/trsm.

	bli_hemm_basic_check( side, alpha, a, b, &BLIS_ZERO, b, cntx );

	// Check object structure.

	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );
}

// -----------------------------------------------------------------------------

void bli_gemm_basic_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform standard checks.

	bli_l3_basic_check( alpha, a, b, beta, c, cntx );

	// Check object dimensions.

	e_val = bli_check_level3_dims( a, b, c );
	bli_check_error_code( e_val );

#ifdef BLIS_ENABLE_GEMM_MD
	// Skip checking for consistent datatypes between A, B, and C since
	// that is totally valid for mixed-datatype gemm.

	// When mixing datatypes, make sure that alpha does not have a non-zero
	// imaginary component.
	if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	     bli_obj_dt( c ) != bli_obj_dt( b ) ||
	     bli_obj_comp_prec( c ) != bli_obj_prec( c ) )
	if ( !bli_obj_imag_is_zero( alpha ) )
	{
		bli_print_msg( "Mixed-datatype gemm does not yet support alpha with a non-zero imaginary component. Please contact BLIS developers for further support.", __FILE__, __LINE__ );
		bli_abort();
	}

#else // BLIS_DISABLE_GEMM_MD

	// Check for consistent datatypes.
	// NOTE: We only perform these tests when mixed datatype support is
	// disabled.

	e_val = bli_check_consistent_object_datatypes( c, a );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, b );
	bli_check_error_code( e_val );
#endif
}

void bli_gemmt_basic_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform standard checks.

	bli_l3_basic_check( alpha, a, b, beta, c, cntx );

	// Check object dimensions.

	e_val = bli_check_level3_dims( a, b, c );
	bli_check_error_code( e_val );
}

void bli_hemm_basic_check
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform standard checks.

	bli_l3_basic_check( alpha, a, b, beta, c, cntx );

	// Check object dimensions.

	if ( bli_is_left( side ) )
	{
		e_val = bli_check_level3_dims( a, b, c );
		bli_check_error_code( e_val );
	}
	else // if ( bli_is_right( side ) )
	{
		e_val = bli_check_level3_dims( b, a, c );
		bli_check_error_code( e_val );
	}

	// Check matrix squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( c, a );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, b );
	bli_check_error_code( e_val );
}

void bli_herk_basic_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  ah,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform standard checks.

	bli_l3_basic_check( alpha, a, ah, beta, c, cntx );

	// Check object dimensions.

	e_val = bli_check_level3_dims( a, ah, c );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( c );
	bli_check_error_code( e_val );

	// Check matrix structure.

	e_val = bli_check_general_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_general_object( ah );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( c, a );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, ah );
	bli_check_error_code( e_val );
}

void bli_her2k_basic_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  bh,
       obj_t*  b,
       obj_t*  ah,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Perform standard checks.

	bli_l3_basic_check( alpha, a, bh, beta, c, cntx );
	bli_l3_basic_check( alpha, b, ah, beta, c, cntx );

	// Check object dimensions.

	e_val = bli_check_level3_dims( a, bh, c );
	bli_check_error_code( e_val );

	e_val = bli_check_level3_dims( b, ah, c );
	bli_check_error_code( e_val );

	// Check matrix squareness.

	e_val = bli_check_square_object( c );
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

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( c, a );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, ah );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, b );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, bh );
	bli_check_error_code( e_val );
}

void bli_l3_basic_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( b );
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

	e_val = bli_check_matrix_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( c );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( b );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( c );
	bli_check_error_code( e_val );
}

