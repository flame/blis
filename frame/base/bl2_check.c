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

#include "blis2.h"

// -- General stuff ------------------------------------------------------------

err_t bl2_check_error_code_helper( int code, char* file, unsigned int line )
{
	if ( code == BLIS_SUCCESS ) return code;

	if ( BLIS_ERROR_CODE_MAX < code && code < BLIS_ERROR_CODE_MIN )
	{
		bl2_print_msg( bl2_error_string_for_code( code ),
		               file, line );
		bl2_abort();
	}
	else
	{
		bl2_print_msg( bl2_error_string_for_code( BLIS_UNDEFINED_ERROR_CODE ),
		               file, line );
		bl2_abort();
	}

	return code;
}

err_t bl2_check_valid_error_level( errlev_t level )
{
	err_t e_val = BLIS_SUCCESS;

	if ( level != BLIS_NO_ERROR_CHECKING &&
	     level != BLIS_FULL_ERROR_CHECKING )
		e_val = BLIS_INVALID_ERROR_CHECKING_LEVEL;

	return e_val;
}

err_t bl2_check_null_pointer( void* ptr )
{
	err_t e_val = BLIS_SUCCESS;

	if ( ptr == NULL )
		e_val = BLIS_NULL_POINTER;

	return e_val;
}


// -- Parameter-related checks -------------------------------------------------

err_t bl2_check_valid_side( side_t side )
{
	err_t e_val = BLIS_SUCCESS;

	if ( side != BLIS_LEFT &&
	     side != BLIS_RIGHT /*&&
	     side != BLIS_TOP &&
	     side != BLIS_BOTTOM*/ )
		e_val = BLIS_INVALID_SIDE;

	return e_val;
}

err_t bl2_check_valid_uplo( uplo_t uplo )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_is_lower( uplo ) &&
	     !bl2_is_upper( uplo ) )
		e_val = BLIS_INVALID_UPLO;

	return e_val;
}

err_t bl2_check_valid_trans( trans_t trans )
{
	err_t e_val = BLIS_SUCCESS;

	if ( trans != BLIS_NO_TRANSPOSE &&
	     trans != BLIS_TRANSPOSE &&
	     trans != BLIS_CONJ_NO_TRANSPOSE &&
	     trans != BLIS_CONJ_TRANSPOSE )
		e_val = BLIS_INVALID_TRANS;

	return e_val;
}

err_t bl2_check_valid_diag( diag_t diag )
{
	err_t e_val = BLIS_SUCCESS;

	if ( diag != BLIS_NONUNIT_DIAG &&
	     diag != BLIS_UNIT_DIAG )
		e_val = BLIS_INVALID_DIAG;

	return e_val;
}

err_t bl2_check_nonunit_diag( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_has_nonunit_diag( *a ) )
		e_val = BLIS_EXPECTED_NONUNIT_DIAG;

	return e_val;
}

// -- Datatype-related checks --------------------------------------------------

err_t bl2_check_valid_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_FLOAT &&
	     dt != BLIS_DOUBLE &&
	     dt != BLIS_SCOMPLEX &&
	     dt != BLIS_DCOMPLEX &&
	     dt != BLIS_INT &&
	     dt != BLIS_CONSTANT )
		e_val = BLIS_INVALID_DATATYPE;

	return e_val;
}

err_t bl2_check_object_valid_datatype( obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bl2_obj_datatype( *a );
	e_val = bl2_check_valid_datatype( dt );

	return e_val;
}

err_t bl2_check_noninteger_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt == BLIS_INT )
		e_val = BLIS_EXPECTED_NONINTEGER_DATATYPE;

	return e_val;
}

err_t bl2_check_noninteger_object( obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bl2_obj_datatype( *a );
	e_val = bl2_check_noninteger_datatype( dt );

	return e_val;
}

err_t bl2_check_nonconstant_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt == BLIS_CONSTANT )
		e_val = BLIS_EXPECTED_NONCONSTANT_DATATYPE;

	return e_val;
}

err_t bl2_check_nonconstant_object( obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bl2_obj_datatype( *a );
	e_val = bl2_check_nonconstant_datatype( dt );

	return e_val;
}

err_t bl2_check_floating_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_FLOAT &&
	     dt != BLIS_DOUBLE &&
	     dt != BLIS_SCOMPLEX &&
	     dt != BLIS_DCOMPLEX )
		e_val = BLIS_EXPECTED_FLOATING_POINT_DATATYPE;

	return e_val;
}

err_t bl2_check_floating_object( obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bl2_obj_datatype( *a );
	e_val = bl2_check_floating_datatype( dt );

	return e_val;
}

err_t bl2_check_real_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_FLOAT &&
	     dt != BLIS_DOUBLE )
		e_val = BLIS_EXPECTED_REAL_DATATYPE;

	return e_val;
}

err_t bl2_check_real_object( obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bl2_obj_datatype( *a );
	e_val = bl2_check_real_datatype( dt );

	return e_val;
}

err_t bl2_check_consistent_datatypes( num_t dt_a, num_t dt_b )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt_a != BLIS_CONSTANT &&
	     dt_b != BLIS_CONSTANT )
		if ( dt_a != dt_b )
			e_val = BLIS_INCONSISTENT_DATATYPES;

	return e_val;
}

err_t bl2_check_consistent_object_datatypes( obj_t* a, obj_t* b )
{
	err_t e_val;
	num_t dt_a;
	num_t dt_b;

	dt_a = bl2_obj_datatype( *a );
	dt_b = bl2_obj_datatype( *b );

	e_val = bl2_check_consistent_datatypes( dt_a, dt_b );

	return e_val;
}

err_t bl2_check_datatype_real_proj_of( num_t dt_c, num_t dt_r )
{
	err_t e_val = BLIS_SUCCESS;

	if ( ( dt_c == BLIS_CONSTANT && bl2_is_complex( dt_r ) ) ||
	     ( dt_c == BLIS_FLOAT    && dt_r != BLIS_FLOAT     ) ||
	     ( dt_c == BLIS_DOUBLE   && dt_r != BLIS_DOUBLE    ) ||
	     ( dt_c == BLIS_SCOMPLEX && dt_r != BLIS_FLOAT     ) ||
	     ( dt_c == BLIS_DCOMPLEX && dt_r != BLIS_DOUBLE    ) )
		e_val = BLIS_EXPECTED_REAL_PROJ_OF;

	return e_val;
}

err_t bl2_check_object_real_proj_of( obj_t* c, obj_t* r )
{
	err_t e_val;
	num_t dt_c;
	num_t dt_r;

	dt_c = bl2_obj_datatype( *c );
	dt_r = bl2_obj_datatype( *r );

	e_val = bl2_check_datatype_real_proj_of( dt_c, dt_r );

	return e_val;
}

err_t bl2_check_real_valued_object( obj_t* a )
{
	err_t  e_val = BLIS_SUCCESS;
	double a_real;
	double a_imag;

	bl2_getsc( a, &a_real, &a_imag );

	if ( a_imag != 0.0 )
		e_val = BLIS_EXPECTED_REAL_VALUED_OBJECT;

	return e_val;
}

// -- Dimension-related checks -------------------------------------------------

err_t bl2_check_conformal_dims( obj_t* a, obj_t* b )
{
	err_t e_val = BLIS_SUCCESS;
	dim_t m_a, n_a;
	dim_t m_b, n_b;

	m_a = bl2_obj_length_after_trans( *a );
	n_a = bl2_obj_width_after_trans( *a );
	m_b = bl2_obj_length_after_trans( *b );
	n_b = bl2_obj_width_after_trans( *b );

	if ( m_a != m_b || n_a != n_b )
		e_val = BLIS_NONCONFORMAL_DIMENSIONS;

	return e_val;
}

err_t bl2_check_level3_dims( obj_t* a, obj_t* b, obj_t* c )
{
	err_t e_val = BLIS_SUCCESS;
	dim_t m_c, n_c;
	dim_t m_a, k_a;
	dim_t k_b, n_b;

	m_c = bl2_obj_length_after_trans( *c );
	n_c = bl2_obj_width_after_trans( *c );

	m_a = bl2_obj_length_after_trans( *a );
	k_a = bl2_obj_width_after_trans( *a );

	k_b = bl2_obj_length_after_trans( *b );
	n_b = bl2_obj_width_after_trans( *b );

	if ( m_c != m_a ||
	     n_c != n_b ||
	     k_a != k_b )
		e_val = BLIS_NONCONFORMAL_DIMENSIONS;

	return e_val;
}

err_t bl2_check_scalar_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_length( *a ) != 1 ||
	     bl2_obj_width( *a )  != 1 )
		e_val = BLIS_EXPECTED_SCALAR_OBJECT;

	return e_val;
}

err_t bl2_check_vector_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_is_vector( *a ) )
		e_val = BLIS_EXPECTED_VECTOR_OBJECT;

	return e_val;
}

err_t bl2_check_equal_vector_lengths( obj_t* x, obj_t* y )
{
	err_t e_val = BLIS_SUCCESS;
	dim_t dim_x;
	dim_t dim_y;

	dim_x = bl2_obj_vector_dim( *x );
	dim_y = bl2_obj_vector_dim( *y );

	if ( dim_x != dim_y )
		e_val = BLIS_UNEQUAL_VECTOR_LENGTHS;

	return e_val;
}

err_t bl2_check_square_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_length( *a ) != bl2_obj_width( *a ) )
		e_val = BLIS_EXPECTED_SQUARE_OBJECT;

	return e_val;
}

err_t bl2_check_object_length_equals( obj_t* a, dim_t m )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_length( *a ) != m )
		e_val = BLIS_UNEXPECTED_OBJECT_LENGTH;

	return e_val;
}

err_t bl2_check_object_width_equals( obj_t* a, dim_t n )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_width( *a ) != n )
		e_val = BLIS_UNEXPECTED_OBJECT_WIDTH;

	return e_val;
}

err_t bl2_check_vector_dim_equals( obj_t* a, dim_t n )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_vector_dim( *a ) != n )
		e_val = BLIS_UNEXPECTED_VECTOR_DIM;

	return e_val;
}

err_t bl2_check_object_diag_offset_equals( obj_t* a, doff_t offset )
{
	err_t e_val = BLIS_SUCCESS;

	if ( offset != bl2_obj_diag_offset( *a ) )
		e_val = BLIS_UNEXPECTED_DIAG_OFFSET;

	return e_val;
}

// -- Stride-related checks ----------------------------------------------------

err_t bl2_check_matrix_strides( dim_t m, dim_t n, dim_t rs, dim_t cs )
{
	err_t e_val = BLIS_SUCCESS;

	// Note: A lot of thought went into designing these checks. Do NOT change
	// them unless you absolutely know what you are doing! Particularly, do
	// not try to merge the general and row-/column-major sections. It might
	// be possible, but it would be a lot less readable.

	// Note: The default case (whereby we interpret rs == cs == 0 as a request
	// for column-major order) is handled prior to calling this function, so
	// we never see zero strides here.

	// Disallow either of the strides to be zero.
	if ( ( rs == 0 || cs == 0 ) )
		return BLIS_INVALID_DIM_STRIDE_COMBINATION;

	// Check stride consistency in cases of general stride.
	if ( rs != 1 && cs != 1 )
	{
		// We apply different tests depending on which way the strides
		// "tilt".
		if ( rs == cs )
		{
			// If rs == cs, then we must be dealing with an m-by-1 or a
			// 1-by-n matrix and thus at least one of the dimensions, m
			// or n, must be unit (even if the other is zero).
			if ( m != 1 && n != 1 )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
		else if ( rs < cs )
		{
			// For column-major tilt, cs must be equal or larger than m * rs.
			if ( m * rs > cs )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
		else if ( cs < rs )
		{
			// For row-major tilt, rs must be equal or larger than n * cs.
			if ( n * cs > rs )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
	}
	else // check stride consistency of row-/column-storage cases.
	{
		if ( rs == 1 && cs == 1 )
		{
			// Only allow rs == cs == 1 for scalars.
			if ( m > 1 || n > 1 )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
		else if ( rs == 1 )
		{
			// For column-major storage, don't allow the column stride to be
			// less than the m dimension.
			if ( cs < m )
				return BLIS_INVALID_COL_STRIDE;
		}
		else if ( cs == 1 )
		{
			// For row-major storage, don't allow the row stride to be less
			// than the n dimension.
			if ( rs < n )
				return BLIS_INVALID_ROW_STRIDE;
		}
	}

	return e_val;
}

// -- Structure-related checks -------------------------------------------------

err_t bl2_check_general_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_is_general( *a ) )
		e_val = BLIS_EXPECTED_GENERAL_OBJECT;

	return e_val;
}

err_t bl2_check_hermitian_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_is_hermitian( *a ) )
		e_val = BLIS_EXPECTED_HERMITIAN_OBJECT;

	return e_val;
}

err_t bl2_check_symmetric_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_is_symmetric( *a ) )
		e_val = BLIS_EXPECTED_SYMMETRIC_OBJECT;

	return e_val;
}

err_t bl2_check_triangular_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_is_triangular( *a ) )
		e_val = BLIS_EXPECTED_TRIANGULAR_OBJECT;

	return e_val;
}

// -- Storage-related checks ---------------------------------------------------

err_t bl2_check_upper_or_lower_object( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bl2_obj_is_lower( *a ) &&
	     !bl2_obj_is_upper( *a ) )
		e_val = BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT;

	return e_val;
}

// -- Partitioning-related checks ----------------------------------------------

err_t bl2_check_valid_3x1_subpart( subpart_t part )
{
	err_t e_val = BLIS_SUCCESS;

	if ( part != BLIS_SUBPART0 &&
	     part != BLIS_SUBPART1 &&
	     part != BLIS_SUBPART2 )
		e_val = BLIS_INVALID_3x1_SUBPART;

	return e_val;
}

err_t bl2_check_valid_1x3_subpart( subpart_t part )
{
	err_t e_val = BLIS_SUCCESS;

	if ( part != BLIS_SUBPART0 &&
	     part != BLIS_SUBPART1 &&
	     part != BLIS_SUBPART2 )
		e_val = BLIS_INVALID_1x3_SUBPART;

	return e_val;
}

err_t bl2_check_valid_3x3_subpart( subpart_t part )
{
	err_t e_val = BLIS_SUCCESS;

	if ( part != BLIS_SUBPART00 &&
	     part != BLIS_SUBPART10 &&
	     part != BLIS_SUBPART20 &&
	     part != BLIS_SUBPART01 &&
	     part != BLIS_SUBPART11 &&
	     part != BLIS_SUBPART21 &&
	     part != BLIS_SUBPART02 &&
	     part != BLIS_SUBPART12 &&
	     part != BLIS_SUBPART22 )
		e_val = BLIS_INVALID_3x3_SUBPART;

	return e_val;
}


// -- Control tree-related checks ----------------------------------------------

err_t bl2_check_valid_cntl( void* cntl )
{
	err_t e_val = BLIS_SUCCESS;

	if ( cntl == NULL )
		e_val = BLIS_UNEXPECTED_NULL_CONTROL_TREE;

	return e_val;
}

// -- Packing-related checks ---------------------------------------------------

err_t bl2_check_packm_schema_on_unpack( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_pack_status( *a ) != BLIS_PACKED_ROWS &&
	     bl2_obj_pack_status( *a ) != BLIS_PACKED_COLUMNS &&
	     bl2_obj_pack_status( *a ) != BLIS_PACKED_ROW_PANELS &&
	     bl2_obj_pack_status( *a ) != BLIS_PACKED_COL_PANELS )
		e_val = BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK;

	return e_val;
}

err_t bl2_check_packv_schema_on_unpack( obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bl2_obj_pack_status( *a ) != BLIS_PACKED_VECTOR )
		e_val = BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK;

	return e_val;
}


// -- Memory allocator checks --------------------------------------------------


