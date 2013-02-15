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

// Current error checking level.
static errlev_t bl2_err_chk_level = BLIS_FULL_ERROR_CHECKING;

// Internal array to hold error strings.
static char bl2_error_string[BLIS_MAX_NUM_ERR_MSGS][BLIS_MAX_ERR_MSG_LENGTH];


errlev_t bl2_error_checking_level()
{
    return bl2_err_chk_level;
}

errlev_t bl2_error_checking_level_set( errlev_t new_level )
{
    err_t    e_val;
    errlev_t old_level;

    e_val = bl2_check_valid_error_level( new_level );
    bl2_check_error_code( e_val );

    old_level = bl2_err_chk_level;

    bl2_err_chk_level = new_level;

    return old_level;
}

bool_t bl2_error_checking_is_enabled()
{
    return bl2_error_checking_level() != BLIS_NO_ERROR_CHECKING;
}

char* bl2_error_string_for_code( int code )
{
	return bl2_error_string[-code];
}

void bl2_abort( void )
{
	fprintf( stderr, "libblis: Aborting.\n" );
	//raise( SIGABRT );
	abort();
}

void bl2_print_msg( char* str, char* file, unsigned int line )
{
	fprintf( stderr, "\n" );
	fprintf( stderr, "libblis: %s (line %d):\n", file, line );
	fprintf( stderr, "libblis: %s\n", str );
	fflush( stderr );
}

void bl2_error_msgs_init( void )
{
	sprintf( bl2_error_string_for_code(BLIS_INVALID_ERROR_CHECKING_LEVEL),
	         "Invalid error checking level." );
	sprintf( bl2_error_string_for_code(BLIS_UNDEFINED_ERROR_CODE),
	         "Undefined error code." );
	sprintf( bl2_error_string_for_code(BLIS_NULL_POINTER),
	         "Encountered unexpected null pointer." );
	sprintf( bl2_error_string_for_code(BLIS_NOT_YET_IMPLEMENTED),
	         "Requested functionality not yet implemented." );

	sprintf( bl2_error_string_for_code(BLIS_INVALID_SIDE),
	         "Invalid side parameter value." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_UPLO),
	         "Invalid uplo parameter value." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_TRANS),
	         "Invalid trans parameter value." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_CONJ),
	         "Invalid conj parameter value." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_DIAG),
	         "Invalid diag parameter value." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_NONUNIT_DIAG),
	         "Expected object with non-unit diagonal." );

	sprintf( bl2_error_string_for_code(BLIS_INVALID_DATATYPE),
	         "Invalid datatype value." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_FLOATING_POINT_DATATYPE),
	         "Expected floating-point datatype value." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_NONINTEGER_DATATYPE),
	         "Expected non-integer datatype value." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_NONCONSTANT_DATATYPE),
	         "Expected non-constant datatype value." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_REAL_DATATYPE),
	         "Expected real datatype value." );
	sprintf( bl2_error_string_for_code(BLIS_INCONSISTENT_DATATYPES),
	         "Expected consistent datatypes (equal, or one being constant)." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_REAL_PROJ_OF),
	         "Expected second datatype to be real projection of first." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_REAL_VALUED_OBJECT),
	         "Expected real-valued object (ie: if complex, imaginary component equals zero)." );

	sprintf( bl2_error_string_for_code(BLIS_NONCONFORMAL_DIMENSIONS),
	         "Encountered non-conformal dimensions between objects." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_SCALAR_OBJECT),
	         "Expected scalar object." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_VECTOR_OBJECT),
	         "Expected vector object." );
	sprintf( bl2_error_string_for_code(BLIS_UNEQUAL_VECTOR_LENGTHS),
	         "Encountered unequal vector lengths." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_SQUARE_OBJECT),
	         "Expected square object." );
	sprintf( bl2_error_string_for_code(BLIS_UNEXPECTED_OBJECT_LENGTH),
	         "Unexpected object length." );
	sprintf( bl2_error_string_for_code(BLIS_UNEXPECTED_OBJECT_WIDTH),
	         "Unexpected object width." );
	sprintf( bl2_error_string_for_code(BLIS_UNEXPECTED_VECTOR_DIM),
	         "Unexpected vector dimension." );
	sprintf( bl2_error_string_for_code(BLIS_UNEXPECTED_DIAG_OFFSET),
	         "Unexpected object diagonal offset." );

	sprintf( bl2_error_string_for_code(BLIS_INVALID_ROW_STRIDE),
	         "Encountered invalid row stride relative to n dimension." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_COL_STRIDE),
	         "Encountered invalid col stride relative to m dimension." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_DIM_STRIDE_COMBINATION),
	         "Encountered invalid stride/dimension combination." );

	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_GENERAL_OBJECT),
	         "Expected general object." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_HERMITIAN_OBJECT),
	         "Expected Hermitian object." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_SYMMETRIC_OBJECT),
	         "Expected symmetric object." );
	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_TRIANGULAR_OBJECT),
	         "Expected triangular object." );

	sprintf( bl2_error_string_for_code(BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT),
	         "Expected upper or lower triangular object." );

	sprintf( bl2_error_string_for_code(BLIS_INVALID_3x1_SUBPART),
	         "Encountered invalid 3x1 (vertical) subpartition label." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_1x3_SUBPART),
	         "Encountered invalid 1x3 (horizontal) subpartition label." );
	sprintf( bl2_error_string_for_code(BLIS_INVALID_3x3_SUBPART),
	         "Encountered invalid 3x3 (diagonal) subpartition label." );

	sprintf( bl2_error_string_for_code(BLIS_UNEXPECTED_NULL_CONTROL_TREE),
	         "Encountered unexpected null control tree node." );

	sprintf( bl2_error_string_for_code(BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK),
	         "Pack schema not yet supported/implemented for use with unpacking." );

	sprintf( bl2_error_string_for_code(BLIS_EXHAUSTED_STATIC_MEMORY_POOL),
	         "Attempted to allocate more memory from static pool than is available." );
}

