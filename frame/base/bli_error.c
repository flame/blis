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

static bool_t bli_error_is_init = FALSE;

void bli_error_init( void )
{
	// If the API is already initialized, return early.
	if ( bli_error_is_initialized() ) return;

	bli_error_init_msgs();

	// Mark API as initialized.
	bli_error_is_init = TRUE;
}

void bli_error_finalize( void )
{
	// Mark API as uninitialized.
	bli_error_is_init = FALSE;
}

bool_t bli_error_is_initialized( void )
{
	return bli_error_is_init;
}

// -----------------------------------------------------------------------------

// Internal array to hold error strings.
static char bli_error_string[BLIS_MAX_NUM_ERR_MSGS][BLIS_MAX_ERR_MSG_LENGTH];

void bli_error_init_msgs( void )
{
	sprintf( bli_error_string_for_code(BLIS_INVALID_ERROR_CHECKING_LEVEL),
	         "Invalid error checking level." );
	sprintf( bli_error_string_for_code(BLIS_UNDEFINED_ERROR_CODE),
	         "Undefined error code." );
	sprintf( bli_error_string_for_code(BLIS_NULL_POINTER),
	         "Encountered unexpected null pointer." );
	sprintf( bli_error_string_for_code(BLIS_NOT_YET_IMPLEMENTED),
	         "Requested functionality not yet implemented." );

	sprintf( bli_error_string_for_code(BLIS_INVALID_SIDE),
	         "Invalid side parameter value." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_UPLO),
	         "Invalid uplo_t parameter value." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_TRANS),
	         "Invalid trans_t parameter value." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_CONJ),
	         "Invalid conj_t parameter value." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_DIAG),
	         "Invalid diag_t parameter value." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_NONUNIT_DIAG),
	         "Expected object with non-unit diagonal." );

	sprintf( bli_error_string_for_code(BLIS_INVALID_DATATYPE),
	         "Invalid datatype value." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_FLOATING_POINT_DATATYPE),
	         "Expected floating-point datatype value." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_NONINTEGER_DATATYPE),
	         "Expected non-integer datatype value." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_NONCONSTANT_DATATYPE),
	         "Expected non-constant datatype value." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_REAL_DATATYPE),
	         "Expected real datatype value." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_INTEGER_DATATYPE),
	         "Expected integer datatype value." );
	sprintf( bli_error_string_for_code(BLIS_INCONSISTENT_DATATYPES),
	         "Expected consistent datatypes (equal, or one being constant)." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_REAL_PROJ_OF),
	         "Expected second datatype to be real projection of first." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_REAL_VALUED_OBJECT),
	         "Expected real-valued object (ie: if complex, imaginary component equals zero)." );

	sprintf( bli_error_string_for_code(BLIS_NONCONFORMAL_DIMENSIONS),
	         "Encountered non-conformal dimensions between objects." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_SCALAR_OBJECT),
	         "Expected scalar object." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_VECTOR_OBJECT),
	         "Expected vector object." );
	sprintf( bli_error_string_for_code(BLIS_UNEQUAL_VECTOR_LENGTHS),
	         "Encountered unequal vector lengths." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_SQUARE_OBJECT),
	         "Expected square object." );
	sprintf( bli_error_string_for_code(BLIS_UNEXPECTED_OBJECT_LENGTH),
	         "Unexpected object length." );
	sprintf( bli_error_string_for_code(BLIS_UNEXPECTED_OBJECT_WIDTH),
	         "Unexpected object width." );
	sprintf( bli_error_string_for_code(BLIS_UNEXPECTED_VECTOR_DIM),
	         "Unexpected vector dimension." );
	sprintf( bli_error_string_for_code(BLIS_UNEXPECTED_DIAG_OFFSET),
	         "Unexpected object diagonal offset." );
	sprintf( bli_error_string_for_code(BLIS_NEGATIVE_DIMENSION),
	         "Encountered negative dimension." );

	sprintf( bli_error_string_for_code(BLIS_INVALID_ROW_STRIDE),
	         "Encountered invalid row stride relative to n dimension." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_COL_STRIDE),
	         "Encountered invalid col stride relative to m dimension." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_DIM_STRIDE_COMBINATION),
	         "Encountered invalid stride/dimension combination." );

	sprintf( bli_error_string_for_code(BLIS_EXPECTED_GENERAL_OBJECT),
	         "Expected general object." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_HERMITIAN_OBJECT),
	         "Expected Hermitian object." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_SYMMETRIC_OBJECT),
	         "Expected symmetric object." );
	sprintf( bli_error_string_for_code(BLIS_EXPECTED_TRIANGULAR_OBJECT),
	         "Expected triangular object." );

	sprintf( bli_error_string_for_code(BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT),
	         "Expected upper or lower triangular object." );

	sprintf( bli_error_string_for_code(BLIS_INVALID_3x1_SUBPART),
	         "Encountered invalid 3x1 (vertical) subpartition label." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_1x3_SUBPART),
	         "Encountered invalid 1x3 (horizontal) subpartition label." );
	sprintf( bli_error_string_for_code(BLIS_INVALID_3x3_SUBPART),
	         "Encountered invalid 3x3 (diagonal) subpartition label." );

	sprintf( bli_error_string_for_code(BLIS_UNEXPECTED_NULL_CONTROL_TREE),
	         "Encountered unexpected null control tree node." );

	sprintf( bli_error_string_for_code(BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK),
	         "Pack schema not yet supported/implemented for use with unpacking." );

	sprintf( bli_error_string_for_code(BLIS_EXPECTED_NONNULL_OBJECT_BUFFER),
	         "Encountered object with non-zero dimensions containing null buffer." );

	sprintf( bli_error_string_for_code(BLIS_INVALID_PACKBUF),
	         "Invalid packbuf_t value." );
	sprintf( bli_error_string_for_code(BLIS_REQUESTED_CONTIG_BLOCK_TOO_BIG ),
	         "Attempted to allocate contiguous memory block that is too big for implementation." );
	sprintf( bli_error_string_for_code(BLIS_EXHAUSTED_CONTIG_MEMORY_POOL),
	         "Attempted to allocate more memory from contiguous pool than is available." );

	sprintf( bli_error_string_for_code(BLIS_EXPECTED_OBJECT_ALIAS),
	         "Expected object to be alias." );
}

void bli_print_msg( char* str, char* file, guint_t line )
{
	fprintf( stderr, "\n" );
	fprintf( stderr, "libblis: %s (line %lu):\n", file, ( long unsigned int )line );
	fprintf( stderr, "libblis: %s\n", str );
	fflush( stderr );
}

void bli_abort( void )
{
	fprintf( stderr, "libblis: Aborting.\n" );
	//raise( SIGABRT );
	abort();
}

// -----------------------------------------------------------------------------

// Current error checking level.
static errlev_t bli_err_chk_level = BLIS_FULL_ERROR_CHECKING;

errlev_t bli_error_checking_level( void )
{
    return bli_err_chk_level;
}

errlev_t bli_error_checking_level_set( errlev_t new_level )
{
    err_t    e_val;
    errlev_t old_level;

    e_val = bli_check_valid_error_level( new_level );
    bli_check_error_code( e_val );

    old_level = bli_err_chk_level;

    bli_err_chk_level = new_level;

    return old_level;
}

bool_t bli_error_checking_is_enabled( void )
{
    return bli_error_checking_level() != BLIS_NO_ERROR_CHECKING;
}

char* bli_error_string_for_code( gint_t code )
{
	return bli_error_string[-code];
}

