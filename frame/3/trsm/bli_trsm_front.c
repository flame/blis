/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2020, Advanced Micro Devices, Inc.

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
//#define PRINT_SMALL_TRSM_INFO

void bli_trsm_front
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl
     )
{
	bli_init_once();

	obj_t   a_local;
	obj_t   b_local;
	obj_t   c_local;
 
#ifdef PRINT_SMALL_TRSM_INFO
        printf("Side:: %c\n", side ? 'R' : 'L');
        if (bli_obj_datatype(*a) == BLIS_FLOAT)
                        printf("Alpha:: %9.2e\n", *((float *)bli_obj_buffer_for_const(BLIS_FLOAT, *alpha)));
        else if (bli_obj_datatype(*a) == BLIS_DOUBLE)
                        printf("Alpha is double:: %9.2e\n", *((double *)bli_obj_buffer_for_const(BLIS_DOUBLE, *alpha)));
        else
                        printf("Unsupported datatype for Alpha\n");

        printf("A:: M = %d, N = %d, elem_size = %d, row_off = %ld, col_off = %ld, rs = %d, cs = %d, trans = %c, TRIANG = %c, unit diag = %c\n", a->dim[0], a->dim[1], bli_obj_elem_size(*a ), bli_obj_row_off(*a), bli_obj_col_off(*a), a->rs, a->cs, bli_obj_has_trans(*a) ? 'Y' : 'N', bli_obj_is_upper(*a) ? 'U' : bli_obj_is_lower(*a) ? 'L' : 'N', bli_obj_has_unit_diag(*a) ? 'Y' : 'N');
#ifdef PRINT_SMALL_TRSM
        //bli_printm("a", a, "%4.1f", "");
#endif
        printf("B:: M = %d, N = %d, elem_size = %d, row_off = %ld, col_off = %ld, rs = %d, cs = %d, trans = %c\n", b->dim[0], b->dim[1], bli_obj_elem_size(*a ), bli_obj_row_off(*a), bli_obj_col_off(*a), b->rs, b->cs, bli_obj_has_trans(*b) ? 'Y' : 'N');
#ifdef PRINT_SMALL_TRSM
        //bli_printm("b", b, "%4.1f", "");
#endif
        fflush(stdout);
#endif
#if 0
for (i = 0; i < m; i++) //no. of cols of B
{
	for (j = 0; j < n; j++) //no. of rows of B
	{
		B[i*n + j] = 1001 + j + (i*n);
	}
}
for (i = 0; i < m; i++) //no. of cols of B
{
	for (j = i; j < m; j++) //no. of rows of B
	{
		L[i*m + j] = 2001 + j + (i*m);
	}
}
#endif
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
	gint_t status = bli_trsm_small( side, alpha, a, b, cntx, cntl );
	if ( status == BLIS_SUCCESS ) return;
#endif

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_trsm_check( side, alpha, a, b, &BLIS_ZERO, b, cntx );

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( alpha, b );
		return;
	}

	// Alias A and B so we can tweak the objects if necessary.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( b, &c_local );

	// We do not explicitly implement the cases where A is transposed.
	// However, we can still handle them. Specifically, if A is marked as
	// needing a transposition, we simply induce a transposition. This
	// allows us to only explicitly implement the no-transpose cases. Once
	// the transposition is induced, the correct algorithm will be called,
	// since, for example, an algorithm over a transposed lower triangular
	// matrix A moves in the same direction (forwards) as a non-transposed
	// upper triangular matrix. And with the transposition induced, the
	// matrix now appears to be upper triangular, so the upper triangular
	// algorithm will grab the correct partitions, as if it were upper
	// triangular (with no transpose) all along.
	if ( bli_obj_has_trans( &a_local ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

#if 1

	// If A is being solved against from the right, transpose all operands
	// so that we can perform the computation as if A were being solved
	// from the left.
	if ( bli_is_right( side ) )
	{
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#else

	// NOTE: Enabling this code requires that BLIS NOT be configured with
	// BLIS_RELAX_MCNR_NCMR_CONSTRAINTS defined.
#ifdef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	#error "BLIS_RELAX_MCNR_NCMR_CONSTRAINTS must not be defined for current trsm_r implementation."
#endif

	// If A is being solved against from the right, swap A and B so that
	// the triangular matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

#endif

	// Set each alias as the root object.
	// NOTE: We MUST wait until we are done potentially swapping the objects
	// before setting the root fields!
	bli_obj_set_as_root( &a_local );
	bli_obj_set_as_root( &b_local );
	bli_obj_set_as_root( &c_local );

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	bli_rntm_set_ways_for_op
	(
	  BLIS_TRSM,
	  side,
	  bli_obj_length( &c_local ),
	  bli_obj_width( &c_local ),
	  bli_obj_width( &a_local ),
	  rntm
	);

	// A sort of hack for communicating the desired pach schemas for A and B
	// to bli_trsm_cntl_create() (via bli_l3_thread_decorator() and
	// bli_l3_cntl_create_if()). This allows us to access the schemas from
	// the control tree, which hopefully reduces some confusion, particularly
	// in bli_packm_init().
	if ( bli_cntx_method( cntx ) == BLIS_NAT )
	{
		bli_obj_set_pack_schema( BLIS_PACKED_ROW_PANELS, &a_local );
		bli_obj_set_pack_schema( BLIS_PACKED_COL_PANELS, &b_local );
	}
	else // if ( bli_cntx_method( cntx ) != BLIS_NAT )
	{
		pack_t schema_a = bli_cntx_schema_a_block( cntx );
		pack_t schema_b = bli_cntx_schema_b_panel( cntx );

		bli_obj_set_pack_schema( schema_a, &a_local );
		bli_obj_set_pack_schema( schema_b, &b_local );
	}

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  bli_trsm_int,
	  BLIS_TRSM, // operation family id
	  alpha,
	  &a_local,
	  &b_local,
	  alpha,
	  &c_local,
	  cntx,
	  rntm,
	  cntl
	);
}

