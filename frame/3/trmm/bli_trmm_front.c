/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

void bli_trmm_front
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

	// Set the obj_t buffer field to the location currently implied by the row
	// and column offsets and then zero the offsets. If any of the original
	// obj_t's were views into larger matrices, this step effectively makes
	// those obj_t's "forget" their lineage.
	bli_obj_reset_origin( &a_local );
	bli_obj_reset_origin( &b_local );
	bli_obj_reset_origin( &c_local );

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

#ifdef BLIS_DISABLE_TRMM_RIGHT
	// NOTE: This case casts right-side trmm in terms of left side. This is
	// necessary when the current subconfiguration uses a gemm microkernel
	// that assumes that the packing kernel will have already duplicated
	// (broadcast) element of B in the packed copy of B. Supporting
	// duplication within the logic that packs micropanels from triangular
	// matrices would be ugly, and so we simply don't support it. As a
	// consequence, those subconfigurations need a way to force the triangular
	// matrix to be on the left (and thus the general matrix to the on the
	// right). So our solution is that in those cases, the subconfigurations
	// simply #define BLIS_DISABLE_TRMM_RIGHT.

	// On dual-socket architectures, disabling trmm right is needed for a
	// different reason. Right-side TRMM forces Jc to be 1, so casting it to a
	// left-side operation allows for dramatically improved performance due to
	// independent parallelization on the two sockets.

	// NOTE: This case casts right-side trmm in terms of left side. This can
	// lead to the microkernel being executed on an output matrix with the
	// microkernel's general stride IO case (unless the microkernel supports
	// both both row and column IO cases as well).

	// NOTE: Casting right-side trmm in terms of left side reduces the number
	// of macrokernels exercised to two (trmm_ll and trmm_lu).

	// If A is being multiplied from the right, transpose all operands
	// so that we can perform the computation as if A were being multiplied
	// from the left.
	if ( bli_is_right( side ) )
	{
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#else /* not BLIS_DISABLE_TRMM_RIGHT */
#ifdef BLIS_DISABLE_TRMM_LEFT
	// NOTE: This case casts left-side trmm in terms of right side. This can
	// lead to the microkernel being executed on an output matrix with the
	// microkernel's general stride IO case (unless the microkernel supports
	// both both row and column IO cases as well).

	// NOTE: Casting left-side trmm in terms of right side reduces the number
	// of macrokernels exercised to two (trmm_rl and trmm_ru).

	// If A is being multiplied from the left, transpose all operands
	// so that we can perform the computation as if A were being multiplied
	// from the right.
	if ( bli_is_left( side ) )
	{
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#else /* not BLIS_DISABLE_TRMM_LEFT */
#ifdef BLIS_DISABLE_TRMM_RIGHT_IF_JC_GT_1_ELSE_DISABLE_LEFT_IF_DP

	// This case was added for the Ampere platforms.
	
	// As noted above, for dual socket (Jc > 1), disable trmm right
	// dramatically improves performance by avoiding the forced Jc=1 for
	// right-side trmm.
	
	// On the other hand, for single socket double-precision trmm (where we
	// already have Jc = 1), performance is significantly improved by
	// disabling trmm left and forcing a transpose to a right-side operation.

	bool toggle = FALSE;
	dim_t jc = bli_rntm_jc_ways( rntm );
	if (jc > 1) {
		// Presume dual socket, disable trmm right
		if ( bli_is_right( side ) ) { toggle = TRUE; }
	} else {
		// Single socket
		bool dp = bli_obj_is_double_prec( &a_local ) && bli_obj_is_double_prec( &b_local );
		if (dp) {
			// Double precision, disable trmm left
			if ( bli_is_left( side ) ) { toggle = TRUE; }
		} else {
			// As in the default case (below), toggle for preferential storage
			if ( bli_cntx_dislikes_storage_of( &c_local, BLIS_GEMM_VIR_UKR, cntx ) ) {
				toggle = TRUE;
			}
		}
	}

	if (toggle) {
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}
	
#else /* not BLIS_DISABLE_TRMM_RIGHT_IF_JC_GT_1_ELSE_DISABLE_LEFT_IF_DP */

	// The default case
	
	// NOTE: This case computes right-side trmm natively with trmm_rl and
	// trmm_ru macrokernels. This code path always gives us the opportunity
	// to transpose the entire operation so that the effective storage format
	// of the output matrix matches the microkernel's output preference.
	// Thus, from a performance perspective, this case is preferred.

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	// NOTE: We disable the optimization for 1x1 matrices since the concept
	// of row- vs. column storage breaks down.
	//if ( !bli_obj_is_1x1( &c_local ) ) // NOTE: This conditional should NOT
	                                     // be enabled. See issue #342 comments.
	if ( bli_cntx_dislikes_storage_of( &c_local, BLIS_GEMM_VIR_UKR, cntx ) )
	{
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#endif /* BLIS_DISABLE_TRMM_RIGHT_IF_JC_GT_1_ELSE_DISABLE_LEFT_IF_DP */
#endif /* BLIS_DISABLE_TRMM_LEFT */
#endif /* BLIS_DISABLE_TRMM_RIGHT */

	// If A is being multiplied from the right, swap A and B so that
	// the matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

	// Set the pack schemas within the objects.
	bli_l3_set_schemas( &a_local, &b_local, &c_local, cntx );

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	bli_rntm_set_ways_for_op
	(
	  BLIS_TRMM,
	  side,
	  bli_obj_length( &c_local ),
	  bli_obj_width( &c_local ),
	  bli_obj_width( &a_local ),
	  rntm
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  bli_l3_int,
	  BLIS_TRMM, // operation family id
	  alpha,
	  &a_local,
	  &b_local,
	  &BLIS_ZERO,
	  &c_local,
	  cntx,
	  rntm,
	  cntl
	);
}

