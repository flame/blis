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

void bli_hemm_front
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             rntm_t* rntm
     )
{
	bli_init_once();

	obj_t   a_local;
	obj_t   b_local;
	obj_t   c_local;

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// Alias A, B, and C in case we need to apply transformations.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( c, &c_local );

	// Set the obj_t buffer field to the location currently implied by the row
	// and column offsets and then zero the offsets. If any of the original
	// obj_t's were views into larger matrices, this step effectively makes
	// those obj_t's "forget" their lineage.
	bli_obj_reset_origin( &a_local );
	bli_obj_reset_origin( &b_local );
	bli_obj_reset_origin( &c_local );

#ifdef BLIS_DISABLE_HEMM_RIGHT
	// NOTE: This case casts right-side hemm in terms of left side. This is
	// necessary when the current subconfiguration uses a gemm microkernel
	// that assumes that the packing kernel will have already duplicated
	// (broadcast) element of B in the packed copy of B. Supporting
	// duplication within the logic that packs micropanels from Hermitian/
	// matrices would be ugly, and so we simply don't support it. As a
	// consequence, those subconfigurations need a way to force the Hermitian
	// matrix to be on the left (and thus the general matrix to the on the
	// right). So our solution is that in those cases, the subconfigurations
	// simply #define BLIS_DISABLE_HEMM_RIGHT.

	// NOTE: This case casts right-side hemm in terms of left side. This can
	// lead to the microkernel being executed on an output matrix with the
	// microkernel's general stride IO case (unless the microkernel supports
	// both both row and column IO cases as well).

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

#else
	// NOTE: This case computes right-side hemm/symm natively by packing
	// elements of the Hermitian/symmetric matrix A to micropanels of the
	// right-hand packed matrix operand "B", and elements of the general
	// matrix B to micropanels of the left-hand packed matrix operand "A".
	// This code path always gives us the opportunity to transpose the
	// entire operation so that the effective storage format of the output
	// matrix matches the microkernel's output preference. Thus, from a
	// performance perspective, this case is preferred.

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	//if ( !bli_obj_is_1x1( &c_local ) ) // NOTE: This conditional should NOT
	                                     // be enabled. See issue #342 comments.
	if ( bli_cntx_dislikes_storage_of( &c_local, BLIS_GEMM_VIR_UKR, cntx ) )
	{
		bli_toggle_side( &side );
		bli_obj_toggle_conj( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

	// If the Hermitian/symmetric matrix A is being multiplied from the right,
	// swap A and B so that the Hermitian/symmetric matrix will actually be on
	// the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}
#endif

	// Set the pack schemas within the objects.
	bli_l3_set_schemas( &a_local, &b_local, &c_local, cntx );

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	bli_rntm_set_ways_for_op
	(
	  BLIS_HEMM,
	  BLIS_LEFT, // ignored for gemm/hemm/symm
	  bli_obj_length( &c_local ),
	  bli_obj_width( &c_local ),
	  bli_obj_width( &a_local ),
	  rntm
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  bli_l3_int,
	  BLIS_GEMM, // operation family id
	  alpha,
	  &a_local,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  rntm
	);
}

