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

void bli_l3_int
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const cntl_t* cntl,
             thrinfo_t* thread
     )
{
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;

	// Return early if the current control tree node is NULL.
	if ( bli_cntl_is_null( cntl ) ) return;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_basic_check( alpha, a, b, beta, c, cntx );

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) )
	{
		return;
	}

	// If A or B has a zero dimension, scale C by beta and return early.
	if ( bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		if ( bli_thrinfo_am_chief( thread ) )
			bli_scalm( beta, c );
		bli_thrinfo_barrier( thread );
		return;
	}

	// If A or B is marked as being filled with zeros, scale C by beta and
	// return early.
	if ( bli_obj_is_zeros( a ) ||
	     bli_obj_is_zeros( b ) )
	{
		// This should never execute.
		bli_abort();

		if ( bli_thrinfo_am_chief( thread ) )
			bli_scalm( beta, c );
		bli_thrinfo_barrier( thread );
		return;
	}

	// Alias A, B, and C in case we need to update attached scalars.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( c, &c_local );

	// Ensure that a valid packing function is set on A and B.
	if ( !bli_obj_pack_fn( &a_local ) )
		bli_obj_set_pack_fn( bli_packm_blk_var1, &a_local );

	if ( !bli_obj_pack_fn( &b_local ) )
		bli_obj_set_pack_fn( bli_packm_blk_var1, &b_local );

	// If we are about to call a leaf-level implementation, and matrix C
	// still needs a transposition, then we must induce one by swapping the
	// strides and dimensions. Note that this transposition would normally
	// be handled explicitly in the packing of C, but if C is not being
	// packed, this is our last chance to handle the transposition.
	//if ( bli_cntl_is_leaf( cntl ) && bli_obj_has_trans( c ) )
	if ( bli_obj_has_trans( c ) )
	{
		bli_obj_induce_trans( &c_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &c_local );
	}

	// If alpha is non-unit, typecast and apply it to the scalar attached
	// to B, unless it happens to be triangular.
	if ( bli_obj_root_is_triangular( b ) )
	{
		if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( alpha, &a_local );
	}
	else // if ( bli_obj_root_is_triangular( b ) )
	{
		if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( alpha, &b_local );
	}

	// If beta is non-unit, typecast and apply it to the scalar attached
	// to C.
	if ( !bli_obj_equals( beta, &BLIS_ONE ) )
		bli_obj_scalar_apply_scalar( beta, &c_local );

	// Extract the function pointer from the current control tree node.
	l3_var_oft f = bli_cntl_var_func( cntl );

	// Invoke the variant.
	f
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  cntl,
	  thread
	);
}

