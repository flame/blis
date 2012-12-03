/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#define FUNCPTR_T her2k_fp

typedef void (*FUNCPTR_T)( obj_t*   alpha,
                           obj_t*   a,
                           obj_t*   bh,
                           obj_t*   alpha_conj,
                           obj_t*   b,
                           obj_t*   ah,
                           obj_t*   beta,
                           obj_t*   c,
                           her2k_t* cntl );

static FUNCPTR_T vars[2][5][3] =
{
	// lower
	{
		// unblocked          optimized unblocked    blocked
		{ NULL,               NULL,                  bl2_her2k_l_blk_var1 },
		{ NULL,               bl2_her2k_l_ker_var2,  bl2_her2k_l_blk_var2 },
		{ NULL,               NULL,                  bl2_her2k_blk_var3   },
		{ NULL,               NULL,                  bl2_her2k_l_blk_var4 },
		{ NULL,               NULL,                  NULL                 },
	},
	// upper
	{
		// unblocked          optimized unblocked    blocked
		{ NULL,               NULL,                  bl2_her2k_u_blk_var1 },
		{ NULL,               bl2_her2k_u_ker_var2,  bl2_her2k_u_blk_var2 },
		{ NULL,               NULL,                  bl2_her2k_blk_var3   },
		{ NULL,               NULL,                  bl2_her2k_u_blk_var4 },
		{ NULL,               NULL,                  NULL                 },
	}
};

void bl2_her2k_int( obj_t*   alpha,
                    obj_t*   a,
                    obj_t*   bh,
                    obj_t*   alpha_conj,
                    obj_t*   b,
                    obj_t*   ah,
                    obj_t*   beta,
                    obj_t*   c,
                    her2k_t* cntl )
{
	obj_t     c_local;
	varnum_t  n;
	impl_t    i;
	bool_t    uplo;
	FUNCPTR_T f;

	// Check parameters.
	if ( bl2_error_checking_is_enabled() )
		bl2_her2k_int_check( alpha, a, bh, alpha_conj, b, ah, beta, c, cntl );

	// Return early if one of the matrix operands has a zero dimension.
	if ( bl2_obj_has_zero_dim( *a  ) ) return;
	if ( bl2_obj_has_zero_dim( *bh ) ) return;
	if ( bl2_obj_has_zero_dim( *b  ) ) return;
	if ( bl2_obj_has_zero_dim( *ah ) ) return;
	if ( bl2_obj_has_zero_dim( *c  ) ) return;

	// Alias C in case we need to induce a transposition.
	bl2_obj_alias_to( *c, c_local );

	// If we are about to call a leaf-level implementation, and matrix C
	// still needs a transposition, then we must induce one by swapping the
	// strides and dimensions. Note that this transposition would normally
	// be handled explicitly in the packing of C, but if C is not being
	// packed, this is our last chance to handle the transposition.
	if ( cntl_is_leaf( cntl ) && bl2_obj_has_trans( *c ) )
		bl2_obj_induce_trans( c_local );

	// Set a bool based on the uplo field of c.
	if ( bl2_obj_root_is_lower( c_local ) ) uplo = 0;
	else                                    uplo = 1;

	// Extract the variant number and implementation type.
	n = cntl_var_num( cntl );
	i = cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[uplo][n][i];

	// Invoke the variant.
	f( alpha,
	   a,
	   bh,
	   alpha_conj,
	   b,
	   ah,
	   beta,
	   &c_local,
	   cntl );
}

