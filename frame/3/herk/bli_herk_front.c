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

void bli_herk_front( obj_t*  alpha,
                     obj_t*  a,
                     obj_t*  beta,
                     obj_t*  c,
                     gemm_t* cntl )
{
	obj_t   a_local;
	obj_t   ah_local;
	obj_t   c_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_herk_check( alpha, a, beta, c );

	// If alpha is zero, scale by beta, zero the imaginary components of
	// the diagonal elements, and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		bli_setid( &BLIS_ZERO, c );
		return;
	}

	// Alias A and C in case we need to apply transformations.
	bli_obj_alias_to( *a, a_local );
	bli_obj_alias_to( *c, c_local );
	bli_obj_set_as_root( c_local );

	// For herk, the right-hand "B" operand is simply A'.
	bli_obj_alias_to( *a, ah_local );
	bli_obj_induce_trans( ah_local );
	bli_obj_toggle_conj( ah_local );

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	if (
	     ( bli_obj_is_row_stored( c_local ) &&
	       bli_func_prefers_contig_cols( bli_obj_datatype( c_local ),
	                                     bli_gemm_cntl_ukrs( cntl ) ) ) ||
	     ( bli_obj_is_col_stored( c_local ) &&
	       bli_func_prefers_contig_rows( bli_obj_datatype( c_local ),
	                                     bli_gemm_cntl_ukrs( cntl ) ) )
	   )
	{
		bli_obj_toggle_conj( a_local );
		bli_obj_toggle_conj( ah_local );

		bli_obj_induce_trans( c_local );
	}

    herk_thrinfo_t** infos = bli_create_herk_thrinfo_paths();
    dim_t n_threads = thread_num_threads( infos[0] );

    // Invoke the internal back-end.
    bli_level3_thread_decorator( n_threads,   
                                 (level3_int_t) bli_herk_int, 
                                 alpha, 
                                 &a_local,  
                                 &ah_local,  
                                 beta, 
                                 &c_local,  
                                 (void*) cntl, 
                                 (void**) infos );

    bli_herk_thrinfo_free_paths( infos, n_threads );

	// The Hermitian rank-k product was computed as A*A', even for the
	// diagonal elements. Mathematically, the imaginary components of
	// diagonal elements of a Hermitian rank-k product should always be
	// zero. However, in practice, they sometimes accumulate meaningless
	// non-zero values. To prevent this, we explicitly set those values
	// to zero before returning.
	bli_setid( &BLIS_ZERO, &c_local );

}

