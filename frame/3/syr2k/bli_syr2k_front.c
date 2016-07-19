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

void bli_syr2k_front( obj_t*  alpha,
                      obj_t*  a,
                      obj_t*  b,
                      obj_t*  beta,
                      obj_t*  c,
                      cntx_t* cntx,
                      gemm_t* cntl )
{
	obj_t    c_local;
	obj_t    a_local;
	obj_t    bt_local;
	obj_t    b_local;
	obj_t    at_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_syr2k_check( alpha, a, b, beta, c, cntx );

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// Reinitialize the memory allocator to accommodate the blocksizes
	// in the current context.
	bli_mem_reinit( cntx );

	// Alias A, B, and C in case we need to apply transformations.
	bli_obj_alias_to( *a, a_local );
	bli_obj_alias_to( *b, b_local );
	bli_obj_alias_to( *c, c_local );
	bli_obj_set_as_root( c_local );

	// For syr2k, the first and second right-hand "B" operands are simply B'
	// and A'.
	bli_obj_alias_to( *b, bt_local );
	bli_obj_induce_trans( bt_local );
	bli_obj_alias_to( *a, at_local );
	bli_obj_induce_trans( at_local );

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	if ( bli_cntx_l3_nat_ukr_dislikes_storage_of( &c_local, BLIS_GEMM_UKR, cntx ) )
	{
		bli_obj_induce_trans( c_local );
	}

#if 0
	// Invoke the internal back-end.
	bli_her2k_int( alpha,
	               &a_local,
	               &bt_local,
	               alpha,
	               &b_local,
	               &at_local,
	               beta,
	               &c_local,
	               cntl );
#else
	// Invoke herk twice, using beta only the first time.
    thrinfo_t** infos = bli_l3_thrinfo_create_paths( BLIS_SYR2K, BLIS_LEFT );
    dim_t n_threads = bli_thread_num_threads( infos[0] );

    // Invoke the internal back-end.
    bli_l3_thread_decorator( n_threads,
                                 (l3_int_t) bli_herk_int, 
                                 alpha, 
                                 &a_local,  
                                 &bt_local,  
                                 beta, 
                                 &c_local,  
                                 (void*) cntx, 
                                 (void*) cntl, 
                                 (void**) infos );

    bli_l3_thread_decorator( n_threads,
                                 (l3_int_t) bli_herk_int, 
                                 alpha, 
                                 &b_local,  
                                 &at_local,  
                                 &BLIS_ONE, 
                                 &c_local,  
                                 (void*) cntx, 
                                 (void*) cntl, 
                                 (void**) infos );

    bli_l3_thrinfo_free_paths( infos, n_threads );
#endif

}

